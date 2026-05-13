#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HTTP router for cross-engine SLO-aware LLM serving.

Architecture:
  client --POST /v1/completions--> Router (this process, CPU-only)
                                     │
                                     ├ reads /dev/shm/vllm_ft_engine_status/engine_*.json
                                     │   to track which engines are alive + their load
                                     │
                                     ├ on dispatch: picks alive engine with lowest
                                     │   (running + waiting); forwards via httpx
                                     │   with X-Request-Id header so the engine's
                                     │   internal request_id is keyed on a value the
                                     │   router controls
                                     │
                                     └ on engine death (status ts older than threshold,
                                       or forward connection error): reads internal_req_id
                                       from /dev/shm/vllm_ft_req_map/<user_req_id>, picks
                                       another alive engine, re-forwards the original body
                                       with `vllm_xargs = {is_rerouted: true,
                                       num_checkpointed_tokens: ...}` so the new engine
                                       routes through the V3 restore path

Does NOT support streaming responses. Targets paper experiments where
non-streaming OpenAI completions are sufficient.
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

SHM_ENGINE_STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
SHM_REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
SHM_PREEMPT_QUEUE_DIR = Path("/dev/shm/vllm_ft_preempt_queue")
STATUS_POLL_INTERVAL_S = 0.5
# Preempt queue poll cadence — kept tighter than status poll because the
# engine's redispatch fallback timer is on the order of seconds; router
# needs to react well before that fires to avoid double-resume.
PREEMPT_POLL_INTERVAL_S = 0.2
STATUS_STALE_THRESHOLD_S = 2.0
ENGINE_REQUEST_TIMEOUT_S = 600.0

logger = logging.getLogger("router")


@dataclass
class EngineState:
    engine_id: int
    base_url: str
    alive: bool = False
    last_ts: float = 0.0
    running: int = 0
    waiting: int = 0
    kv_usage: float = 0.0

    @property
    def load_score(self) -> int:
        return self.running + self.waiting


@dataclass
class InFlightReq:
    user_req_id: str
    engine_id: int
    endpoint: str
    body: dict
    # The asyncio.Task running the forward() call to the engine. The
    # preempt-queue poller cancels this to tear down the connection
    # when an engine asks for cross-engine redispatch; cancellation
    # propagates into httpx which closes the underlying TCP, vllm sees
    # client disconnect and aborts the local copy.
    task: "asyncio.Task | None" = None
    # Set by the preempt-queue poller right before it cancels `task`.
    # Distinguishes "engine asked us to redispatch" from "client gave up
    # and cancelled". On the former we reroute; on the latter we don't.
    redispatch_requested: bool = False


class Router:
    def __init__(self, engines: dict[int, str]):
        self.engines: dict[int, EngineState] = {
            eid: EngineState(engine_id=eid, base_url=url)
            for eid, url in engines.items()
        }
        self.in_flight: dict[str, InFlightReq] = {}
        self.dispatch_count = 0
        self.reroute_count = 0
        self.dead_event_count = 0
        self.client = httpx.AsyncClient(timeout=ENGINE_REQUEST_TIMEOUT_S)

    async def close(self) -> None:
        await self.client.aclose()

    # ── Engine status / health ─────────────────────────────────────

    def scan_engine_status(self) -> None:
        """Poll every status file once. Updates engine load + alive flag."""
        now = time.time()
        if not SHM_ENGINE_STATUS_DIR.exists():
            # Engines haven't started writing yet; nothing to scan.
            return
        for fp in SHM_ENGINE_STATUS_DIR.glob("engine_*.json"):
            try:
                data = json.loads(fp.read_text())
            except (OSError, json.JSONDecodeError):
                # Engine mid-write or unreadable; skip this round.
                continue
            eid = data.get("engine_id")
            engine = self.engines.get(eid)
            if engine is None:
                continue
            engine.last_ts = float(data.get("ts", 0))
            engine.running = int(data.get("running", 0))
            engine.waiting = int(data.get("waiting", 0))
            engine.kv_usage = float(data.get("kv_usage", 0.0))
            was_alive = engine.alive
            engine.alive = (now - engine.last_ts) < STATUS_STALE_THRESHOLD_S
            if was_alive and not engine.alive:
                self.dead_event_count += 1
                logger.warning(
                    "engine %d marked DEAD (last status ts=%.3f, now=%.3f, "
                    "stale=%.2fs)",
                    eid, engine.last_ts, now, now - engine.last_ts,
                )

    def pick_engine(self, exclude: int | None = None) -> EngineState | None:
        """Pick alive engine with lowest load using a layered tuple sort.

        Primary key: router-side in_flight count (realtime, same-process
        dict). This is what reflects "how many requests this engine has
        on its plate right now"; shm status is stale up to 200ms and
        concurrent dispatches before the first refresh would otherwise
        all land on engine 0.

        Tie-breakers (only used when in_flight counts are equal):
          1. kv_usage  — engine with lower KV pressure preferred
          2. waiting   — engine with fewer queued (KV-blocked) reqs

        These come from shm. They capture engine-internal state the
        router can't infer from in_flight alone (which only knows the
        total count, not the running-vs-waiting split or KV pressure).
        """
        alive = [
            e for e in self.engines.values()
            if e.alive and e.engine_id != exclude
        ]
        if not alive:
            return None
        local_count: dict[int, int] = {e.engine_id: 0 for e in alive}
        for in_f in self.in_flight.values():
            if in_f.engine_id in local_count:
                local_count[in_f.engine_id] += 1
        return min(
            alive,
            key=lambda e: (
                local_count[e.engine_id],
                e.kv_usage,
                e.waiting,
            ),
        )

    # ── Preempt queue (cross-engine redispatch on picker preempt) ──

    def scan_preempt_queue(self) -> None:
        """Poll /dev/shm/vllm_ft_preempt_queue/*.json. For each entry:
          - read engine A's payload (router_req_id, internal_req_id,
            num_checkpointed_tokens, preempt_ts)
          - look up the in_flight entry on router side
          - if we have it AND there's another alive engine to take
            over: unlink the shm file (so engine A won't see it again
            after its abort), mark in_f.redispatch_requested, cancel
            the forward task so proxy's CancelledError handler kicks
            off the reroute path
          - otherwise leave the file (engine A's fallback timer will
            convert to local V3 reload after FT_REDISPATCH_TIMEOUT_S)

        Robust to partial writes / vanished entries.
        """
        if not SHM_PREEMPT_QUEUE_DIR.exists():
            return
        for fp in SHM_PREEMPT_QUEUE_DIR.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
            except (OSError, json.JSONDecodeError):
                # Mid-write or unlink raced; skip this round.
                continue
            router_req_id = data.get("router_req_id") or ""
            if not router_req_id:
                # Engine A published an entry without router_req_id —
                # likely a direct-vllm (no router) test. Nothing for
                # router to do; engine A's timeout will handle it.
                continue
            in_f = self.in_flight.get(router_req_id)
            if in_f is None:
                # Two cases: (a) the request already finished and we
                # popped it from in_flight, (b) it was a direct vllm
                # call. Either way, leave the entry; engine A cleans
                # it on its own (FINISHED_* path or timeout path).
                continue
            origin_eid = data.get("engine_id")
            if self.pick_engine(exclude=origin_eid) is None:
                # No alternative engine — let engine A's local fallback
                # handle it.
                continue
            try:
                fp.unlink()
            except OSError:
                pass
            in_f.redispatch_requested = True
            if in_f.task is not None and not in_f.task.done():
                in_f.task.cancel()
            logger.info(
                "preempt-redispatch: signaled req=%s (origin engine "
                "%s, ckpt_tokens=%s) for reroute",
                router_req_id, origin_eid,
                data.get("num_checkpointed_tokens"),
            )

    # ── Req map (user_req_id → internal_req_id) ────────────────────

    @staticmethod
    def read_internal_req_id(user_req_id: str) -> str | None:
        """Read /dev/shm/vllm_ft_req_map/<user_req_id> and return the
        internal_req_id the engine assigned (or None if not present yet).
        """
        fp = SHM_REQ_MAP_DIR / user_req_id
        try:
            data = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return data.get("internal_req_id")

    @staticmethod
    def read_checkpoint_progress(internal_req_id: str) -> int | None:
        """Read /dev/shm/vllm_ft_checkpoints/<internal_req_id>/latest_rank0
        and the manifest it points to, return covered_tokens. None if the
        request never published anything (or files unreadable).

        Used at reroute time: the new engine's V3 reload state machine
        needs to know how many tokens of KV the original engine managed
        to publish so it can allocate that many blocks + restore them.
        """
        ckpt_dir = Path("/dev/shm/vllm_ft_checkpoints") / internal_req_id
        latest_fp = ckpt_dir / "latest_rank0"
        try:
            manifest_filename = latest_fp.read_text().strip()
        except OSError:
            return None
        if not manifest_filename:
            return None
        try:
            manifest = json.loads((ckpt_dir / manifest_filename).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        ct = manifest.get("covered_tokens")
        return int(ct) if ct is not None else None

    # ── Forwarding ─────────────────────────────────────────────────

    async def forward(
        self,
        target: EngineState,
        endpoint: str,
        body: dict,
        user_req_id: str,
    ) -> httpx.Response:
        """POST body to target engine. Caller handles errors.

        The router-side id is delivered to the engine via
        body["vllm_xargs"]["router_req_id"] (not via X-Request-Id, which
        vllm's OpenAI handler mangles with "cmpl-..."/-0 prefixes that the
        router can't predict). The engine writes the req_map file keyed
        on router_req_id so the router can find it later for reroute.
        """
        url = target.base_url.rstrip("/") + endpoint
        body_with_router_id = dict(body)
        vllm_xargs = dict(body_with_router_id.get("vllm_xargs") or {})
        # Router owns router_req_id — it's the key for in_flight tracking
        # AND for /dev/shm/vllm_ft_req_map/<id> lookup on reroute. Any
        # router_req_id the client supplied must be ignored, otherwise
        # in_flight (keyed by router's id) and req_map (keyed by the
        # client's id) get out of sync and reroute can't find the engine's
        # checkpoint metadata.
        vllm_xargs["router_req_id"] = user_req_id
        body_with_router_id["vllm_xargs"] = vllm_xargs
        return await self.client.post(
            url,
            json=body_with_router_id,
            headers={"Content-Type": "application/json"},
        )

    async def proxy(self, endpoint: str, body: dict) -> JSONResponse:
        """Dispatch → forward → on engine failure or preempt-redispatch
        signal, reroute once.

        The forward call is wrapped in an asyncio.Task so the preempt-
        queue poller can cancel it externally. Two cancellation paths
        are distinguished:
          - in_f.redispatch_requested=True : engine A's picker asked
            for cross-engine pickup. We reroute to another engine.
          - otherwise : caller (client) cancelled. We propagate the
            CancelledError instead of pretending the request succeeded.
        """
        target = self.pick_engine()
        if target is None:
            raise HTTPException(503, "no alive engine to dispatch to")

        user_req_id = uuid.uuid4().hex
        in_f = InFlightReq(
            user_req_id=user_req_id,
            engine_id=target.engine_id,
            endpoint=endpoint,
            body=body,
        )
        self.in_flight[user_req_id] = in_f
        self.dispatch_count += 1

        try:
            forward_task = asyncio.create_task(
                self.forward(target, endpoint, body, user_req_id)
            )
            in_f.task = forward_task
            try:
                resp = await forward_task
                return JSONResponse(
                    content=resp.json(),
                    status_code=resp.status_code,
                    headers={"X-Request-Id": user_req_id},
                )
            except asyncio.CancelledError:
                if in_f.redispatch_requested:
                    logger.info(
                        "preempt-redispatch: engine %d asked router "
                        "to redispatch req=%s",
                        target.engine_id, user_req_id,
                    )
                    return await self.reroute(user_req_id)
                # Client-side cancel — propagate.
                raise
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.ReadError,
            ) as e:
                logger.warning(
                    "engine %d forward failed for req=%s: %s. "
                    "Attempting reroute.",
                    target.engine_id, user_req_id, e,
                )
                return await self.reroute(user_req_id)
        finally:
            self.in_flight.pop(user_req_id, None)

    async def reroute(self, user_req_id: str) -> JSONResponse:
        """Re-forward an in-flight request to a different alive engine,
        passing the original engine's internal_req_id via vllm_xargs so
        the new engine routes through the V3 restore path.
        """
        original = self.in_flight.get(user_req_id)
        if original is None:
            raise HTTPException(500, "in_flight entry vanished mid-reroute")

        new_target = self.pick_engine(exclude=original.engine_id)
        if new_target is None:
            raise HTTPException(
                503, "no alive engine to reroute to (all dead)"
            )

        internal_req_id = self.read_internal_req_id(user_req_id)
        ckpt_tokens = (
            self.read_checkpoint_progress(internal_req_id)
            if internal_req_id is not None else None
        )

        # Augment body: vllm_xargs carries reroute signal so the engine's
        # EngineCore.add_request lifts it into Request fields and diverts
        # the request into the V3 reload state machine.
        new_body = dict(original.body)
        vllm_xargs = dict(new_body.get("vllm_xargs") or {})
        vllm_xargs["is_rerouted"] = True
        if internal_req_id is not None:
            vllm_xargs["original_internal_req_id"] = internal_req_id
        if ckpt_tokens is not None and ckpt_tokens > 0:
            # New engine reads this to know how many tokens of KV the
            # old engine published; needed to size the V3 reload alloc.
            vllm_xargs["num_checkpointed_tokens"] = ckpt_tokens
        new_body["vllm_xargs"] = vllm_xargs

        self.reroute_count += 1
        logger.info(
            "reroute req=%s: engine %d -> engine %d, "
            "internal_req_id=%s, ckpt_tokens=%s",
            user_req_id,
            original.engine_id,
            new_target.engine_id,
            internal_req_id,
            ckpt_tokens,
        )

        resp = await self.forward(
            new_target, original.endpoint, new_body, user_req_id
        )
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
            headers={
                "X-Request-Id": user_req_id,
                "X-Rerouted-From": str(original.engine_id),
            },
        )


# ── FastAPI app ────────────────────────────────────────────────────


def create_app(router: Router) -> FastAPI:
    app = FastAPI(title="vllm-router (paper)")
    app.state.router = router
    app.state.poller_task = None

    @app.on_event("startup")
    async def _startup() -> None:
        async def status_poll_loop():
            while True:
                router.scan_engine_status()
                await asyncio.sleep(STATUS_POLL_INTERVAL_S)

        async def preempt_poll_loop():
            while True:
                router.scan_preempt_queue()
                await asyncio.sleep(PREEMPT_POLL_INTERVAL_S)

        app.state.poller_task = asyncio.create_task(status_poll_loop())
        app.state.preempt_poller_task = asyncio.create_task(
            preempt_poll_loop()
        )
        logger.info(
            "router started; engines=%s",
            {eid: e.base_url for eid, e in router.engines.items()},
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if app.state.poller_task is not None:
            app.state.poller_task.cancel()
        if getattr(app.state, "preempt_poller_task", None) is not None:
            app.state.preempt_poller_task.cancel()
        await router.close()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engines": {
                eid: {
                    "alive": e.alive,
                    "running": e.running,
                    "waiting": e.waiting,
                    "kv_usage": e.kv_usage,
                    "last_ts": e.last_ts,
                }
                for eid, e in router.engines.items()
            },
            "in_flight": len(router.in_flight),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return {
            "dispatch_count": router.dispatch_count,
            "reroute_count": router.reroute_count,
            "dead_event_count": router.dead_event_count,
            "in_flight": len(router.in_flight),
        }

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        return await router.proxy("/v1/completions", body)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        return await router.proxy("/v1/chat/completions", body)

    return app


# ── CLI ────────────────────────────────────────────────────────────


def _parse_engines(specs: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"engine spec must be 'ID=URL', got {spec!r}"
            )
        eid_str, url = spec.split("=", 1)
        out[int(eid_str)] = url
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HTTP router for cross-engine SLO-aware LLM serving. "
            "Reads engine status from /dev/shm to dispatch + reroute."
        )
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--engines",
        nargs="+",
        required=True,
        metavar="ID=URL",
        help=(
            "One or more engine specs, e.g. "
            "--engines 0=http://127.0.0.1:8001 1=http://127.0.0.1:8002"
        ),
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engines = _parse_engines(args.engines)
    router = Router(engines)
    app = create_app(router)
    uvicorn.run(
        app, host=args.host, port=args.port, log_level=args.log_level
    )


if __name__ == "__main__":
    main()
