# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request Pool for Fault-Tolerant Multi-GPU Serving.

Maintains a centralized pool of requests in host memory (RAM), serving as
the single source of truth for request state. The pool supports:
- Holding pending requests awaiting admission.
- Tracking admitted requests assigned to replicas.
- Re-queuing requests displaced by GPU failures for re-routing.
"""

import threading
from enum import Enum

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


class RequestPoolStatus(Enum):
    """Status of a request within the pool."""

    PENDING = "pending"  # Awaiting admission.
    ADMITTED = "admitted"  # Admitted and assigned to a replica.
    DISPLACED = "displaced"  # Replica failed; awaiting re-routing.
    COMPLETED = "completed"  # Finished generation.


class RequestPool:
    """Centralized request pool in host memory.

    Corresponds to the "request pool" box in the architecture diagram.
    All requests pass through this pool, enabling the scheduler to make
    global admission and routing decisions.
    """

    def __init__(self) -> None:
        self._requests: dict[str, Request] = {}
        self._status: dict[str, RequestPoolStatus] = {}
        # Use sets for O(1) membership tracking instead of unbounded deques.
        self._pending_ids: set[str] = set()
        self._displaced_ids: set[str] = set()
        self._lock = threading.Lock()
        # Track insertion order for FIFO within pending/displaced.
        self._pending_order: list[str] = []
        self._displaced_order: list[str] = []

    def add_request(self, request: Request) -> None:
        """Add a new request to the pool as PENDING."""
        with self._lock:
            self._requests[request.request_id] = request
            self._status[request.request_id] = RequestPoolStatus.PENDING
            self._pending_ids.add(request.request_id)
            self._pending_order.append(request.request_id)

    def admit_request(
        self, request_id: str, replica_id: int
    ) -> Request | None:
        """Mark a PENDING request as admitted and assign it to a replica."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return None
            self._status[request_id] = RequestPoolStatus.ADMITTED
            # Clean up from pending set.
            self._pending_ids.discard(request_id)
            request.assigned_replica_id = replica_id
            return request

    def readmit_request(
        self, request_id: str, replica_id: int
    ) -> Request | None:
        """Re-admit a DISPLACED request to a new replica after failover."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return None
            self._status[request_id] = RequestPoolStatus.ADMITTED
            # Clean up from displaced set.
            self._displaced_ids.discard(request_id)
            request.assigned_replica_id = replica_id
            return request

    def displace_requests(self, replica_id: int) -> list[Request]:
        """Mark all requests on a failed replica as displaced.

        Called by the Recovery Manager when a GPU failure is detected.
        Returns the list of displaced requests for re-routing.
        """
        displaced = []
        with self._lock:
            for req_id, request in self._requests.items():
                if (
                    request.assigned_replica_id == replica_id
                    and self._status.get(req_id) == RequestPoolStatus.ADMITTED
                ):
                    self._status[req_id] = RequestPoolStatus.DISPLACED
                    self._displaced_ids.add(req_id)
                    self._displaced_order.append(req_id)
                    displaced.append(request)
        if displaced:
            logger.info(
                "Displaced %d requests from failed replica %d",
                len(displaced),
                replica_id,
            )
        return displaced

    def get_pending_requests(self) -> list[Request]:
        """Get all pending requests in FIFO order."""
        with self._lock:
            self._maybe_compact_pending()
            requests = []
            for req_id in self._pending_order:
                if req_id in self._pending_ids:
                    req = self._requests.get(req_id)
                    if req is not None:
                        requests.append(req)
            return requests

    def get_admitted_requests(self) -> list[Request]:
        """Get all currently admitted requests."""
        with self._lock:
            return [
                req
                for req in self._requests.values()
                if self._status.get(req.request_id)
                == RequestPoolStatus.ADMITTED
            ]

    def get_displaced_requests(self) -> list[Request]:
        """Get all displaced requests awaiting re-routing."""
        with self._lock:
            self._maybe_compact_displaced()
            requests = []
            for req_id in self._displaced_order:
                if req_id in self._displaced_ids:
                    req = self._requests.get(req_id)
                    if req is not None:
                        requests.append(req)
            return requests

    def complete_request(self, request_id: str) -> None:
        """Mark a request as completed and remove it from the pool."""
        with self._lock:
            self._status.pop(request_id, None)
            self._requests.pop(request_id, None)
            self._pending_ids.discard(request_id)
            self._displaced_ids.discard(request_id)

    def remove_request(self, request_id: str) -> None:
        """Remove a request from the pool entirely (abort/cancel)."""
        with self._lock:
            self._requests.pop(request_id, None)
            self._status.pop(request_id, None)
            self._pending_ids.discard(request_id)
            self._displaced_ids.discard(request_id)

    def get_request(self, request_id: str) -> Request | None:
        return self._requests.get(request_id)

    def get_requests_on_replica(self, replica_id: int) -> list[Request]:
        """Get all admitted requests currently assigned to a replica."""
        with self._lock:
            return [
                req
                for req in self._requests.values()
                if (
                    req.assigned_replica_id == replica_id
                    and self._status.get(req.request_id)
                    == RequestPoolStatus.ADMITTED
                )
            ]

    def _maybe_compact_pending(self) -> None:
        """Auto-compact pending order when stale entries exceed live ones.

        Must hold _lock.
        """
        if len(self._pending_order) > max(len(self._pending_ids) * 2, 64):
            self._pending_order = [
                rid for rid in self._pending_order
                if rid in self._pending_ids
            ]

    def _maybe_compact_displaced(self) -> None:
        """Auto-compact displaced order when stale entries exceed live ones.

        Must hold _lock.
        """
        if len(self._displaced_order) > max(len(self._displaced_ids) * 2, 64):
            self._displaced_order = [
                rid for rid in self._displaced_order
                if rid in self._displaced_ids
            ]

    def compact(self) -> None:
        """Remove stale entries from order lists to prevent unbounded growth."""
        with self._lock:
            self._pending_order = [
                rid for rid in self._pending_order if rid in self._pending_ids
            ]
            self._displaced_order = [
                rid for rid in self._displaced_order
                if rid in self._displaced_ids
            ]

    @property
    def num_pending(self) -> int:
        return len(self._pending_ids)

    @property
    def num_admitted(self) -> int:
        return sum(
            1
            for s in self._status.values()
            if s == RequestPoolStatus.ADMITTED
        )

    @property
    def num_displaced(self) -> int:
        return len(self._displaced_ids)
