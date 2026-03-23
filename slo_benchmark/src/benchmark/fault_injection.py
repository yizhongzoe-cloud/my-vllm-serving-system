"""Fault injection for fault-tolerant serving benchmarks.

Provides configurable GPU failure simulation during benchmark runs.
Supports scheduled failures (at specific times) and random failures
(Poisson process), enabling controlled evaluation of the failover system.
"""

import random
import threading
import time
from dataclasses import dataclass, field

from typing import Callable


@dataclass
class FaultEvent:
    """A single simulated fault event."""

    replica_id: int
    trigger_time: float  # Absolute time to trigger.
    triggered: bool = False
    actual_trigger_time: float | None = None


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection."""

    # Mode: "scheduled" or "random"
    mode: str = "scheduled"

    # For scheduled mode: list of (time_offset_sec, replica_id) tuples.
    scheduled_faults: list[tuple[float, int]] = field(default_factory=list)

    # For random mode: mean time between failures (seconds).
    mean_time_between_failures: float = 30.0

    # Total number of random faults to inject.
    max_random_faults: int = 1

    # Random seed for reproducibility.
    seed: int = 42

    # Duration of simulated failure (seconds). After this, replica "recovers".
    failure_duration_sec: float = 10.0


class FaultInjector:
    """Injects simulated GPU failures during benchmark runs.

    Usage:
        injector = FaultInjector(config, on_failure=handle_failure)
        injector.start(benchmark_start_time)
        # ... benchmark runs ...
        injector.stop()
        report = injector.get_report()
    """

    def __init__(
        self,
        config: FaultInjectionConfig,
        on_failure: Callable[[int], None],
        num_replicas: int = 1,
    ) -> None:
        """
        Args:
            config: Fault injection configuration.
            on_failure: Callback when a fault is triggered.
                Takes replica_id as argument.
            num_replicas: Total number of replicas (for random targeting).
        """
        self.config = config
        self.on_failure = on_failure
        self.num_replicas = num_replicas

        self._events: list[FaultEvent] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0

    def start(self, benchmark_start_time: float | None = None) -> None:
        """Start the fault injection thread."""
        self._start_time = benchmark_start_time or time.time()
        self._events = self._generate_events()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._injection_loop,
            daemon=True,
            name="fault-injector",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the fault injection thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _generate_events(self) -> list[FaultEvent]:
        """Generate fault events based on configuration."""
        events = []

        if self.config.mode == "scheduled":
            for offset_sec, replica_id in self.config.scheduled_faults:
                events.append(
                    FaultEvent(
                        replica_id=replica_id,
                        trigger_time=self._start_time + offset_sec,
                    )
                )
        elif self.config.mode == "random":
            rng = random.Random(self.config.seed)
            t = self._start_time
            for _ in range(self.config.max_random_faults):
                t += rng.expovariate(
                    1.0 / self.config.mean_time_between_failures
                )
                replica_id = rng.randint(0, self.num_replicas - 1)
                events.append(
                    FaultEvent(replica_id=replica_id, trigger_time=t)
                )

        # Sort by trigger time.
        events.sort(key=lambda e: e.trigger_time)
        return events

    def _injection_loop(self) -> None:
        """Background thread that triggers faults at scheduled times."""
        for event in self._events:
            if self._stop_event.is_set():
                break

            # Wait until trigger time.
            wait_time = event.trigger_time - time.time()
            if wait_time > 0:
                if self._stop_event.wait(timeout=wait_time):
                    break  # Stopped.

            # Trigger the fault.
            event.triggered = True
            event.actual_trigger_time = time.time()
            self.on_failure(event.replica_id)

    def get_report(self) -> dict:
        """Get a summary report of all injected faults."""
        triggered = [e for e in self._events if e.triggered]
        return {
            "total_planned": len(self._events),
            "total_triggered": len(triggered),
            "events": [
                {
                    "replica_id": e.replica_id,
                    "planned_time": e.trigger_time - self._start_time,
                    "actual_time": (
                        e.actual_trigger_time - self._start_time
                        if e.actual_trigger_time
                        else None
                    ),
                    "triggered": e.triggered,
                }
                for e in self._events
            ],
        }
