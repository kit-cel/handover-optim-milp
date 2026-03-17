"""Event-based simulation core."""

from dataclasses import dataclass, field
from enum import IntEnum
import heapq
from typing import Callable


class Priority(IntEnum):
    """Event priorities."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(order=True)
class ScheduledEvent:
    """A single event in the future."""

    time: int  # absolute time [ms] – the field used for ordering
    priority: int  # tie‑breaker (lower = earlier)
    action: Callable[[], None] = field(compare=False, repr=False)  # what to run
    name: str = field(default="", compare=False)  # optional name for debugging
    cancelled: bool = field(default=False, compare=False)


class VirtualClock:
    """Deterministic event queue.  Time is an integer (ms)."""

    def __init__(self, tick_ms: int = 10, simulation_time_ms: int = 1_000):
        self.tick_ms = tick_ms  # smallest resolution
        self.simulation_time_ms = simulation_time_ms  # total sim time
        self.n_steps = simulation_time_ms // tick_ms  # total number of steps

        self.now = 0
        self.step_now = 0
        self._queue: list[ScheduledEvent] = []  # heap‑queue (min‑heap)

    def schedule(
        self,
        delay_ms: int,
        action: Callable[[], None],
        priority: Priority = Priority.NORMAL,
        name: str = "",
    ) -> ScheduledEvent:
        """Schedule *action* to fire after *delay_ms* (rounded up to tick)."""
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if (delay_ms % self.tick_ms) != 0:
            raise ValueError("delay_ms must be multiple of tick_ms")

        expiry = self.now + delay_ms
        ev = ScheduledEvent(expiry, priority, action, name=name)
        heapq.heappush(self._queue, ev)
        return ev

    def cancel(self, ev: ScheduledEvent) -> None:
        """Cancel a previously scheduled event (if still pending)."""
        try:
            ev.cancelled = True
        except ValueError:
            pass  # already fired

    def advance(self, step_ms: int = 1) -> None:
        """Advance the clock by *step_ms* and run all due events."""
        self.now += step_ms
        self.step_now += 1

    def run_due_events(self) -> None:
        """Run all events that are due at the current time."""
        while self._queue and self._queue[0].time <= self.now:
            ev = heapq.heappop(self._queue)
            if ev.cancelled:
                continue
            ev.action()
