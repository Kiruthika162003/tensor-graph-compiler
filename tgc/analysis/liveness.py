from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, ScheduleError
from tgc.ir.graph import Graph, Node

# When each tensor has to exist.
#
# A value is live from the step that produces it to the step that last reads it. Two values
# whose intervals overlap need separate storage; two whose intervals do not can share it.
# That is the whole of buffer allocation, and everything downstream is a question of how well
# a particular allocator exploits it.
#
# The part worth being careful about is the ends. A graph input is live before the first step,
# a graph output is live after the last one, and a value that nothing reads is live for
# exactly one step rather than for none. Getting any of those wrong produces a plan that
# allocates less memory than the program needs, which is a class of bug that shows up as
# corrupted numbers rather than as a crash.


@dataclass
class Interval:
    """The window during which one tensor has to exist."""

    name: str
    start: int
    end: int
    size: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("an interval needs a name")
        if self.end < self.start:
            raise ScheduleError(
                f"{self.name} ends at {self.end} before it starts at {self.start}"
            )
        if self.size < 0:
            raise ConfigError(f"{self.name} cannot occupy {self.size} bytes")

    @property
    def length(self) -> int:
        """Steps the value stays alive."""
        return self.end - self.start + 1

    def overlaps(self, other: Interval) -> bool:
        """Whether two values are ever alive at the same time.

        Inclusive at both ends. A value produced at the step another one is last read at does
        overlap it, because the reading step needs both of them present.
        """
        return self.start <= other.end and other.start <= self.end

    def contains(self, step: int) -> bool:
        """Whether the value is alive at a given step."""
        return self.start <= step <= self.end

    def as_dict(self) -> dict[str, int | str]:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "size": self.size,
            "length": self.length,
        }


def compute_intervals(graph: Graph, order: Sequence[Node] | None = None) -> list[Interval]:
    """When every value in a graph is live, under a given execution order.

    Inputs start before step zero and outputs end after the last step, because both are alive
    outside the window the compiler controls. An intermediate nothing reads still occupies its
    own step, since it is written before anybody could have decided it was pointless.
    """
    nodes = list(order) if order is not None else list(graph.nodes)
    positions = {node.name: index for index, node in enumerate(nodes)}
    if len(positions) != len(nodes):
        raise ScheduleError("an execution order cannot run the same node twice")
    last = len(nodes) - 1

    intervals: list[Interval] = []
    for value in graph.inputs:
        end = -1
        for index, node in enumerate(nodes):
            if value.name in node.inputs:
                end = index
        if value.name in graph.outputs:
            end = last
        intervals.append(
            Interval(name=value.name, start=-1, end=max(end, -1), size=value.bytes)
        )

    for index, node in enumerate(nodes):
        end = index
        for later, reader in enumerate(nodes):
            if node.name in reader.inputs:
                end = max(end, later)
        if node.name in graph.outputs:
            end = last
        intervals.append(Interval(name=node.name, start=index, end=end, size=node.output.bytes))
    return intervals


def live_at(intervals: Sequence[Interval], step: int) -> list[Interval]:
    """Every value alive at one step."""
    return [interval for interval in intervals if interval.contains(step)]


def bytes_live_at(intervals: Sequence[Interval], step: int) -> int:
    """Storage in use at one step."""
    return sum(interval.size for interval in live_at(intervals, step))


def peak_bytes(intervals: Sequence[Interval]) -> int:
    """The most storage that is ever needed at once.

    A lower bound on what any allocator can achieve, because the values counted here are all
    alive simultaneously and no arrangement of them shares a byte. An allocator that reports
    less than this has a bug, and comparing an allocator against this number rather than
    against another allocator is the only way to know how much room is left.
    """
    if not intervals:
        return 0
    first = min(interval.start for interval in intervals)
    last = max(interval.end for interval in intervals)
    return max(bytes_live_at(intervals, step) for step in range(first, last + 1))


def peak_step(intervals: Sequence[Interval]) -> int:
    """The step at which the most storage is needed."""
    if not intervals:
        raise ScheduleError("an empty schedule has no peak")
    first = min(interval.start for interval in intervals)
    last = max(interval.end for interval in intervals)
    return max(range(first, last + 1), key=lambda step: bytes_live_at(intervals, step))


def total_bytes(intervals: Sequence[Interval]) -> int:
    """Storage a plan that never reuses anything would need."""
    return sum(interval.size for interval in intervals)


def conflict_graph(intervals: Sequence[Interval]) -> dict[str, set[str]]:
    """Which values cannot share storage with which.

    An interference graph. Colouring it optimally is the classic formulation and is not what
    an allocator does, because the colours have different sizes and a colouring says nothing
    about how to lay them out in one contiguous arena.
    """
    conflicts: dict[str, set[str]] = {interval.name: set() for interval in intervals}
    for index, first in enumerate(intervals):
        for second in intervals[index + 1 :]:
            if first.overlaps(second):
                conflicts[first.name].add(second.name)
                conflicts[second.name].add(first.name)
    return conflicts


def max_simultaneous(intervals: Sequence[Interval]) -> int:
    """The most values alive at once, regardless of their sizes."""
    if not intervals:
        return 0
    first = min(interval.start for interval in intervals)
    last = max(interval.end for interval in intervals)
    return max(len(live_at(intervals, step)) for step in range(first, last + 1))


@dataclass
class LivenessReport:
    """A summary of when things are alive."""

    intervals: list[Interval] = field(default_factory=list)

    @property
    def peak(self) -> int:
        """The floor any allocator has to reach."""
        return peak_bytes(self.intervals)

    @property
    def total(self) -> int:
        """What never reusing anything would cost."""
        return total_bytes(self.intervals)

    @property
    def reuse_headroom(self) -> float:
        """How much of the naive total reuse could in principle remove."""
        if self.total == 0:
            return 0.0
        return 1.0 - self.peak / self.total

    @property
    def longest_lived(self) -> Interval | None:
        """The value that stays around longest, which is usually the one to attack."""
        return max(self.intervals, key=lambda interval: interval.length, default=None)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "values": len(self.intervals),
            "peak_bytes": self.peak,
            "total_bytes": self.total,
            "reuse_headroom": round(self.reuse_headroom, 4),
            "max_simultaneous": max_simultaneous(self.intervals),
        }


def analyse(graph: Graph, order: Sequence[Node] | None = None) -> LivenessReport:
    """Liveness for a whole graph under one order."""
    return LivenessReport(intervals=compute_intervals(graph, order))
