from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.analysis.liveness import Interval, peak_bytes, total_bytes
from tgc.errors import AllocationError, ConfigError

# Placing tensors in one arena so that values alive at the same time never overlap.
#
# This is not register allocation with different sized registers, and the difference matters.
# A colouring says which values may share a colour; it says nothing about laying them out
# contiguously, and a plan that assigns the same colour to a large tensor and a small one has
# to give the small one the large one's footprint or the arena stops being contiguous.
#
# So it is done as offset assignment against an interference test, which is a bin packing
# problem, which is why the order the values are considered in decides almost everything.
# Placing the biggest first is the rule that wins, for the same reason it wins in every other
# packing problem: a large block placed late has to go above everything already down.


@dataclass
class Allocation:
    """Where one tensor lives in the arena."""

    name: str
    offset: int
    size: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ConfigError(f"{self.name} cannot sit at {self.offset}")
        if self.size < 0:
            raise ConfigError(f"{self.name} cannot occupy {self.size} bytes")

    @property
    def end(self) -> int:
        """The first byte past this tensor."""
        return self.offset + self.size

    def overlaps(self, other: Allocation) -> bool:
        """Whether two placements share a byte."""
        return self.offset < other.end and other.offset < self.end

    def as_dict(self) -> dict[str, int | str]:
        """Flat mapping for logging."""
        return {"name": self.name, "offset": self.offset, "size": self.size}


@dataclass
class Plan:
    """A complete assignment of tensors to arena offsets."""

    allocations: list[Allocation] = field(default_factory=list)
    strategy: str = ""

    @property
    def arena_bytes(self) -> int:
        """How large the arena has to be."""
        return max((allocation.end for allocation in self.allocations), default=0)

    def by_name(self) -> dict[str, Allocation]:
        """The placements indexed by value."""
        return {allocation.name: allocation for allocation in self.allocations}

    def overhead_against(self, floor: int) -> float:
        """How much more than the theoretical minimum this plan uses."""
        if floor <= 0:
            return 0.0
        return self.arena_bytes / floor - 1.0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "strategy": self.strategy,
            "tensors": len(self.allocations),
            "arena_bytes": self.arena_bytes,
        }


def validate_plan(intervals: Sequence[Interval], plan: Plan) -> None:
    """Check that no two values alive at once were given the same bytes.

    Run on every plan in the test suite. A plan that overlaps two live tensors produces
    corrupted numbers rather than a crash, which is the worst failure mode a compiler has,
    and the check is quadratic in a number that is small.
    """
    placements = plan.by_name()
    missing = [interval.name for interval in intervals if interval.name not in placements]
    if missing:
        raise AllocationError(f"no placement for {missing}")

    for index, first in enumerate(intervals):
        for second in intervals[index + 1 :]:
            if not first.overlaps(second):
                continue
            if placements[first.name].overlaps(placements[second.name]):
                raise AllocationError(
                    f"{first.name} and {second.name} are alive together and share bytes "
                    f"{placements[first.name].offset} and {placements[second.name].offset}"
                )


def plan_is_valid(intervals: Sequence[Interval], plan: Plan) -> bool:
    """Whether the plan places every value without overlapping a live one."""
    try:
        validate_plan(intervals, plan)
    except AllocationError:
        return False
    return True


def plan_without_reuse(intervals: Sequence[Interval]) -> Plan:
    """Give every value its own bytes.

    The baseline. Correct by construction and needs the sum of every tensor in the graph,
    which is the number people quote when they say a model does not fit.
    """
    offset = 0
    allocations = []
    for interval in intervals:
        allocations.append(Allocation(name=interval.name, offset=offset, size=interval.size))
        offset += interval.size
    return Plan(allocations=allocations, strategy="no reuse")


def _first_free_offset(interval: Interval, placed: list[tuple[Interval, Allocation]]) -> int:
    """The lowest offset where a value fits without touching anything alive at the same time."""
    blocked = sorted(
        (allocation for other, allocation in placed if interval.overlaps(other)),
        key=lambda allocation: allocation.offset,
    )
    offset = 0
    for allocation in blocked:
        if offset + interval.size <= allocation.offset:
            return offset
        offset = max(offset, allocation.end)
    return offset


def plan_first_fit(intervals: Sequence[Interval]) -> Plan:
    """Place values in the order they are produced, each at the lowest offset that fits.

    The obvious allocator. It leaves holes, because a small tensor placed early can sit in
    the middle of the arena and force every later large one above it.
    """
    placed: list[tuple[Interval, Allocation]] = []
    for interval in sorted(intervals, key=lambda item: (item.start, item.name)):
        offset = _first_free_offset(interval, placed)
        placed.append(
            (interval, Allocation(name=interval.name, offset=offset, size=interval.size))
        )
    return Plan(allocations=[allocation for _, allocation in placed], strategy="first fit")


def plan_largest_first(intervals: Sequence[Interval]) -> Plan:
    """Place the biggest values first, each at the lowest offset that fits.

    The rule that wins, for the same reason it wins in every other packing problem: a large
    block placed late has to go above everything already down, and a small one placed late
    fits in a hole.
    """
    placed: list[tuple[Interval, Allocation]] = []
    ordered = sorted(intervals, key=lambda item: (-item.size, item.start, item.name))
    for interval in ordered:
        offset = _first_free_offset(interval, placed)
        placed.append(
            (interval, Allocation(name=interval.name, offset=offset, size=interval.size))
        )
    return Plan(allocations=[allocation for _, allocation in placed], strategy="largest first")


def plan_longest_lived_first(intervals: Sequence[Interval]) -> Plan:
    """Place the values that stay alive longest first.

    Plausible and worse than sorting by size. A long lived value conflicts with many others
    and so is hard to place late, but the arena is measured in bytes and not in conflicts,
    and a long lived small tensor placed at offset zero blocks nothing worth blocking.
    """
    placed: list[tuple[Interval, Allocation]] = []
    ordered = sorted(intervals, key=lambda item: (-item.length, item.start, item.name))
    for interval in ordered:
        offset = _first_free_offset(interval, placed)
        placed.append(
            (interval, Allocation(name=interval.name, offset=offset, size=interval.size))
        )
    return Plan(
        allocations=[allocation for _, allocation in placed], strategy="longest lived first"
    )


STRATEGIES = {
    "no reuse": plan_without_reuse,
    "first fit": plan_first_fit,
    "largest first": plan_largest_first,
    "longest lived first": plan_longest_lived_first,
}


def get_strategy(name: str):
    """Look up an allocator by name."""
    if name not in STRATEGIES:
        raise ConfigError(f"unknown strategy {name!r}, expected one of {sorted(STRATEGIES)}")
    return STRATEGIES[name]


def compare_strategies(intervals: Sequence[Interval]) -> list[dict]:
    """Every allocator on the same liveness, against the floor it cannot beat.

    The floor is the peak of simultaneously live bytes. Comparing an allocator against
    another allocator says which is better; comparing it against the floor says whether there
    is anything left to win, which is the question worth asking before writing a third one.
    """
    floor = peak_bytes(intervals)
    rows = []
    for name, strategy in STRATEGIES.items():
        plan = strategy(intervals)
        validate_plan(intervals, plan)
        rows.append(
            {
                "strategy": name,
                "arena_bytes": plan.arena_bytes,
                "overhead": round(plan.overhead_against(floor), 4),
                "reaches_floor": plan.arena_bytes == floor,
            }
        )
    rows.append(
        {
            "strategy": "floor",
            "arena_bytes": floor,
            "overhead": 0.0,
            "reaches_floor": True,
        }
    )
    return rows


def fragmentation(intervals: Sequence[Interval], plan: Plan) -> float:
    """Share of the arena that is holes rather than tensors.

    Measured against the peak rather than against the arena, because an allocator that is
    generous everywhere has low fragmentation by this definition and is still wasteful. The
    number that matters is how far above the floor it sits.
    """
    floor = peak_bytes(intervals)
    if plan.arena_bytes == 0:
        return 0.0
    return 1.0 - floor / plan.arena_bytes


def savings_against_no_reuse(intervals: Sequence[Interval], plan: Plan) -> float:
    """How much of the naive footprint the plan gives back."""
    naive = total_bytes(intervals)
    if naive == 0:
        return 0.0
    return 1.0 - plan.arena_bytes / naive


def packing_hazard() -> list[Interval]:
    """Intervals where placing in production order leaves a hole nothing fills.

    Four large tensors and four small ones, arranged so that some of the small ones are
    produced first. First fit puts them at the bottom of the arena and every large tensor
    afterwards has to start above them, which costs a third of the arena for tensors that
    occupy an eighth of it.
    """
    return [
        Interval(name="v0", start=4, end=7, size=16384),
        Interval(name="v1", start=1, end=5, size=4096),
        Interval(name="v2", start=2, end=6, size=2048),
        Interval(name="v3", start=6, end=7, size=1024),
        Interval(name="v4", start=5, end=8, size=1024),
        Interval(name="v5", start=5, end=8, size=16384),
        Interval(name="v6", start=6, end=9, size=16384),
        Interval(name="v7", start=1, end=4, size=16384),
    ]


def random_intervals(seed: int = 0, count: int = 8) -> list[Interval]:
    """A random set of live ranges with a spread of sizes."""
    if count < 1:
        raise ConfigError(f"there has to be at least one interval, got {count}")
    generator = random.Random(seed)
    intervals = []
    for index in range(count):
        start = generator.randrange(0, 8)
        intervals.append(
            Interval(
                name=f"v{index}",
                start=start,
                end=start + generator.randrange(0, 5),
                size=generator.choice([1, 2, 4, 8, 16]) * 1024,
            )
        )
    return intervals


def floor_miss_rate(strategy_name: str, trials: int = 500) -> float:
    """How often an allocator fails to reach the theoretical minimum.

    The number that decides whether a better allocator is worth writing. Sorting by size
    misses the floor on a small fraction of random interval sets and placing in production
    order misses it several times more often, which is a difference worth the two lines it
    costs.
    """
    if trials < 1:
        raise ConfigError(f"the trial count must be positive, got {trials}")
    strategy = get_strategy(strategy_name)
    misses = 0
    for seed in range(trials):
        intervals = random_intervals(seed=seed)
        plan = strategy(intervals)
        if plan.arena_bytes != peak_bytes(intervals):
            misses += 1
    return misses / trials
