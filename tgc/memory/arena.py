from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.analysis.liveness import Interval, compute_intervals, peak_bytes
from tgc.errors import AllocationError, ConfigError
from tgc.ir.builder import Builder, elementwise_chain
from tgc.ir.graph import Graph
from tgc.memory.planner import Allocation, Plan, plan_largest_first, validate_plan
from tgc.schedule.order import depth_first_order

# One contiguous buffer with everything laid out inside it, at addresses the hardware likes.
#
# The planner in memory/planner.py places tensors at whatever offset fits. That is correct and
# not sufficient: a vector load of sixty four bytes issued at an address that is not a multiple
# of sixty four either costs two loads or faults, depending on the machine, and an offset that
# happens to be a multiple of four is exactly what a packing algorithm produces when nobody
# asks it for more.
#
# Rounding every offset up to an alignment is trivial and costs padding. How much padding is
# the interesting question, and the answer is that it depends almost entirely on how many
# tensors there are rather than how large they are: the waste per tensor is bounded by the
# alignment, so a graph with many small values pays proportionally far more than one with a
# few large ones.


def is_power_of_two(value: int) -> bool:
    """Whether a number is a power of two, which every sensible alignment is."""
    return value > 0 and value & (value - 1) == 0


def align_up(offset: int, alignment: int) -> int:
    """The next offset at or above one that satisfies an alignment."""
    if not is_power_of_two(alignment):
        raise ConfigError(f"an alignment has to be a power of two, got {alignment}")
    if offset < 0:
        raise ConfigError(f"an offset cannot be negative, got {offset}")
    return (offset + alignment - 1) & ~(alignment - 1)


def is_aligned(offset: int, alignment: int) -> bool:
    """Whether an offset already satisfies an alignment."""
    if not is_power_of_two(alignment):
        raise ConfigError(f"an alignment has to be a power of two, got {alignment}")
    return offset % alignment == 0


@dataclass
class Arena:
    """A contiguous buffer with aligned slots inside it."""

    alignment: int = 64
    allocations: list[Allocation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not is_power_of_two(self.alignment):
            raise ConfigError(f"an alignment has to be a power of two, got {self.alignment}")
        misaligned = [
            allocation.name
            for allocation in self.allocations
            if not is_aligned(allocation.offset, self.alignment)
        ]
        if misaligned:
            raise AllocationError(
                f"these are not aligned to {self.alignment}: {sorted(misaligned)}"
            )

    @property
    def size(self) -> int:
        """Bytes the arena occupies."""
        return max((allocation.end for allocation in self.allocations), default=0)

    @property
    def used(self) -> int:
        """Bytes some tensor actually occupies, counting a shared byte once.

        The union of the placed ranges rather than the sum of the sizes, and the distinction
        is not pedantic. Under a reusing plan several tensors sit at the same offset, so
        summing their sizes counts those bytes repeatedly and reports more used than the arena
        holds, which is how the first version of this produced negative padding.
        """
        if not self.allocations:
            return 0
        ranges = sorted((allocation.offset, allocation.end) for allocation in self.allocations)
        covered = 0
        current_start, current_end = ranges[0]
        for start, end in ranges[1:]:
            if start > current_end:
                covered += current_end - current_start
                current_start, current_end = start, end
                continue
            current_end = max(current_end, end)
        return covered + current_end - current_start

    @property
    def padding(self) -> int:
        """Bytes inside the arena that no tensor occupies."""
        return self.size - self.used

    @property
    def padding_fraction(self) -> float:
        """Share of the arena that exists only to satisfy the alignment."""
        if self.size == 0:
            return 0.0
        return self.padding / self.size

    def by_name(self) -> dict[str, Allocation]:
        """The placements indexed by value."""
        return {allocation.name: allocation for allocation in self.allocations}

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "alignment": self.alignment,
            "size": self.size,
            "used": self.used,
            "padding": self.padding,
            "padding_fraction": round(self.padding_fraction, 4),
            "tensors": len(self.allocations),
        }


def align_plan(plan: Plan, alignment: int = 64) -> Arena:
    """Round every offset in a plan up to an alignment.

    Done by re placing rather than by nudging each offset in place. Nudging a single offset
    upward can push a tensor into the one above it, and the plan was only valid because the
    offsets were exactly where the packer put them.
    """
    if not is_power_of_two(alignment):
        raise ConfigError(f"an alignment has to be a power of two, got {alignment}")

    ordered = sorted(plan.allocations, key=lambda item: (item.offset, item.name))
    aligned: list[Allocation] = []
    for allocation in ordered:
        offset = align_up(allocation.offset, alignment)
        aligned.append(Allocation(name=allocation.name, offset=offset, size=allocation.size))
    return Arena(alignment=alignment, allocations=aligned)


def stacked_arena(intervals: Sequence[Interval], alignment: int = 64) -> Arena:
    """Lay every value out one after another with no reuse, aligned.

    The version that shows the padding clearly, since nothing overlaps and every tensor pays
    its own rounding. A reusing plan pays less because two values sharing a slot share its
    padding, which is a saving nobody designed and is worth knowing about.
    """
    if not is_power_of_two(alignment):
        raise ConfigError(f"an alignment has to be a power of two, got {alignment}")
    offset = 0
    allocations = []
    for interval in sorted(intervals, key=lambda item: item.name):
        offset = align_up(offset, alignment)
        allocations.append(Allocation(name=interval.name, offset=offset, size=interval.size))
        offset += interval.size
    return Arena(alignment=alignment, allocations=allocations)


def arena_for(graph: Graph, alignment: int = 64) -> Arena:
    """The aligned arena a compiled graph would use."""
    order = depth_first_order(graph)
    intervals = compute_intervals(graph, order)
    plan = plan_largest_first(intervals)
    validate_plan(intervals, plan)
    return align_plan(plan, alignment)


def alignment_sweep(
    graph: Graph, alignments: Sequence[int] = (1, 4, 16, 64, 256, 4096)
) -> list[dict]:
    """Arena size across a range of alignments.

    Padding rises with the alignment and the rise is not proportional to the data. It is
    bounded by the alignment times the number of tensors, so the cost is a property of how
    many values a graph holds rather than how big they are.
    """
    if not alignments:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for alignment in alignments:
        arena = arena_for(graph, alignment)
        row = arena.as_dict()
        row["floor"] = peak_bytes(compute_intervals(graph, depth_first_order(graph)))
        rows.append(row)
    return rows


def awkward_graph(nodes: int = 12) -> Graph:
    """A graph whose tensors are not multiples of any sensible alignment.

    Sixty bytes each. Every fixture elsewhere in this repository uses power of two shapes, so
    every offset a packer produces is already aligned to sixty four by accident and an
    alignment sweep over them measures nothing.
    """
    if nodes < 1:
        raise ConfigError(f"the graph needs at least one node, got {nodes}")
    builder = Builder()
    current = builder.input([3, 5], name="x")
    for index in range(nodes):
        current = builder.relu(current) if index % 2 else builder.tanh(current)
    return builder.finish(current)


def compare_shapes(alignment: int = 64) -> list[dict]:
    """Padding on a graph of many small tensors against one of few large ones.

    Same alignment, opposite outcomes. The waste per slot is bounded by the alignment, so a
    graph holding many small values pays a large share and one holding a few large values pays
    almost nothing, and neither number is a property of the alignment on its own.
    """
    rows = []
    for label, graph in (
        ("many small", awkward_graph(16)),
        ("few large", elementwise_chain(4, sizes=(256, 256))),
    ):
        arena = align_plan(_plain_plan(graph), alignment)
        row = arena.as_dict()
        row["graph"] = label
        rows.append(row)
    return rows


def _plain_plan(graph: Graph) -> Plan:
    """The unaligned plan for a graph."""
    order = depth_first_order(graph)
    intervals = compute_intervals(graph, order)
    plan = plan_largest_first(intervals)
    validate_plan(intervals, plan)
    return plan


def reuse_shares_padding(graph: Graph, alignment: int = 64) -> dict:
    """Padding under a reusing plan against a stacked one.

    Two values sharing a slot share its padding, so reuse gives back alignment waste as well
    as tensor bytes. That is a second order effect and it is real: the stacked arena pays a
    rounding per value and the reusing one pays a rounding per slot.
    """
    order = depth_first_order(graph)
    intervals = compute_intervals(graph, order)
    reusing = align_plan(plan_largest_first(intervals), alignment)
    stacked = stacked_arena(intervals, alignment)
    return {
        "reusing_padding": reusing.padding,
        "stacked_padding": stacked.padding,
        "reusing_size": reusing.size,
        "stacked_size": stacked.size,
    }


def check_alignment(arena: Arena) -> None:
    """Raise if anything in an arena sits at an address the hardware would refuse."""
    for allocation in arena.allocations:
        if not is_aligned(allocation.offset, arena.alignment):
            raise AllocationError(
                f"{allocation.name} sits at {allocation.offset}, "
                f"which is not a multiple of {arena.alignment}"
            )


def plan_survives_alignment(graph: Graph, alignment: int = 64) -> bool:
    """Whether an aligned arena still keeps live values apart.

    Rounding offsets upward moves tensors, and moving tensors is exactly how a valid plan
    stops being one. Re placing rather than nudging is what keeps this true, and this checks
    it rather than assuming it.
    """
    order = depth_first_order(graph)
    intervals = compute_intervals(graph, order)
    arena = align_plan(plan_largest_first(intervals), alignment)
    plan = Plan(allocations=list(arena.allocations), strategy="aligned")
    try:
        validate_plan(intervals, plan)
    except AllocationError:
        return False
    return True
