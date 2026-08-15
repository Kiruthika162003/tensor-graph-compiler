from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import CodegenError, ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import elementwise_chain, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.graph import Graph, Node

# Doing several elements per instruction, and counting the lanes that go to waste.
#
# A vector unit works on a fixed number of elements at a time. A loop over a length that is a
# multiple of that width uses every lane on every iteration; a loop over anything else runs a
# final iteration with some lanes disabled, and those lanes are work the hardware did and
# threw away.
#
# The waste is bounded by the width and is therefore a bigger share of a short loop than of a
# long one, which is the same shape as the alignment argument in memory/arena.py and has the
# same consequence: it is a property of how many loops there are rather than how much data
# they touch.
#
# The other half is which operations can vectorise at all. An elementwise chain can, because
# element i of the output depends only on element i of the inputs. A reduction cannot without
# a rearrangement, because every lane has to reach the same accumulator, and that rearrangement
# is exactly the reduction splitting in passes/reduction.py.

WIDTHS = (1, 2, 4, 8, 16)


@dataclass
class VectorPlan:
    """One loop mapped onto a vector unit."""

    length: int
    width: int

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ConfigError(f"a loop cannot run {self.length} times")
        if self.width < 1:
            raise ConfigError(f"a vector is at least one element wide, got {self.width}")

    @property
    def full_iterations(self) -> int:
        """Iterations where every lane does useful work."""
        return self.length // self.width

    @property
    def remainder(self) -> int:
        """Elements left over after the full iterations."""
        return self.length % self.width

    @property
    def has_tail(self) -> bool:
        """Whether a partial iteration is needed."""
        return self.remainder != 0

    @property
    def iterations(self) -> int:
        """Vector iterations issued, tail included."""
        return math.ceil(self.length / self.width) if self.length else 0

    @property
    def lanes_issued(self) -> int:
        """Lanes the hardware ran, useful or not."""
        return self.iterations * self.width

    @property
    def wasted_lanes(self) -> int:
        """Lanes that did work nobody kept."""
        return self.lanes_issued - self.length

    @property
    def waste_fraction(self) -> float:
        """Share of the issued lanes that went to waste."""
        if self.lanes_issued == 0:
            return 0.0
        return self.wasted_lanes / self.lanes_issued

    @property
    def speedup(self) -> float:
        """How much shorter the loop gets, counting the tail as a full iteration.

        Bounded above by the width and reached only when the width divides the length. A loop
        of nine at width eight issues two iterations, so the speedup is four and a half rather
        than eight and half the second iteration is thrown away.
        """
        if self.iterations == 0:
            return 1.0
        return self.length / self.iterations

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "length": self.length,
            "width": self.width,
            "iterations": self.iterations,
            "wasted_lanes": self.wasted_lanes,
            "waste_fraction": round(self.waste_fraction, 4),
            "speedup": round(self.speedup, 3),
        }


def can_vectorise(node: Node) -> tuple[bool, str]:
    """Whether one operation maps onto a vector unit without rearrangement.

    Elementwise yes, because element i of the output depends only on element i of the inputs.
    Reductions no, because every lane has to reach the same accumulator, and a matrix product
    no for the same reason along its contracted axis.
    """
    if node.op.is_elementwise:
        return True, ""
    if node.op.is_leaf:
        return False, "a leaf produces nothing to vectorise over"
    if node.op.category == ops.REDUCTION:
        return False, "every lane would have to reach the same accumulator"
    if node.op is ops.MATMUL:
        return False, "the contracted axis is a reduction in disguise"
    return False, f"{node.op.name} is not elementwise"


def vectorisable_fraction(graph: Graph) -> float:
    """Share of a graph's nodes that vectorise as written."""
    candidates = [node for node in graph.nodes if not node.op.is_leaf]
    if not candidates:
        return 0.0
    return sum(1 for node in candidates if can_vectorise(node)[0]) / len(candidates)


def refusals(graph: Graph) -> dict[str, str]:
    """Why each node that cannot vectorise cannot.

    Leaves are skipped rather than reported. They produce a value without reading one, so
    there is no loop over elements for a vector unit to take, and counting them as refusals
    made a layernorm look less vectorisable than it is.
    """
    reasons = {}
    for node in graph.nodes:
        if node.op.is_leaf:
            continue
        allowed, reason = can_vectorise(node)
        if not allowed and reason:
            reasons[node.name] = reason
    return reasons


def width_sweep(length: int = 100, widths: Sequence[int] = WIDTHS) -> list[dict]:
    """Speedup and waste across a range of vector widths.

    A wider unit is faster until the remainder grows with it. At length one hundred a width of
    eight wastes four lanes and a width of sixteen wastes twenty eight, and the speedup stops
    keeping up with the width long before the width stops growing.
    """
    if not widths:
        raise ConfigError("there is nothing to sweep")
    return [VectorPlan(length=length, width=width).as_dict() for width in widths]


def length_sweep(
    width: int = 8, lengths: Sequence[int] = (7, 8, 9, 64, 65, 1000)
) -> list[dict]:
    """Waste across a range of loop lengths at one width.

    The waste is bounded by the width, so it is most of a short loop and nothing much of a
    long one. A loop of nine at width eight throws away seven lanes out of sixteen; a loop of
    a thousand throws away zero point four percent.
    """
    if not lengths:
        raise ConfigError("there is nothing to sweep")
    return [VectorPlan(length=length, width=width).as_dict() for length in lengths]


def best_width(length: int, widths: Sequence[int] = WIDTHS) -> int:
    """The width that gives the shortest loop, breaking ties toward the narrower one.

    Ties matter here. At length eight both four and eight give two and one iterations, and the
    wider one is strictly better; at length nine the wider one issues the same two iterations
    as at eight and wastes seven lanes doing it.
    """
    if length < 1:
        raise ConfigError(f"a loop has to run, got {length}")
    if not widths:
        raise ConfigError("there is nothing to choose between")

    def score(width: int) -> tuple[int, int]:
        plan = VectorPlan(length=length, width=width)
        return (plan.iterations, plan.wasted_lanes)

    return min(widths, key=score)


def widest_is_not_always_best(lengths: Sequence[int] = (7, 9, 17, 33, 100)) -> list[dict]:
    """Where the widest available unit is the wrong choice.

    Not often, and the cases exist. The widest width always issues the fewest iterations, so
    it always wins on time and can lose badly on wasted work, which matters when the lanes
    cost power rather than latency.
    """
    if not lengths:
        raise ConfigError("there is nothing to compare")
    rows = []
    for length in lengths:
        chosen = best_width(length)
        widest = max(WIDTHS)
        rows.append(
            {
                "length": length,
                "chosen": chosen,
                "widest": widest,
                "chosen_waste": VectorPlan(length=length, width=chosen).wasted_lanes,
                "widest_waste": VectorPlan(length=length, width=widest).wasted_lanes,
            }
        )
    return rows


@dataclass
class GraphVectorReport:
    """How much of a graph a vector unit can take."""

    vectorisable: int = 0
    refused: int = 0
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Nodes considered."""
        return self.vectorisable + self.refused

    @property
    def fraction(self) -> float:
        """Share that vectorises as written."""
        if self.total == 0:
            return 0.0
        return self.vectorisable / self.total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "vectorisable": self.vectorisable,
            "refused": self.refused,
            "fraction": round(self.fraction, 4),
            "reasons": sorted(set(self.reasons.values())),
        }


def analyse(graph: Graph) -> GraphVectorReport:
    """Which nodes of a graph vectorise, and why the rest do not."""
    report = GraphVectorReport(reasons=refusals(graph))
    for node in graph.nodes:
        if node.op.is_leaf:
            continue
        if can_vectorise(node)[0]:
            report.vectorisable += 1
        else:
            report.refused += 1
    return report


def compare_graphs() -> list[dict]:
    """How much of each fixture a vector unit can take.

    An elementwise chain is entirely vectorisable and a softmax is not, because a softmax is
    two reductions holding an elementwise chain between them. That is the shape of most tensor
    code and the reason reduction splitting exists.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(8)),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        row = analyse(graph).as_dict()
        row["graph"] = label
        rows.append(row)
    return rows


def check_plan(plan: VectorPlan) -> None:
    """Raise if a plan does not account for every element.

    The arithmetic is three lines and the off by one it protects against costs a lane of the
    answer, which is silent. Full iterations times the width plus the remainder has to be the
    length, exactly.
    """
    accounted = plan.full_iterations * plan.width + plan.remainder
    if accounted != plan.length:
        raise CodegenError(
            f"a plan for {plan.length} at width {plan.width} accounts for {accounted}"
        )


def every_plan_accounts_for_its_elements(
    lengths: Sequence[int] = range(0, 200), widths: Sequence[int] = WIDTHS
) -> int:
    """Sweep the arithmetic across every length and width.

    Swept rather than sampled, for the same reason peeling was in codegen/loops.py: an off by
    one at a boundary is invisible at whatever size somebody picked and obvious across a range.
    """
    checked = 0
    for length in lengths:
        for width in widths:
            check_plan(VectorPlan(length=length, width=width))
            checked += 1
    return checked
