from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tgc.analysis.liveness import compute_intervals, peak_bytes
from tgc.errors import ConfigError, ScheduleError
from tgc.ir.graph import Graph, Node
from tgc.memory.planner import (
    STRATEGIES,
    plan_largest_first,
    plan_without_reuse,
    validate_plan,
)

# Choosing which order to run the nodes in.
#
# Every topological order computes the same answer and they do not cost the same. Running a
# branch to completion before starting the next keeps one branch's intermediates alive;
# running the branches in lockstep keeps all of them alive at once, and no allocator can
# recover the difference because the values genuinely do overlap.
#
# That is the point of this file. Buffer allocation gets the attention because it is the part
# that looks like an algorithm, and once any reusing allocator is in place the three sensible
# ones agree on most graphs while the schedule still moves the peak by half as much again.
# Both measurements are here so the comparison is not an assertion, and the first version of
# this claim was wrong: comparing against a plan that never reuses anything makes the
# allocator look decisive everywhere, because that comparison is having an allocator against
# not having one.


@dataclass
class OrderReport:
    """What one execution order costs."""

    name: str
    peak_bytes: int
    arena_bytes: int

    @property
    def allocator_overhead(self) -> float:
        """How much the allocator adds on top of what the order made unavoidable."""
        if self.peak_bytes == 0:
            return 0.0
        return self.arena_bytes / self.peak_bytes - 1.0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "order": self.name,
            "peak_bytes": self.peak_bytes,
            "arena_bytes": self.arena_bytes,
            "allocator_overhead": round(self.allocator_overhead, 4),
        }


def is_valid_order(graph: Graph, order: Sequence[Node]) -> bool:
    """Whether an order runs every node after the ones it reads."""
    if len(order) != len(graph.nodes):
        return False
    seen = {value.name for value in graph.inputs}
    for node in order:
        if any(name not in seen for name in node.inputs):
            return False
        seen.add(node.name)
    return True


def check_order(graph: Graph, order: Sequence[Node]) -> None:
    """Raise if an order cannot be executed."""
    if not is_valid_order(graph, order):
        raise ScheduleError("this order runs a node before something it reads")


def peak_for_order(graph: Graph, order: Sequence[Node]) -> int:
    """Simultaneously live bytes under one order."""
    check_order(graph, order)
    return peak_bytes(compute_intervals(graph, order))


def arena_for_order(
    graph: Graph, order: Sequence[Node], strategy: str = "largest first"
) -> int:
    """Arena size an allocator needs under one order."""
    if strategy not in STRATEGIES:
        raise ConfigError(f"unknown strategy {strategy!r}")
    intervals = compute_intervals(graph, order)
    plan = STRATEGIES[strategy](intervals)
    validate_plan(intervals, plan)
    return plan.arena_bytes


def ready_nodes(graph: Graph, done: set[str]) -> list[Node]:
    """Nodes whose inputs have all been produced."""
    available = {value.name for value in graph.inputs} | done
    return [
        node
        for node in graph.nodes
        if node.name not in done and all(name in available for name in node.inputs)
    ]


def source_order(graph: Graph) -> list[Node]:
    """The order the graph was written in."""
    return list(graph.nodes)


def breadth_first_order(graph: Graph) -> list[Node]:
    """Run everything that is ready before going deeper.

    The order a naive worklist produces, and the worst one for memory on a wide graph: it
    starts every branch before finishing any, so every branch's intermediates are alive at
    the same time.
    """
    done: set[str] = set()
    order: list[Node] = []
    while len(order) < len(graph.nodes):
        wave = ready_nodes(graph, done)
        if not wave:
            raise ScheduleError("the graph has a cycle")
        for node in wave:
            order.append(node)
            done.add(node.name)
    return order


def depth_first_order(graph: Graph) -> list[Node]:
    """Finish what was just started before starting anything else.

    Keeps one branch's intermediates alive rather than all of them. On a graph with any width
    this is most of the memory saving available, and it costs a different traversal rather
    than a better allocator.
    """
    done: set[str] = set()
    order: list[Node] = []
    producers = {node.name: node for node in graph.nodes}

    def emit(node: Node) -> None:
        if node.name in done:
            return
        for name in node.inputs:
            parent = producers.get(name)
            if parent is not None:
                emit(parent)
        if node.name not in done:
            done.add(node.name)
            order.append(node)

    for name in graph.outputs:
        producer = producers.get(name)
        if producer is not None:
            emit(producer)
    for node in graph.nodes:
        emit(node)
    return order


def greedy_min_peak_order(graph: Graph) -> list[Node]:
    """At every step run whichever ready node leaves the least alive afterwards.

    A local rule for a problem that is not local, so it is not optimal and it is cheap. The
    enumeration below says how far off it lands on graphs small enough to check exhaustively,
    which is the only honest way to describe a heuristic.
    """
    done: set[str] = set()
    order: list[Node] = []

    while len(order) < len(graph.nodes):
        candidates = ready_nodes(graph, done)
        if not candidates:
            raise ScheduleError("the graph has a cycle")
        best = min(
            candidates,
            key=lambda node: (_live_after(graph, [*order, node]), node.name),
        )
        order.append(best)
        done.add(best.name)
    return order


def _live_after(graph: Graph, prefix: Sequence[Node]) -> int:
    """Bytes still needed once a prefix of the schedule has run."""
    produced = {node.name for node in prefix}
    available = {value.name for value in graph.inputs} | produced
    remaining = [node for node in graph.nodes if node.name not in produced]
    wanted = set(graph.outputs)
    for node in remaining:
        wanted.update(node.inputs)

    total = 0
    for name in available:
        if name in wanted:
            total += graph.value(name).bytes
    return total


def all_orders(graph: Graph, limit: int = 5000) -> list[list[Node]]:
    """Every valid execution order, up to a limit.

    Exponential, and the limit is not a safety net so much as an admission. It exists to give
    the heuristics something exact to be compared against on small graphs, which is the only
    place an exact answer is available.
    """
    if limit < 1:
        raise ConfigError(f"the limit must be positive, got {limit}")
    results: list[list[Node]] = []

    def extend(order: list[Node], done: set[str]) -> None:
        if len(results) >= limit:
            return
        if len(order) == len(graph.nodes):
            results.append(list(order))
            return
        for node in ready_nodes(graph, done):
            order.append(node)
            done.add(node.name)
            extend(order, done)
            order.pop()
            done.discard(node.name)

    extend([], set())
    return results


def best_order(graph: Graph, limit: int = 5000) -> list[Node]:
    """The order with the smallest peak, found exhaustively."""
    orders = all_orders(graph, limit=limit)
    if not orders:
        raise ScheduleError("the graph has no valid order")
    return min(orders, key=lambda order: peak_for_order(graph, order))


def worst_order(graph: Graph, limit: int = 5000) -> list[Node]:
    """The order with the largest peak, found exhaustively."""
    orders = all_orders(graph, limit=limit)
    if not orders:
        raise ScheduleError("the graph has no valid order")
    return max(orders, key=lambda order: peak_for_order(graph, order))


ORDERINGS = {
    "source": source_order,
    "breadth first": breadth_first_order,
    "depth first": depth_first_order,
    "greedy": greedy_min_peak_order,
}


def compare_orders(graph: Graph) -> list[dict]:
    """Every ordering heuristic on one graph, with the allocator held fixed."""
    rows = []
    for name, ordering in ORDERINGS.items():
        order = ordering(graph)
        rows.append(
            OrderReport(
                name=name,
                peak_bytes=peak_for_order(graph, order),
                arena_bytes=arena_for_order(graph, order),
            ).as_dict()
        )
    return rows


def order_versus_allocator(graph: Graph) -> dict:
    """How much the schedule moves the peak, against how much the allocator does.

    Two spreads measured the same way. The ordering spread holds the allocator fixed and
    varies the schedule; the allocator spread holds the schedule fixed and varies the
    allocator.

    The allocator spread deliberately excludes the plan that never reuses anything. Including
    it makes the allocator look decisive on every graph, because it is comparing having an
    allocator against not having one, and that is not the choice anybody faces. Once any
    reusing allocator is in place, the three sensible ones agree on most graphs and the
    schedule is what is left to win.
    """
    peaks = [peak_for_order(graph, ordering(graph)) for ordering in ORDERINGS.values()]
    order_spread = max(peaks) / min(peaks) if min(peaks) else 1.0

    fixed = depth_first_order(graph)
    intervals = compute_intervals(graph, fixed)
    arenas = []
    for name, strategy in STRATEGIES.items():
        if name == "no reuse":
            continue
        plan = strategy(intervals)
        validate_plan(intervals, plan)
        arenas.append(plan.arena_bytes)
    allocator_spread = max(arenas) / min(arenas) if min(arenas) else 1.0

    return {
        "best_peak": min(peaks),
        "worst_peak": max(peaks),
        "order_spread": round(order_spread, 4),
        "best_arena": min(arenas),
        "worst_arena": max(arenas),
        "allocator_spread": round(allocator_spread, 4),
        "no_reuse_arena": plan_without_reuse(intervals).arena_bytes,
        "order_matters_more": order_spread > allocator_spread,
    }


def scheduled_arena(graph: Graph) -> int:
    """The footprint of the best schedule this compiler produces, allocated well."""
    order = depth_first_order(graph)
    intervals = compute_intervals(graph, order)
    plan = plan_largest_first(intervals)
    validate_plan(intervals, plan)
    return plan.arena_bytes
