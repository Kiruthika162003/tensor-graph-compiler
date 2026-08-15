from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.analysis.cost import annotate_matmuls, node_flops
from tgc.errors import ConfigError
from tgc.ir.builder import (
    Builder,
    branching_graph,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.graph import Graph
from tgc.verify.fuzz import generate_many

# How much of a graph could run at once, and how much of that is worth having.
#
# Two numbers describe the parallelism available in a graph and neither is the number people
# quote. The work is the total cost of every node, which is what a single processor takes. The
# span is the cost of the most expensive chain of dependencies, which is what infinitely many
# processors take, because no amount of hardware shortens a chain. Their ratio is the average
# parallelism, and it bounds any speedup from any schedule on any machine.
#
# The bound is worth stating exactly, because it is the only thing here that is not an estimate.
# With p processors the finishing time is at least the work over p and at least the span, so the
# speedup is at most p and at most the ratio, whichever is smaller. Nothing a scheduler does can
# get past that, and a scheduler that claims to has measured something else.
#
# Run over the fixtures, the answer is that the graphs a compiler is usually shown have almost
# no parallelism in them at all. A chain has a ratio of one by construction. A softmax and a
# layernorm are barely better. The parallelism in real workloads is between examples in a batch
# and between layers of a pipeline, neither of which is visible in a single graph, and a
# compiler that reports the ratio for one graph is reporting something close to one.


@dataclass
class Span:
    """The longest dependency chain through a graph and what it costs."""

    work: float
    span: float
    longest_chain: tuple[str, ...] = ()

    @property
    def parallelism(self) -> float:
        """Work over span: the average number of nodes that could run at once."""
        if self.span <= 0:
            return 0.0
        return self.work / self.span

    @property
    def depth(self) -> int:
        """Nodes on the longest chain."""
        return len(self.longest_chain)

    def speedup_bound(self, processors: int) -> float:
        """The most any schedule could gain from a given number of processors."""
        if processors < 1:
            raise ConfigError(f"there has to be at least one processor, got {processors}")
        return min(float(processors), self.parallelism)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "work": round(self.work, 3),
            "span": round(self.span, 3),
            "parallelism": round(self.parallelism, 3),
            "depth": self.depth,
        }


def node_cost(graph: Graph, name: str) -> float:
    """What one node costs, in the same units the roofline model uses.

    Leaves cost nothing. They are values that already exist, and giving them a cost puts a
    length on a chain that has no work in it.
    """
    node = graph.producer_of(name)
    if node is None or node.op.is_leaf:
        return 0.0
    return node_flops(node)


def total_work(graph: Graph) -> float:
    """What a single processor would take."""
    prepared = annotate_matmuls(graph)
    return sum(node_cost(prepared, node.name) for node in prepared.nodes)


def critical_path(graph: Graph) -> Span:
    """The longest chain of dependencies, weighted by cost.

    One pass in topological order, because the nodes are already sorted and a value's producer
    always appears before it. That invariant is the reason this is linear rather than a search:
    the moment a graph is allowed to be unsorted, the same question needs a full traversal per
    node.
    """
    prepared = annotate_matmuls(graph)
    finish: dict[str, float] = {}
    previous: dict[str, str] = {}
    for value in prepared.inputs:
        finish[value.name] = 0.0

    for node in prepared.nodes:
        best = 0.0
        best_from = ""
        for name in node.inputs:
            if finish.get(name, 0.0) > best:
                best = finish[name]
                best_from = name
        finish[node.name] = best + node_cost(prepared, node.name)
        if best_from:
            previous[node.name] = best_from

    if not finish:
        return Span(work=0.0, span=0.0)

    last = max(finish, key=lambda name: finish[name])
    chain = [last]
    while chain[-1] in previous:
        chain.append(previous[chain[-1]])
    chain = [name for name in reversed(chain) if prepared.producer_of(name) is not None]
    return Span(work=total_work(prepared), span=finish[last], longest_chain=tuple(chain))


def levels(graph: Graph) -> list[list[str]]:
    """The nodes grouped by how deep they sit, counted in edges rather than cost.

    Everything in one level can run at the same time, because nothing in it reads anything else
    in it. The list of level sizes is the shape of the parallelism over time, and it is usually
    a lot lumpier than a single average suggests.
    """
    depth: dict[str, int] = {value.name: 0 for value in graph.inputs}
    grouped: dict[int, list[str]] = {}
    for node in graph.nodes:
        here = max((depth.get(name, 0) for name in node.inputs), default=0) + 1
        depth[node.name] = here
        grouped.setdefault(here, []).append(node.name)
    return [grouped[key] for key in sorted(grouped)]


def level_widths(graph: Graph) -> list[int]:
    """How many nodes sit at each depth."""
    return [len(level) for level in levels(graph)]


def widest_level(graph: Graph) -> int:
    """The most nodes that could ever run at once."""
    return max(level_widths(graph), default=0)


@dataclass
class ParallelismReport:
    """What one graph offers a scheduler."""

    label: str
    work: float = 0.0
    span: float = 0.0
    widest: int = 0
    levels: int = 0
    widths: list[int] = field(default_factory=list)

    @property
    def parallelism(self) -> float:
        """Work over span."""
        if self.span <= 0:
            return 0.0
        return self.work / self.span

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "graph": self.label,
            "parallelism": round(self.parallelism, 3),
            "widest_level": self.widest,
            "levels": self.levels,
        }


def analyse(graph: Graph, label: str = "") -> ParallelismReport:
    """Every parallelism number for one graph."""
    result = critical_path(graph)
    widths = level_widths(graph)
    return ParallelismReport(
        label=label,
        work=result.work,
        span=result.span,
        widest=max(widths, default=0),
        levels=len(widths),
        widths=widths,
    )


def compare_graphs() -> list[dict]:
    """How much parallelism each fixture has.

    Almost none, in every case that resembles a layer. That is the finding worth carrying into
    any scheduling work: a compiler looking at one graph of one example is looking at something
    with a ratio near one, and every claim about a clever schedule has to fit under that.
    """
    return [
        analyse(graph, label).as_dict()
        for label, graph in (
            ("chain", elementwise_chain(8)),
            ("diamond", diamond_graph()),
            ("softmax", softmax_graph()),
            ("layernorm", layernorm_graph()),
            ("mlp", mlp_graph()),
            ("branching", branching_graph()),
        )
    ]


def a_chain_has_no_parallelism() -> dict:
    """The degenerate case, stated as a measurement.

    Every node reads the one before it, so the span is the work and the ratio is exactly one.
    Any scheduler reporting a gain on this graph is reporting a measurement error.
    """
    result = critical_path(elementwise_chain(8))
    return {
        "parallelism": result.parallelism,
        "depth": result.depth,
        "nodes": len(elementwise_chain(8).nodes),
    }


def branching_is_the_only_fixture_with_any() -> dict:
    """Which fixture a scheduler could actually do something with."""
    rows = {row["graph"]: row["parallelism"] for row in compare_graphs()}
    best = max(rows, key=lambda name: rows[name])
    return {
        "best": best,
        "its_parallelism": rows[best],
        "next_best": sorted(rows.values())[-2],
    }


def speedup_bounds(graph: Graph, counts: Sequence[int] = (1, 2, 4, 8, 16, 64)) -> list[dict]:
    """The most any schedule could gain, per processor count.

    Brent's bound written out. It climbs with the processor count until it hits the ratio and
    then stops, and where it stops is a property of the graph rather than of the machine.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    result = critical_path(graph)
    return [
        {
            "processors": count,
            "bound": round(result.speedup_bound(count), 3),
            "limited_by": "processors" if count <= result.parallelism else "the graph",
        }
        for count in counts
    ]


def where_more_processors_stop_helping(graph: Graph) -> int:
    """The processor count past which the graph is the limit.

    The ceiling of the ratio. Beyond it a machine is buying nothing, and the useful thing about
    having the number is that it is knowable before any hardware is chosen.
    """
    parallelism = critical_path(graph).parallelism
    if parallelism <= 0:
        return 1
    whole = int(parallelism)
    return whole if whole == parallelism else whole + 1


def diminishing_returns() -> list[dict]:
    """Where each fixture stops using more processors."""
    return [
        {"graph": label, "useful_processors": where_more_processors_stop_helping(graph)}
        for label, graph in (
            ("chain", elementwise_chain(8)),
            ("softmax", softmax_graph()),
            ("layernorm", layernorm_graph()),
            ("mlp", mlp_graph()),
            ("branching", branching_graph()),
        )
    ]


def the_chain_is_the_whole_graph(graph: Graph) -> dict:
    """What share of a graph's nodes lie on its longest chain.

    Almost all of them on anything that looks like a layer, which is another way of saying the
    same thing the ratio says, and a more legible one: if nine nodes out of ten are on the
    critical path then nine nodes out of ten have to run one after another.
    """
    result = critical_path(graph)
    total = len(graph.nodes)
    return {
        "nodes": total,
        "on_the_chain": result.depth,
        "share": round(result.depth / total, 4) if total else 0.0,
    }


def chain_share_by_graph() -> list[dict]:
    """The share of nodes on the critical path, per fixture."""
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(8)),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
        ("branching", branching_graph()),
    ):
        row = the_chain_is_the_whole_graph(graph)
        row["graph"] = label
        rows.append(row)
    return rows


def unbalanced_graph(cheap: int = 10, size: int = 64) -> Graph:
    """A long cheap branch and a short expensive one, rejoined.

    Built to separate the two ways of measuring a path. The cheap branch is ten elementwise
    operations and the expensive one is two matrix products, so the first is longer counted in
    nodes and the second is longer counted in time by a factor of the matrix dimension. Nothing
    in the ordinary fixture set makes that distinction, because they are all close to chains and
    a chain has only one path to choose.
    """
    if cheap < 1:
        raise ConfigError(f"the cheap branch needs some length, got {cheap}")
    builder = Builder()
    x = builder.input([size, size], name="x")
    weight = builder.input([size, size], name="w")

    slow = x
    for _ in range(cheap):
        slow = builder.tanh(slow)

    fast = builder.matmul(builder.matmul(x, weight), weight)
    return builder.finish(builder.add(slow, fast))


def longest_chain_by_nodes(graph: Graph) -> tuple[str, ...]:
    """The longest path counted in edges rather than in cost."""
    depth: dict[str, int] = {value.name: 0 for value in graph.inputs}
    previous: dict[str, str] = {}
    for node in graph.nodes:
        best = 0
        best_from = ""
        for name in node.inputs:
            if depth.get(name, 0) >= best and name in depth:
                best = depth[name]
                best_from = name
        depth[node.name] = best + 1
        if best_from:
            previous[node.name] = best_from

    if not depth:
        return ()
    last = max(depth, key=lambda name: depth[name])
    chain = [last]
    while chain[-1] in previous:
        chain.append(previous[chain[-1]])
    return tuple(name for name in reversed(chain) if graph.producer_of(name) is not None)


def cost_weighting_changes_the_answer(graph: Graph) -> dict:
    """Whether the longest chain by cost is the longest chain by node count.

    The point of weighting at all. A path through two matrix products is shorter in nodes and
    far longer in time than a path through ten elementwise operations, and a scheduler that
    counts nodes will set about shortening the wrong one.
    """
    weighted = set(critical_path(graph).longest_chain)
    counted = set(longest_chain_by_nodes(graph))
    return {
        "weighted_length": len(weighted),
        "counted_length": len(counted),
        "same_path": weighted == counted,
    }


def weighting_matters_somewhere() -> list[dict]:
    """Where counting nodes and counting cost disagree about the longest path.

    On every ordinary fixture they agree, which is not a reason to skip the weighting. They
    agree because those graphs have one long path and no choice to get wrong. The unbalanced
    fixture has a choice and the two answers differ, which is the case a real model is full of.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(8)),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
        ("branching", branching_graph()),
        ("unbalanced", unbalanced_graph()),
    ):
        row = cost_weighting_changes_the_answer(graph)
        row["graph"] = label
        rows.append(row)
    return rows


def parallelism_on_generated_graphs(count: int = 24) -> dict:
    """Whether graphs nobody wrote have more parallelism than the fixtures.

    They do, and it does not mean anything encouraging. The fuzzer builds nodes from whatever is
    available rather than from what a person would write next, so it produces wide shallow
    graphs that no model resembles. It is worth measuring precisely so the fixtures are not
    mistaken for a biased sample when they are the representative one.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    ratios = [critical_path(graph).parallelism for graph in generate_many(count)]
    fixtures = [row["parallelism"] for row in compare_graphs()]
    return {
        "generated": len(ratios),
        "generated_mean": round(sum(ratios) / len(ratios), 3),
        "fixture_mean": round(sum(fixtures) / len(fixtures), 3),
    }
