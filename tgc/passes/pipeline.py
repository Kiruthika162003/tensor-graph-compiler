from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph
from tgc.passes.algebraic import simplify
from tgc.passes.canonicalize import canonicalise
from tgc.passes.constfold import fold_constants
from tgc.passes.cse import eliminate_common_subexpressions
from tgc.passes.dce import eliminate_dead_code
from tgc.passes.hoist import sink_broadcasts
from tgc.passes.layout import cancel_transposes
from tgc.passes.manager import Pass, Pipeline
from tgc.verify.fuzz import generate_many
from tgc.verify.reference import outputs_agree, random_feeds, run

# Which order the passes run in, and whether it matters.
#
# Pass ordering is the part of a compiler everybody has an opinion about and almost nobody
# measures. The opinions are reasonable: folding before simplifying exposes identities that
# only appear once the literals are known, and removing dead code first means every later pass
# has less to look at. Both are true and neither says how much.
#
# So this file enumerates orderings and runs them. Over the exact passes in this compiler the
# answer turns out to be that the order barely matters for the result and matters a lot for
# how many rounds it takes to settle, because the pipeline runs to a fixed point and a bad
# order simply needs more passes over the same graph to reach the same place.

EXACT_PASSES = {
    "canonicalise": canonicalise,
    "constant folding": fold_constants,
    "algebraic": simplify,
    "transposes": cancel_transposes,
    "broadcasts": sink_broadcasts,
    "subexpressions": eliminate_common_subexpressions,
    "dead code": eliminate_dead_code,
}


def pipeline_from(names: Sequence[str], *, max_rounds: int = 12) -> Pipeline:
    """A pipeline running the named passes in the order given."""
    unknown = [name for name in names if name not in EXACT_PASSES]
    if unknown:
        raise ConfigError(f"unknown passes {unknown}, expected some of {sorted(EXACT_PASSES)}")
    if not names:
        raise ConfigError("a pipeline needs at least one pass")
    return Pipeline(
        passes=[Pass(name=name, transform=EXACT_PASSES[name]) for name in names],
        max_rounds=max_rounds,
    )


DEFAULT_ORDER = (
    "canonicalise",
    "constant folding",
    "algebraic",
    "transposes",
    "broadcasts",
    "subexpressions",
    "dead code",
)


@dataclass
class OrderResult:
    """What one pass ordering achieved on one graph."""

    order: tuple[str, ...]
    nodes: int
    rounds: int
    passes_run: int

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "order": " then ".join(self.order),
            "nodes": self.nodes,
            "rounds": self.rounds,
            "passes_run": self.passes_run,
        }


def run_order(graph: Graph, names: Sequence[str], *, max_rounds: int = 12) -> OrderResult:
    """Run one ordering to its fixed point and report what it took."""
    pipeline = pipeline_from(names, max_rounds=max_rounds)
    result, report = pipeline.run(graph)
    return OrderResult(
        order=tuple(names),
        nodes=len(result.nodes),
        rounds=report.rounds,
        passes_run=len(report.results),
    )


def messy_graph(depth: int = 4) -> Graph:
    """A graph holding something for every pass to find.

    Repeated subexpressions, a foldable literal chain, a multiplication by one that only
    appears once the literals are folded, a pair of transposes and a broadcast read only by
    elementwise operations. Built so that no single pass can finish the job alone, which is the
    condition under which ordering could possibly matter.
    """
    if depth < 1:
        raise ConfigError(f"the graph needs some depth, got {depth}")
    builder = Builder()
    x = builder.input([8, 8], name="x")

    scale = builder.mul(builder.constant(4.0), builder.constant(0.25))
    scaled = builder.mul(x, scale)

    column = builder.sum(x, axes=[1], keepdims=True)
    wide = builder.broadcast_to(column, [8, 8])

    current = scaled
    for _ in range(depth):
        first = builder.add(current, wide)
        second = builder.add(wide, current)
        current = builder.mul(first, second)

    once = builder.transpose(current, [1, 0])
    twice = builder.transpose(once, [1, 0])
    return builder.finish(builder.relu(twice))


def compare_orders(
    graph: Graph | None = None, *, sample: int = 24, seed: int = 0
) -> list[dict]:
    """Several orderings of the same passes on the same graph.

    Sampled rather than exhaustive, because seven passes have five thousand orderings and the
    interesting spread shows up in the first few dozen.
    """
    target = graph if graph is not None else messy_graph()
    if sample < 1:
        raise ConfigError(f"the sample must be positive, got {sample}")

    generator = random.Random(seed)
    orders = [tuple(DEFAULT_ORDER)]
    while len(orders) < sample:
        shuffled = list(DEFAULT_ORDER)
        generator.shuffle(shuffled)
        candidate = tuple(shuffled)
        if candidate not in orders:
            orders.append(candidate)
    return [run_order(target, order).as_dict() for order in orders]


def order_spread(graph: Graph | None = None, *, sample: int = 24) -> dict:
    """How much the ordering changes the result and how much it changes the work.

    Two numbers that behave differently. Every ordering reaches the same node count, because
    the pipeline runs to a fixed point and the passes are confluent over this graph. The work
    it takes to get there is not the same at all, and that is the cost a bad order carries.
    """
    rows = compare_orders(graph, sample=sample)
    nodes = [row["nodes"] for row in rows]
    rounds = [row["rounds"] for row in rows]
    return {
        "orderings": len(rows),
        "best_nodes": min(nodes),
        "worst_nodes": max(nodes),
        "same_result": min(nodes) == max(nodes),
        "fewest_rounds": min(rounds),
        "most_rounds": max(rounds),
        "round_spread": round(max(rounds) / min(rounds), 3) if min(rounds) else 0.0,
    }


def default_is_competitive(graph: Graph | None = None, *, sample: int = 24) -> dict:
    """Whether the ordering in the default pipeline is a good one.

    The reason to write this down rather than to argue about it. If the default sits at the
    fewest rounds then the reasoning behind it was right; if it does not, the reasoning was a
    story and the measurement is the correction.
    """
    rows = compare_orders(graph, sample=sample)
    default = rows[0]
    best = min(row["rounds"] for row in rows)
    return {
        "default_rounds": default["rounds"],
        "best_rounds": best,
        "default_is_best": default["rounds"] == best,
        "extra_rounds": default["rounds"] - best,
    }


def single_pass_is_not_enough(graph: Graph | None = None) -> list[dict]:
    """What each pass achieves on its own.

    None of them finishes the job, which is the condition that makes ordering a question at
    all. A graph where one pass does everything has no ordering problem to study.
    """
    target = graph if graph is not None else messy_graph()
    rows = [{"pipeline": "nothing", "nodes": len(target.nodes)}]
    for name in DEFAULT_ORDER:
        result, _ = pipeline_from([name]).run(target)
        rows.append({"pipeline": name, "nodes": len(result.nodes)})
    rows.append({"pipeline": "all of them", "nodes": run_order(target, DEFAULT_ORDER).nodes})
    return rows


def every_order_preserves_the_answer(graph: Graph | None = None, *, sample: int = 12) -> bool:
    """Whether every sampled ordering computes the same thing.

    Bit equality, because every pass here is exact. An ordering that changed the answer would
    mean one of them is not, and the confluence result above would be describing a bug rather
    than a property.
    """
    target = graph if graph is not None else messy_graph()
    feeds = random_feeds(target, positive=True)
    expected = run(target, feeds)

    for row in compare_orders(target, sample=sample):
        order = tuple(row["order"].split(" then "))
        result, _ = pipeline_from(order).run(target)
        if not outputs_agree(run(result, feeds), expected):
            return False
    return True


def confluence_on_generated_graphs(count: int = 12, *, sample: int = 6) -> dict:
    """Whether orderings agree on graphs nobody wrote.

    The fixture above was built to give every pass something to do, which is exactly the kind
    of graph where a confluence claim is most likely to be an accident of the fixture.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    disagreements = 0
    checked = 0
    for graph in generate_many(count):
        sizes = {row["nodes"] for row in compare_orders(graph, sample=sample)}
        checked += 1
        if len(sizes) > 1:
            disagreements += 1
    return {
        "graphs": checked,
        "orderings_each": sample,
        "graphs_where_order_changed_the_result": disagreements,
    }


def exhaustive_orders(names: Sequence[str], graph: Graph) -> list[dict]:
    """Every ordering of a small set of passes.

    Only usable for four or five passes. It exists so the sampled comparison above can be
    checked against a complete one on a subset rather than trusted.
    """
    if len(names) > 5:
        raise PassError(
            f"{len(names)} passes have too many orderings to enumerate, use compare_orders"
        )
    return [run_order(graph, order).as_dict() for order in itertools.permutations(names)]


@dataclass
class PipelineComparison:
    """The default pipeline against a shuffled one."""

    rows: list[dict] = field(default_factory=list)

    @property
    def best_rounds(self) -> int:
        """Fewest rounds any ordering needed."""
        return min((row["rounds"] for row in self.rows), default=0)

    @property
    def worst_rounds(self) -> int:
        """Most rounds any ordering needed."""
        return max((row["rounds"] for row in self.rows), default=0)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "orderings": len(self.rows),
            "best_rounds": self.best_rounds,
            "worst_rounds": self.worst_rounds,
        }
