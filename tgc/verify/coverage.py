from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import (
    Builder,
    branching_graph,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT16, FLOAT32
from tgc.ir.graph import Graph

# An audit of what the corpus actually exercises, rather than of what it passes.
#
# Everything in tests/ runs against graphs built by tgc/ir/builder.py, and every pass, every
# rewrite and every cost model is checked on those graphs and no others. That makes the shared
# fixtures the real specification of what this compiler is known to handle, and nobody wrote
# them with that in mind. This counts what they reach.
#
# Six shared fixtures reach fourteen of the thirty operations. Fifteen of the sixteen they miss
# are operations a graph could contain, and I expected the missing ones to be the recent
# additions. They are not. The three added for the gradient and batching passes are missing and
# so are ten that have been in the table from the start, and two whole categories are at zero:
# nothing in the corpus reshapes, transposes, casts, broadcasts, concatenates or slices
# anything, and no fixture carries an assertion. Every pass in this repository has been verified
# on graphs that contain no view operations at all.
#
# Two smaller things fell out of counting. Four operations have no builder method, which means
# no graph anybody writes by hand can contain one and they only enter when a pass inserts them.
# And one operation in the table can never be a node: inputs live on the graph rather than in
# the node list, so a coverage number computed straight off the table caps at twenty nine and it
# took a while to work out why it would not reach thirty. Constants are the opposite and do
# appear as nodes, so the leaf category is half reachable rather than not at all.
#
# The extended corpus at the bottom closes the gap. It is four graphs written by walking the
# table rather than by thinking of a shape, which is a poor way to get coverage and a much
# better position than not knowing.


def ops_in(graph: Graph) -> frozenset[str]:
    """The operation names that appear as nodes in a graph."""
    return frozenset(node.op.name for node in graph.nodes)


def categories_in(graph: Graph) -> frozenset[str]:
    """The operation categories a graph reaches."""
    return frozenset(node.op.category for node in graph.nodes)


def node_counts(graph: Graph) -> dict[str, int]:
    """How many nodes of each operation a graph has."""
    counts: dict[str, int] = {}
    for node in graph.nodes:
        counts[node.op.name] = counts.get(node.op.name, 0) + 1
    return counts


ALL_NAMES = frozenset(operation.name for operation in ops.ALL_OPS)
LEAF_NAMES = frozenset(
    operation.name for operation in ops.ALL_OPS if operation.category == ops.LEAF
)
# An input is a value on the graph and never an entry in the node list, so it is the one
# operation no corpus can reach. A constant is the other leaf and is emitted as a node.
NEVER_A_NODE = frozenset({ops.INPUT.name})
REACHABLE = ALL_NAMES - NEVER_A_NODE


def shared_corpus() -> dict[str, Graph]:
    """The fixtures every test in the suite is written against."""
    return {
        "diamond": diamond_graph(),
        "mlp": mlp_graph(),
        "softmax": softmax_graph(),
        "layernorm": layernorm_graph(),
        "branching": branching_graph(),
        "chain": elementwise_chain(),
    }


def unary_graph(rows: int = 8, columns: int = 32) -> Graph:
    """A graph reaching every unary elementwise operation the table has.

    Written by walking the table rather than by thinking of operations, because the point of
    this file is that thinking of operations is how the gap appeared.
    """
    builder = Builder()
    value = builder.input((rows, columns))
    value = builder.neg(value)
    value = builder.apply(ops.ABS, value)
    value = builder.sqrt(builder.apply(ops.ABS, value))
    value = builder.log(builder.add(value, builder.constant(2.0)))
    value = builder.exp(value)
    value = builder.sigmoid(value)
    value = builder.tanh(value)
    value = builder.relu(value)
    value = builder.step(value)
    value = builder.reciprocal(builder.add(value, builder.constant(1.0)))
    return builder.finish(value)


def binary_graph(rows: int = 8, columns: int = 32) -> Graph:
    """A graph reaching every binary elementwise operation, including the two nothing uses."""
    builder = Builder()
    left = builder.input((rows, columns))
    right = builder.input((rows, columns))
    total = builder.add(left, right)
    total = builder.sub(total, right)
    total = builder.mul(total, left)
    total = builder.div(total, builder.add(right, builder.constant(4.0)))
    total = builder.maximum(total, left)
    total = builder.apply(ops.MINIMUM, total, right)
    return builder.finish(total)


def view_graph(rows: int = 8, columns: int = 32) -> Graph:
    """A graph reaching the view operations, three of which nothing else builds."""
    builder = Builder()
    value = builder.input((rows, columns))
    moved = builder.transpose(value, (1, 0))
    flat = builder.reshape(moved, (columns * rows,))
    half = builder.slice(flat, 0, 0, columns * rows // 2)
    doubled = builder.concat(half, half, 0)
    narrow = builder.cast(doubled, FLOAT16)
    wide = builder.cast(narrow, FLOAT32)
    scalar = builder.mean(wide, (0,))
    spread = builder.broadcast_to(scalar, (rows, columns))
    return builder.finish(builder.add(value, spread))


def checked_graph(rows: int = 8, columns: int = 32) -> Graph:
    """A graph carrying the side effect operations, which no pass builds and two insert."""
    builder = Builder()
    value = builder.input((rows, columns))
    total = builder.sum(value, (1,))
    guarded = builder.apply(ops.ASSERT_FINITE, total)
    watched = builder.apply(ops.PRINT, guarded)
    largest = builder.max(value, (1,))
    return builder.finish(builder.add(watched, largest))


def extended_corpus() -> dict[str, Graph]:
    """The shared fixtures plus the four graphs written to close the gap."""
    return {
        **shared_corpus(),
        "unary": unary_graph(),
        "binary": binary_graph(),
        "view": view_graph(),
        "checked": checked_graph(),
    }


CORPORA: dict[str, Callable[[], dict[str, Graph]]] = {
    "shared": shared_corpus,
    "extended": extended_corpus,
}


def corpus_named(name: str) -> dict[str, Graph]:
    """One of the two corpora, by name."""
    if name not in CORPORA:
        raise ConfigError(f"unknown corpus {name!r}, expected one of {sorted(CORPORA)}")
    return CORPORA[name]()


def covered_by(corpus: dict[str, Graph]) -> frozenset[str]:
    """Every operation that appears somewhere in a corpus."""
    if not corpus:
        raise ConfigError("an empty corpus covers nothing, which is not worth reporting")
    return frozenset().union(*(ops_in(graph) for graph in corpus.values()))


def uncovered_by(corpus: dict[str, Graph]) -> frozenset[str]:
    """Every operation in the table that a corpus never reaches."""
    return ALL_NAMES - covered_by(corpus)


def coverage_of(corpus: dict[str, Graph]) -> float:
    """The share of the table a corpus reaches, leaves included."""
    return len(covered_by(corpus)) / len(ALL_NAMES)


def reachable_coverage_of(corpus: dict[str, Graph]) -> float:
    """The same share, counting only operations that can be nodes.

    The honest denominator, and it drops exactly one operation rather than the whole leaf
    category. Including the input makes every coverage number here one operation short of
    whatever it deserves and makes a complete corpus look incomplete.
    """
    return len(covered_by(corpus) & REACHABLE) / len(REACHABLE)


@dataclass(frozen=True)
class CategoryReport:
    """What one operation category's coverage looks like."""

    category: str
    total: int
    covered: int

    @property
    def share(self) -> float:
        """Fraction of the category reached."""
        if self.total == 0:
            raise ConfigError(f"category {self.category} has no operations in it")
        return self.covered / self.total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "category": self.category,
            "total": self.total,
            "covered": self.covered,
            "share": round(self.share, 4),
        }


def by_category(corpus: dict[str, Graph]) -> list[dict]:
    """Coverage broken down by what kind of operation it is."""
    reached = covered_by(corpus)
    categories = sorted({operation.category for operation in ops.ALL_OPS})
    rows = []
    for category in categories:
        names = {operation.name for operation in ops.ALL_OPS if operation.category == category}
        rows.append(
            CategoryReport(
                category=category, total=len(names), covered=len(names & reached)
            ).as_dict()
        )
    return rows


def builder_methods() -> frozenset[str]:
    """The operations the builder exposes as a method of their own."""
    return frozenset(name for name in ALL_NAMES if hasattr(Builder, name))


def ops_without_a_builder_method() -> frozenset[str]:
    """Operations that exist in the table and cannot be written down.

    Four of them, and the reason is the same in each case: they are inserted by a pass rather
    than requested by a user. That is a reasonable design and it means the only graphs
    containing them are the ones a pass produced, so nothing checks them on a graph a pass did
    not build until this file does.
    """
    return ALL_NAMES - builder_methods() - LEAF_NAMES


def the_shared_fixtures_leave_half_the_table() -> dict:
    """How much of the operation table the suite actually runs on.

    Under half. Six fixtures, every test in the repository, and fifteen operations that could
    appear in a graph and never appear in one of these. Every claim made about a pass is a claim
    about the fourteen it does see.
    """
    corpus = shared_corpus()
    return {
        "graphs": len(corpus),
        "operations": len(ALL_NAMES),
        "covered": len(covered_by(corpus)),
        "coverage": round(coverage_of(corpus), 4),
        "reachable_coverage": round(reachable_coverage_of(corpus), 4),
    }


def the_gap_is_not_what_it_looks_like() -> dict:
    """What is in the gap, sorted into the reason it is there.

    Three groups, and the one I expected to dominate is the smallest. Three operations were
    added for the gradient and batching passes and two are inserted by the assertion pass, so
    five of the fifteen have an excuse. The other ten have been in the table since the first
    commit and no fixture ever needed one, which is a different problem and a worse one.
    """
    missing = uncovered_by(shared_corpus())
    recent = {"step", "concat", "slice"}
    inserted = {"assert_finite", "print"}
    return {
        "missing": sorted(missing),
        "added_for_a_pass": sorted(missing & recent),
        "inserted_by_a_pass": sorted(missing & inserted),
        "just_never_used": sorted(missing - recent - inserted - NEVER_A_NODE),
        "never_a_node": sorted(missing & NEVER_A_NODE),
    }


def one_operation_can_never_be_a_node() -> dict:
    """Why no corpus reaches the whole table, and why only one of the two leaves is the reason.

    An input is a value on the graph rather than an entry in the node list, so no corpus will
    ever contain one and a coverage number that counts it caps below one. A constant is not the
    same: it is emitted as a node, the layer normalisation fixture has one, and it counts. Both
    are leaves and only one of them is unreachable, which is not something the category tells
    you.
    """
    shared = uncovered_by(shared_corpus())
    extended = uncovered_by(extended_corpus())
    return {
        "leaves": sorted(LEAF_NAMES),
        "missing_from_shared": sorted(LEAF_NAMES & shared),
        "missing_from_extended": sorted(LEAF_NAMES & extended),
        "unreachable_either_way": sorted(LEAF_NAMES & shared & extended),
        "reachable_leaves": sorted(LEAF_NAMES - extended),
    }


def four_operations_cannot_be_written_down() -> dict:
    """Which operations have no builder method, and what that means for the tests.

    Absolute value, the elementwise minimum, the finiteness assertion and the print. The first
    two arrive from the gradient rules for the rectifier and the clamp, and the second two from
    the assertion placement pass. Nothing outside those three modules had ever built a graph
    containing one.
    """
    missing = ops_without_a_builder_method()
    return {
        "without_a_method": sorted(missing),
        "count": len(missing),
        "in_the_shared_corpus": sorted(missing & covered_by(shared_corpus())),
        "in_the_extended_corpus": sorted(missing & covered_by(extended_corpus())),
    }


def the_extended_corpus_closes_it() -> dict:
    """What four graphs written against the table are worth.

    Everything that can be reached. The four graphs take the reachable coverage to one, which
    is the number that should have been true before any of the passes were written and is at
    least true now.
    """
    shared = shared_corpus()
    extended = extended_corpus()
    return {
        "shared": round(reachable_coverage_of(shared), 4),
        "extended": round(reachable_coverage_of(extended), 4),
        "still_missing": sorted(uncovered_by(extended) - NEVER_A_NODE),
        "complete": not (uncovered_by(extended) - NEVER_A_NODE),
        "graphs_added": len(extended) - len(shared),
    }


def node_weighted_coverage(corpus: dict[str, Graph]) -> float:
    """The share of nodes in a corpus whose operation the corpus covers.

    One, always, and that is the point of computing it. Weighting coverage by how often
    something appears is the natural thing to do and it answers a question nobody asked: a
    corpus covers every operation it contains by construction. Coverage has to be measured
    against the table.
    """
    counts: dict[str, int] = {}
    for graph in corpus.values():
        for name, count in node_counts(graph).items():
            counts[name] = counts.get(name, 0) + count
    total = sum(counts.values())
    if total == 0:
        raise ConfigError("a corpus with no nodes in it does not weight anything")
    reached = covered_by(corpus)
    return sum(count for name, count in counts.items() if name in reached) / total


def the_common_operations_dominate_the_node_count() -> dict:
    """How lopsided the corpus is, measured in nodes rather than in operations.

    Badly. Three operations account for more than half the nodes in the whole corpus, and the
    weighted coverage is one, which is the number a weighted measure will always produce. A
    corpus covers everything it contains. Coverage only means something against the table.
    """
    corpus = shared_corpus()
    counts: dict[str, int] = {}
    for graph in corpus.values():
        for name, count in node_counts(graph).items():
            counts[name] = counts.get(name, 0) + count
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    return {
        "distinct_operations": len(counts),
        "nodes": total,
        "top_three": [name for name, _ in ranked[:3]],
        "top_three_share": round(sum(count for _, count in ranked[:3]) / total, 4),
        "weighted_coverage": round(node_weighted_coverage(corpus), 4),
    }


def compare_corpora() -> list[dict]:
    """Both corpora, side by side."""
    rows = []
    for name in sorted(CORPORA):
        corpus = corpus_named(name)
        rows.append(
            {
                "corpus": name,
                "graphs": len(corpus),
                "covered": len(covered_by(corpus)),
                "coverage": round(coverage_of(corpus), 4),
                "reachable": round(reachable_coverage_of(corpus), 4),
                "missing": len(uncovered_by(corpus) - NEVER_A_NODE),
            }
        )
    return rows


def per_graph_coverage(corpus: dict[str, Graph] | None = None) -> list[dict]:
    """What each graph in a corpus contributes on its own.

    The interesting column is the last one. Most of the shared fixtures add nothing the others
    did not already have, which is what a corpus grown by writing the next convenient graph
    looks like from the outside.
    """
    target = corpus if corpus is not None else shared_corpus()
    rows = []
    for name, graph in target.items():
        others = {label: other for label, other in target.items() if label != name}
        unique = ops_in(graph) - (covered_by(others) if others else frozenset())
        rows.append(
            {
                "graph": name,
                "nodes": len(graph.nodes),
                "operations": len(ops_in(graph)),
                "categories": sorted(categories_in(graph)),
                "unique": sorted(unique),
            }
        )
    return rows


def the_fixtures_are_not_redundant() -> dict:
    """How many of the shared graphs are the only source of anything.

    Five of the six, which was the opposite of what I expected to find and makes the coverage
    number worse rather than better. The corpus is not padded with graphs that duplicate each
    other. It is six graphs that each pull their weight and together still reach under half the
    table, which means the fix is more graphs and not better ones.
    """
    rows = per_graph_coverage()
    contributing = [row["graph"] for row in rows if row["unique"]]
    return {
        "graphs": len(rows),
        "contributing": contributing,
        "redundant": [row["graph"] for row in rows if not row["unique"]],
        "share_redundant": round(1 - len(contributing) / len(rows), 4),
    }


def every_category_appears_somewhere(corpus: dict[str, Graph] | None = None) -> dict:
    """Whether the corpus at least touches each kind of operation.

    Two categories are empty. Nothing in the shared corpus reshapes, transposes, casts,
    broadcasts, concatenates or slices, and nothing carries an assertion. A category at zero is
    the coarsest signal available and two of the six fire, which is a fair indication of how
    much attention this was getting before somebody counted.
    """
    target = corpus if corpus is not None else shared_corpus()
    rows = by_category(target)
    return {
        "categories": len(rows),
        "empty": [row["category"] for row in rows if row["covered"] == 0],
        "complete": [row["category"] for row in rows if row["share"] == 1.0],
        "partial": [row["category"] for row in rows if 0 < row["share"] < 1.0],
    }


def an_unknown_corpus_is_refused() -> bool:
    """Whether asking for a corpus that does not exist names the ones that do."""
    try:
        corpus_named("everything")
    except ConfigError:
        return True
    return False


def an_empty_corpus_is_refused() -> bool:
    """Whether measuring coverage of nothing is refused rather than reported as zero.

    Zero coverage of an empty corpus is arithmetically right and useless, and it is exactly the
    number a broken corpus loader would produce.
    """
    try:
        covered_by({})
    except ConfigError:
        return True
    return False


@dataclass(frozen=True)
class CoverageReport:
    """One corpus's numbers, packaged."""

    corpus: str
    graphs: int = 0
    covered: int = 0
    missing: tuple[str, ...] = ()

    @property
    def share(self) -> float:
        """Reachable coverage."""
        return self.covered / len(REACHABLE)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "corpus": self.corpus,
            "graphs": self.graphs,
            "covered": self.covered,
            "share": round(self.share, 4),
            "missing": list(self.missing),
        }


def report_for(name: str) -> CoverageReport:
    """Assemble the summary for one corpus."""
    corpus = corpus_named(name)
    return CoverageReport(
        corpus=name,
        graphs=len(corpus),
        covered=len(covered_by(corpus) & REACHABLE),
        missing=tuple(sorted(uncovered_by(corpus) - NEVER_A_NODE)),
    )


def audit(names: Sequence[str] = ("shared", "extended")) -> list[dict]:
    """Both reports, for whatever runs this at the end of a build."""
    if not names:
        raise ConfigError("there is nothing to audit")
    return [report_for(name).as_dict() for name in names]
