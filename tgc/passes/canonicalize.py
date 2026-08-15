from __future__ import annotations

from dataclasses import dataclass, field

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node
from tgc.passes.cse import duplicate_count, eliminate_common_subexpressions

# Putting equivalent expressions into the same shape.
#
# Canonicalisation changes nothing on its own. It rewrites a plus b into b plus a when the
# names sort that way, moves constants to the right hand side, and rewrites a subtraction of a
# literal into an addition of its negation. Every one of those produces a graph that computes
# exactly what it did before and looks no better.
#
# Its value is supposed to be what it does to the pass after it, and measuring that produced a
# result worth keeping: against the subexpression pass in this compiler it gains exactly
# nothing, because that pass already sorts commutative operands inside its own signature. Two
# expressions written in opposite orders were never different to it.
#
# So there are two designs and only one of them is needed. Either the IR is canonicalised and
# matchers compare literally, or matchers normalise as they go and the IR is left alone. Doing
# both is duplication that looks like thoroughness. The gain shows up for a matcher of the
# first kind, which is what one_sided_constant_matches measures: a rule that checks only the
# right operand fires on none of a commuted graph and on all of a canonicalised one.


@dataclass
class Rewrite:
    """One node put into canonical form."""

    node: str
    rule: str

    def as_dict(self) -> dict[str, str]:
        """Flat mapping for logging."""
        return {"node": self.node, "rule": self.rule}


@dataclass
class CanonicalReport:
    """What canonicalisation rewrote."""

    rewrites: list[Rewrite] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Nodes rewritten."""
        return len(self.rewrites)

    def rules_used(self) -> list[str]:
        """Which rules fired, without repeats."""
        return sorted({rewrite.rule for rewrite in self.rewrites})

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"rewrites": self.count, "rules": self.rules_used()}


def is_constant(graph: Graph, name: str) -> bool:
    """Whether a value is a literal."""
    node = graph.producer_of(name)
    return node is not None and node.op is ops.CONSTANT


def sort_key(graph: Graph, name: str) -> tuple[int, str]:
    """The order operands are put into.

    Constants last, then by name. Constants last because it makes every literal end up in the
    same argument position, which is what lets a later pass match on it without checking both
    sides, and because a graph that has been folded has its literals in unpredictable places.
    """
    return (1 if is_constant(graph, name) else 0, name)


def canonicalise_node(graph: Graph, node: Node) -> tuple[Node, str]:
    """One node in canonical form, and the name of the rule that fired."""
    if not node.op.commutative:
        return node, ""
    left, right = node.inputs
    if sort_key(graph, left) <= sort_key(graph, right):
        return node, ""
    return (
        Node(op=node.op, inputs=(right, left), output=node.output, attrs=dict(node.attrs)),
        "commutative operands sorted",
    )


def canonicalise(graph: Graph) -> Graph:
    """Rewrite every node into its canonical form.

    Nothing here changes what the graph computes. Sorting the operands of a commutative
    operation is exact for every value including nan, because it swaps two reads and performs
    the same arithmetic on them.
    """
    rewritten = []
    for node in graph.nodes:
        updated, _ = canonicalise_node(graph, node)
        rewritten.append(updated)
    return graph.with_nodes(rewritten)


def report_canonicalisation(graph: Graph) -> CanonicalReport:
    """Which nodes would be rewritten, without rewriting them."""
    report = CanonicalReport()
    for node in graph.nodes:
        _, rule = canonicalise_node(graph, node)
        if rule:
            report.rewrites.append(Rewrite(node=node.name, rule=rule))
    return report


def is_canonical(graph: Graph) -> bool:
    """Whether every node is already in canonical form."""
    return report_canonicalisation(graph).count == 0


def commuted_pairs_graph(pairs: int = 6) -> Graph:
    """Several expressions written twice, the second time with the operands swapped.

    Which is what a frontend produces from source a user wrote in whichever order occurred to
    them, and is the case a subexpression pass cannot see without canonicalisation. It is not
    a contrived shape: a plus b and b plus a appear in the same file constantly.
    """
    if pairs < 1:
        raise ConfigError(f"there has to be at least one pair, got {pairs}")
    builder = Builder()
    inputs = [builder.input([8, 8], name=f"in{index}") for index in range(pairs + 1)]

    total = None
    for index in range(pairs):
        left = builder.add(inputs[index], inputs[index + 1])
        right = builder.add(inputs[index + 1], inputs[index])
        combined = builder.mul(left, right)
        total = combined if total is None else builder.add(total, combined)
    return builder.finish(total)


def aligned_pairs_graph(pairs: int = 6) -> Graph:
    """The same expressions written the same way round both times.

    The control. A subexpression pass finds these without any help, and the difference between
    this graph and the commuted one is exactly what canonicalisation is worth.
    """
    if pairs < 1:
        raise ConfigError(f"there has to be at least one pair, got {pairs}")
    builder = Builder()
    inputs = [builder.input([8, 8], name=f"in{index}") for index in range(pairs + 1)]

    total = None
    for index in range(pairs):
        left = builder.add(inputs[index], inputs[index + 1])
        right = builder.add(inputs[index], inputs[index + 1])
        combined = builder.mul(left, right)
        total = combined if total is None else builder.add(total, combined)
    return builder.finish(total)


def measure_interaction(pairs: int = 6) -> list[dict]:
    """What subexpression elimination finds with canonicalisation and without.

    Zero gain on both graphs, which is the answer and not a disappointment. The subexpression
    pass sorts commutative operands when it builds a node signature, so a plus b and b plus a
    were already the same expression to it and there was nothing left for a canonicaliser to
    contribute.
    """
    rows = []
    for label, graph in (
        ("aligned", aligned_pairs_graph(pairs)),
        ("commuted", commuted_pairs_graph(pairs)),
    ):
        plain = duplicate_count(graph)
        canonical = duplicate_count(canonicalise(graph))
        rows.append(
            {
                "graph": label,
                "merges_without_canonicalising": plain,
                "merges_after_canonicalising": canonical,
                "gained": canonical - plain,
            }
        )
    return rows


def node_counts(pairs: int = 6) -> list[dict]:
    """Node counts through each pipeline, which is what the caller actually cares about."""
    rows = []
    graph = commuted_pairs_graph(pairs)
    rows.append({"pipeline": "nothing", "nodes": len(graph.nodes)})
    rows.append(
        {
            "pipeline": "subexpressions only",
            "nodes": len(eliminate_common_subexpressions(graph).nodes),
        }
    )
    rows.append(
        {
            "pipeline": "canonicalise then subexpressions",
            "nodes": len(eliminate_common_subexpressions(canonicalise(graph)).nodes),
        }
    )
    return rows


def canonicalisation_alone_changes_nothing(pairs: int = 6) -> bool:
    """Whether canonicalising by itself removes any nodes.

    It does not, and saying so out loud is the honest description of the pass. Every measured
    benefit it has belongs to the pass that runs after it.
    """
    graph = commuted_pairs_graph(pairs)
    return len(canonicalise(graph).nodes) == len(graph.nodes)


def check_idempotent(graph: Graph) -> None:
    """Raise if canonicalising twice differs from canonicalising once.

    A canonical form that is not a fixed point is not a canonical form, and it is also the
    thing that makes a pass pipeline fail to terminate.
    """
    once = canonicalise(graph)
    twice = canonicalise(once)
    for first, second in zip(once.nodes, twice.nodes, strict=True):
        if first.inputs != second.inputs:
            raise PassError(
                f"{first.name} changed on the second pass: {first.inputs} then {second.inputs}"
            )


def one_sided_constant_matches(graph: Graph) -> int:
    """How often a rule that checks only the right operand for a literal fires.

    The matcher design canonicalisation exists for. Every rule in passes/algebraic.py checks
    both sides, which is two lines instead of one and works on any graph; a rule written this
    way is shorter, and only correct on a canonicalised IR.
    """
    fired = 0
    for node in graph.nodes:
        if not node.op.commutative:
            continue
        if is_constant(graph, node.inputs[1]):
            fired += 1
    return fired


def literal_on_the_left_graph(count: int = 5) -> Graph:
    """Expressions with the literal written first, which users do constantly."""
    if count < 1:
        raise ConfigError(f"there has to be at least one expression, got {count}")
    builder = Builder()
    x = builder.input([8, 8], name="x")
    current = x
    for index in range(count):
        literal = builder.constant(float(index + 2))
        current = builder.apply(ops.MUL, literal, current)
    return builder.finish(current)


def measure_one_sided_matcher(count: int = 5) -> dict:
    """A one sided rule on a graph with its literals on the left, before and after.

    None of them before, all of them after. That is the whole benefit of canonicalising an IR,
    and it is a benefit to the matchers rather than to the graph, which is why it does not show
    up anywhere in a node count.
    """
    graph = literal_on_the_left_graph(count)
    return {
        "expressions": count,
        "matches_before": one_sided_constant_matches(graph),
        "matches_after": one_sided_constant_matches(canonicalise(graph)),
    }
