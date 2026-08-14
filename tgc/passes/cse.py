from __future__ import annotations

from dataclasses import dataclass, field

from tgc.ir.graph import Graph, Node

# Computing the same thing twice, once.
#
# The pass is a dictionary from a node's signature to the value that already holds its
# result. Everything difficult is in what goes into the signature, which lives on the node
# rather than here, because the same question is asked by every pass that wants to know
# whether two nodes are interchangeable.
#
# The traps, in the order people fall into them: attributes have to be part of the key, or a
# sum over rows collides with a sum over columns. Output type has to be part of it, or a cast
# collides with its own input. Impure nodes must never be candidates, because two prints are
# two prints. Commutative operands should be sorted, which is not a trap but is where most of
# the wins come from once a frontend has emitted the same expression with its arguments in
# whichever order the user wrote them.


@dataclass
class CseReport:
    """What common subexpression elimination merged."""

    merged: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Nodes eliminated."""
        return len(self.merged)

    def as_dict(self) -> dict[str, int | dict[str, str]]:
        """Flat mapping for logging."""
        return {"merged": self.count, "mapping": dict(self.merged)}


def is_candidate(node: Node) -> bool:
    """Whether a node may be merged with an identical one.

    Impure nodes never are. Two prints of the same tensor are two prints, and a pass that
    merges them removes an output the author asked for.
    """
    return node.op.pure and not node.op.is_leaf


def find_duplicates(graph: Graph) -> dict[str, str]:
    """A map from each redundant value to the one that already holds its result."""
    seen: dict[tuple, str] = {}
    replacement: dict[str, str] = {}

    for node in graph.nodes:
        rewritten = node.replace_inputs(replacement)
        if not is_candidate(node):
            continue
        key = rewritten.signature()
        if key in seen:
            replacement[node.name] = seen[key]
        else:
            seen[key] = node.name
    return replacement


def eliminate_common_subexpressions(graph: Graph) -> Graph:
    """Replace every repeated computation with the first one.

    Rewriting as it goes rather than in a second sweep is what makes it work on a chain: once
    the first duplicate is merged, the nodes reading it become identical to their
    counterparts, and those merge in the same pass instead of needing another round.
    """
    seen: dict[tuple, str] = {}
    replacement: dict[str, str] = {}
    kept: list[Node] = []

    for node in graph.nodes:
        rewritten = node.replace_inputs(replacement)
        if not is_candidate(node):
            kept.append(rewritten)
            continue
        key = rewritten.signature()
        if key in seen:
            replacement[node.name] = seen[key]
            continue
        seen[key] = node.name
        kept.append(rewritten)

    outputs = [replacement.get(name, name) for name in graph.outputs]
    return Graph(nodes=kept, inputs=list(graph.inputs), outputs=outputs)


def report_common_subexpressions(graph: Graph) -> CseReport:
    """What the pass would merge, without merging it."""
    return CseReport(merged=find_duplicates(graph))


def duplicate_count(graph: Graph) -> int:
    """How many nodes recompute something the graph already has."""
    return len(find_duplicates(graph))


def signature_groups(graph: Graph) -> dict[tuple, list[str]]:
    """Every node grouped by what it computes.

    Useful for seeing why a merge did not happen. A group of one where two were expected
    points at the part of the signature that differed, which is usually an attribute.
    """
    groups: dict[tuple, list[str]] = {}
    for node in graph.nodes:
        if not is_candidate(node):
            continue
        groups.setdefault(node.signature(), []).append(node.name)
    return groups
