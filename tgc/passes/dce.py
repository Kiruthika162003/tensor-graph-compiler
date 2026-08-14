from __future__ import annotations

from dataclasses import dataclass

from tgc.errors import ConfigError
from tgc.ir.graph import Graph, Node, reachable_from_outputs

# Removing work whose result nobody reads.
#
# The pass is four lines and the interesting part is what it must not remove. A node with a
# side effect is still worth running when its value is dead, because the value is not why it
# is there. Deleting it turns a graph that printed a tensor into one that does not, and the
# author of the print statement has no way to tell that a compiler pass is the reason.
#
# The other thing it must not do is remove a node whose value feeds one it has already
# decided to keep. That sounds obvious and is the reason this walks backwards to a fixed
# point rather than filtering the list once.


@dataclass
class DceReport:
    """What dead code elimination removed."""

    removed: list[str]
    kept_for_effects: list[str]

    @property
    def count(self) -> int:
        """Nodes removed."""
        return len(self.removed)

    def as_dict(self) -> dict[str, int | list[str]]:
        """Flat mapping for logging."""
        return {
            "removed": self.count,
            "names": self.removed,
            "kept_for_effects": self.kept_for_effects,
        }


def live_values(graph: Graph) -> set[str]:
    """Every value that has to be computed.

    Seeded from the outputs and from anything with a side effect, then closed backwards.
    Seeding only from the outputs is the version that deletes the print statement.
    """
    live = set(reachable_from_outputs(graph))
    for node in graph.nodes:
        if not node.op.can_be_removed_if_unused:
            live.add(node.name)

    changed = True
    while changed:
        changed = False
        for node in reversed(graph.nodes):
            if node.name in live:
                for name in node.inputs:
                    if name not in live:
                        live.add(name)
                        changed = True
    return live


def eliminate_dead_code(graph: Graph) -> Graph:
    """Drop every node whose value nobody needs."""
    live = live_values(graph)
    return graph.with_nodes([node for node in graph.nodes if node.name in live])


def report_dead_code(graph: Graph) -> DceReport:
    """What the pass would remove, without removing it."""
    live = live_values(graph)
    return DceReport(
        removed=[node.name for node in graph.nodes if node.name not in live],
        kept_for_effects=[
            node.name
            for node in graph.nodes
            if not node.op.can_be_removed_if_unused
            and node.name not in reachable_from_outputs(graph)
        ],
    )


def dead_node_count(graph: Graph) -> int:
    """How many nodes compute nothing anybody reads."""
    return len(graph.nodes) - len(live_values(graph) - {value.name for value in graph.inputs})


def unused_inputs(graph: Graph) -> list[str]:
    """Graph inputs nothing reads.

    Reported rather than removed. An unused input usually means the frontend traced a branch
    that did not run, and silently dropping it changes the signature of the compiled
    function, which breaks the caller rather than the graph.
    """
    live = live_values(graph)
    return [value.name for value in graph.inputs if value.name not in live]


def append_dead_node(graph: Graph, node: Node) -> Graph:
    """Add a node nothing reads, for testing the pass that removes it."""
    if node.name in graph.value_names:
        raise ConfigError(f"{node.name!r} is already defined")
    return graph.with_nodes([*graph.nodes, node])
