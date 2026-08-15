from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, GraphError
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
from tgc.ir.graph import Graph, Node
from tgc.passes.algebraic import simplify
from tgc.passes.constfold import fold_constants
from tgc.passes.dce import eliminate_dead_code
from tgc.verify.fuzz import generate_many
from tgc.verify.reference import outputs_agree, random_feeds, run

# Deciding whether two graphs are the same graph, without looking at the names.
#
# Value names are invented by the builder from a counter, so the same computation written twice
# gets different names both times and nothing that compares names can tell they match. What can
# is a hash computed bottom up: a node's identity is its operation, its attributes and the
# identities of its inputs, which makes the whole thing a tree hash over a graph and makes two
# structurally identical graphs collide by construction.
#
# The reason to want it is that it turns several separate questions into one lookup. Whether a
# subexpression already exists, whether a subgraph repeats, whether a pass changed anything,
# whether a compiled artefact in a cache is the one being asked for. All of them are the same
# question about structural identity and all of them are wrong if they compare names.
#
# The thing to be careful about is what it does not say. Two graphs with the same hash compute
# the same thing; two graphs that compute the same thing need not have the same hash. A graph
# and its optimised form are the standard example, and they are not a defect: rewriting one
# into the other is what the optimiser is for. Anything using this as a cache key has to key on
# the graph at a fixed point in the pipeline rather than on whatever came in.
#
# One thing fell out of the collision measurement that was not designed in. The hash is taken
# over the outputs, so work nobody reads is not part of the identity, and eleven percent of two
# hundred generated graphs turned out to share a hash for exactly that reason. That is the
# behaviour a cache key wants and it was worth checking rather than assuming.

DIGEST_BITS = 64


def _digest(text: str) -> str:
    """A short stable hash of a string.

    Truncated rather than full length, deliberately. A cache key that is longer than the thing
    it identifies is a cache key nobody stores, and the collision measurement below is what
    justifies the truncation rather than a guess about how many bits are enough.
    """
    return hashlib.blake2b(text.encode(), digest_size=DIGEST_BITS // 8).hexdigest()


def _attrs_text(node: Node) -> str:
    """A node's attributes in a stable order."""
    return ",".join(f"{key}={value}" for key, value in sorted(node.attrs.items()))


def node_identities(graph: Graph) -> dict[str, str]:
    """A name independent identity for every value in a graph.

    Inputs are identified by position rather than by name, so two graphs whose inputs are called
    different things still match. That is a choice with a consequence: it means a graph and the
    same graph with its two inputs swapped hash the same only if the swap is genuinely
    unobservable, which for a subtraction it is not, and the position is what carries that.
    """
    identities: dict[str, str] = {}
    for index, value in enumerate(graph.inputs):
        identities[value.name] = _digest(f"input:{index}:{value.dtype.name}:{value.shape}")

    for node in graph.nodes:
        parts = [identities[name] for name in node.inputs]
        if node.op.commutative:
            parts.sort()
        body = f"{node.op.name}|{_attrs_text(node)}|{node.output.dtype.name}|{'.'.join(parts)}"
        identities[node.name] = _digest(body)
    return identities


def structural_hash(graph: Graph) -> str:
    """One hash for a whole graph, over its outputs in order."""
    identities = node_identities(graph)
    missing = [name for name in graph.outputs if name not in identities]
    if missing:
        raise GraphError(f"{missing} are outputs with no identity")
    return _digest("|".join(identities[name] for name in graph.outputs))


def are_isomorphic(left: Graph, right: Graph) -> bool:
    """Whether two graphs are the same computation under a renaming."""
    return structural_hash(left) == structural_hash(right)


def renamed(graph: Graph, prefix: str = "t") -> Graph:
    """The same graph with every value renamed.

    Exists to be compared against the original. A structural hash that changed under this would
    be hashing the names, which is the failure the whole file is written to avoid, and the only
    way to know it does not is to try it.
    """
    if not prefix:
        raise ConfigError("a prefix cannot be empty")
    builder = Builder()
    builder.prefix = prefix
    mapping: dict[str, str] = {}
    for value in graph.inputs:
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=f"{prefix}_{value.name}",
        )
    for node in graph.nodes:
        if node.op is ops.CONSTANT:
            mapping[node.name] = builder.constant(
                float(node.attrs["value"]), dtype=node.output.dtype
            )
            continue
        operands = [mapping[name] for name in node.inputs]
        mapping[node.name] = builder.apply(node.op, *operands, **node.attrs)
    return builder.finish(*[mapping[name] for name in graph.outputs])


def renaming_changes_nothing() -> list[dict]:
    """Every fixture against a renamed copy of itself."""
    return [
        {"graph": label, "same_hash": are_isomorphic(graph, renamed(graph))}
        for label, graph in _fixtures()
    ]


def _fixtures() -> list[tuple[str, Graph]]:
    """The graphs used throughout this file."""
    return [
        ("chain", elementwise_chain(6)),
        ("diamond", diamond_graph()),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
        ("branching", branching_graph()),
    ]


def different_graphs_hash_differently() -> dict:
    """Whether the fixtures all get distinct hashes."""
    hashes = {label: structural_hash(graph) for label, graph in _fixtures()}
    return {"graphs": len(hashes), "distinct": len(set(hashes.values()))}


def commutativity_is_absorbed() -> dict:
    """Whether a plus b and b plus a hash the same, and a minus b does not.

    The one place the hash is allowed to see through a difference. Sorting the operands of a
    commutative op is what makes two spellings of the same sum collide; doing it for a
    subtraction would make two different computations collide, which is a hash that lies.
    """
    builder = Builder()
    x = builder.input([8, 8], name="x")
    y = builder.input([8, 8], name="y")
    forwards = builder.finish(builder.add(x, y))

    other = Builder()
    a = other.input([8, 8], name="a")
    b = other.input([8, 8], name="b")
    backwards = other.finish(other.add(b, a))

    third = Builder()
    p = third.input([8, 8], name="p")
    q = third.input([8, 8], name="q")
    left_difference = third.finish(third.sub(p, q))

    fourth = Builder()
    r = fourth.input([8, 8], name="r")
    s = fourth.input([8, 8], name="s")
    right_difference = fourth.finish(fourth.sub(s, r))

    return {
        "addition_commutes": are_isomorphic(forwards, backwards),
        "subtraction_does_not": not are_isomorphic(left_difference, right_difference),
    }


def attributes_are_part_of_the_identity() -> dict:
    """Whether a sum over axis zero and one over axis one hash differently.

    They have to. It is the same op reading the same value and it is not the same value, and a
    hash that ignores the attributes is the same mistake as a subexpression pass that ignores
    them, arriving by a different route.
    """
    builder = Builder()
    x = builder.input([8, 8], name="x")
    down = builder.finish(builder.sum(x, axes=[0], keepdims=True))

    other = Builder()
    y = other.input([8, 8], name="y")
    across = other.finish(other.sum(y, axes=[1], keepdims=True))
    return {"different_axes_differ": not are_isomorphic(down, across)}


def shapes_are_part_of_the_identity() -> dict:
    """Whether the same operations on different shapes hash differently."""
    small = elementwise_chain(4, sizes=(8, 8))
    large = elementwise_chain(4, sizes=(64, 64))
    return {"different_shapes_differ": not are_isomorphic(small, large)}


@dataclass
class SubgraphReport:
    """Repeated structure inside one graph."""

    groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def repeated(self) -> int:
        """Identities appearing more than once."""
        return sum(1 for names in self.groups.values() if len(names) > 1)

    @property
    def duplicate_nodes(self) -> int:
        """Nodes that are a repeat of an earlier one."""
        return sum(len(names) - 1 for names in self.groups.values() if len(names) > 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "distinct": len(self.groups),
            "repeated": self.repeated,
            "duplicate_nodes": self.duplicate_nodes,
        }


def repeated_subgraphs(graph: Graph) -> SubgraphReport:
    """Values inside one graph that compute the same thing.

    The same question a subexpression pass asks, arrived at from the other end. A pass compares
    a node against the ones before it; this groups every node by identity in one pass and reads
    the answer off, which costs the same and says more, because the group sizes say how much
    repetition there is rather than only that there is some.
    """
    groups: dict[str, list[str]] = {}
    for name, identity in node_identities(graph).items():
        if graph.producer_of(name) is None:
            continue
        groups.setdefault(identity, []).append(name)
    return SubgraphReport(groups=groups)


def duplication_by_graph() -> list[dict]:
    """How much repeated structure each fixture holds.

    Very little in most of them, which is the honest answer and not the flattering one. The
    fixtures are single layers written once. Repetition at the scale that makes outlining worth
    doing lives between layers of a model, and a graph of one layer cannot show it.
    """
    rows = []
    for label, graph in _fixtures():
        row = repeated_subgraphs(graph).as_dict()
        row["graph"] = label
        row["nodes"] = len(graph.nodes)
        rows.append(row)
    return rows


def stacked_layers(count: int = 4) -> Graph:
    """The same layer applied several times, which is what a model is.

    Written because nothing else here has repetition in it. Each layer is the same sequence of
    operations on a different value, so no two nodes compute the same thing and a subexpression
    pass finds nothing, while the structure repeats exactly and an outliner finds everything.
    That distinction is the whole reason the two passes are different passes.
    """
    if count < 1:
        raise ConfigError(f"there has to be at least one layer, got {count}")
    builder = Builder()
    current = builder.input([16, 16], name="x")
    weight = builder.input([16, 16], name="w")
    for _ in range(count):
        current = builder.tanh(builder.matmul(current, weight))
    return builder.finish(current)


def layer_shapes_repeat_but_values_do_not() -> dict:
    """Why a subexpression pass finds nothing in a stack of identical layers.

    Every layer computes something different, because each reads the output of the one before
    it. The shape of the computation repeats and the values do not, so identity by value finds
    nothing and identity by shape finds everything.
    """
    graph = stacked_layers()
    by_value = repeated_subgraphs(graph).duplicate_nodes
    by_shape = op_pattern_report(graph)
    return {
        "nodes": len(graph.nodes),
        "duplicate_values": by_value,
        "repeated_patterns": by_shape["repeated"],
    }


def op_patterns(graph: Graph, length: int = 3) -> dict[str, list[str]]:
    """Runs of operations grouped by the sequence of ops they perform.

    Ignores which values are involved and looks only at the shape of the computation, which is
    what an outliner needs and what a subexpression pass must not use. A run is keyed by the op
    names and attributes along it, so a matmul then a tanh matches another matmul then tanh
    wherever it appears.
    """
    if length < 1:
        raise ConfigError(f"a pattern needs some length, got {length}")
    groups: dict[str, list[str]] = {}
    nodes = graph.nodes
    for index in range(len(nodes) - length + 1):
        window = nodes[index : index + length]
        if not _is_a_chain(window):
            continue
        parts = [f"{node.op.name}:{_attrs_text(node)}:{node.output.shape}" for node in window]
        key = _digest("|".join(parts))
        groups.setdefault(key, []).append(window[0].name)
    return groups


def _is_a_chain(window: Sequence[Node]) -> bool:
    """Whether each node in a window reads the one before it."""
    return all(earlier.name in later.inputs for earlier, later in itertools.pairwise(window))


def op_pattern_report(graph: Graph, length: int = 3) -> dict:
    """How many runs of a given length repeat in a graph."""
    groups = op_patterns(graph, length)
    return {
        "windows": sum(len(names) for names in groups.values()),
        "distinct": len(groups),
        "repeated": sum(1 for names in groups.values() if len(names) > 1),
        "largest_group": max((len(names) for names in groups.values()), default=0),
    }


def pattern_length_sweep(lengths: Sequence[int] = (2, 3, 4, 6)) -> list[dict]:
    """How the repetition found depends on how long a pattern has to be.

    Longer patterns repeat less, which is arithmetic rather than a finding. What is worth having
    is where it falls off: a stack of four identical two operation layers has every window of
    two and three repeating and nothing at six, because six operations spans a layer boundary
    and the boundary is where the pattern stops.
    """
    if not lengths:
        raise ConfigError("there is nothing to sweep")
    graph = stacked_layers()
    rows = []
    for length in lengths:
        row = op_pattern_report(graph, length)
        row["length"] = length
        rows.append(row)
    return rows


def same_hash_means_same_answer(count: int = 24, *, seed: int = 0) -> dict:
    """Whether two graphs with one hash really do compute one thing.

    The direction the hash is allowed to promise. Checked by hashing a set of generated graphs
    and their renamed copies and running both on the same inputs, because a hash that collided
    on two different computations would be worse than no hash at all.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    agreed = 0
    checked = 0
    for graph in generate_many(count, start=seed):
        copy = renamed(graph)
        if not are_isomorphic(graph, copy):
            continue
        feeds = random_feeds(graph, positive=True)
        renamed_feeds = {
            value.name: feeds[original.name]
            for value, original in zip(copy.inputs, graph.inputs, strict=True)
        }
        checked += 1
        if outputs_agree(run(graph, feeds), run(copy, renamed_feeds)):
            agreed += 1
    return {"pairs": checked, "agreed": agreed, "all_agreed": agreed == checked}


def same_answer_does_not_mean_same_hash() -> dict:
    """The direction the hash cannot promise, with an example.

    A graph and its optimised form compute the same thing and hash differently, which is not a
    defect. It is what an optimiser is for. Anything using a structural hash as a cache key has
    to key on the graph at a fixed point in the pipeline, because the same source hashes
    differently before and after any pass that fires.
    """
    builder = Builder()
    x = builder.input([8, 8], name="x")
    scale = builder.mul(builder.constant(2.0), builder.constant(0.5))
    graph = builder.finish(builder.mul(x, scale))

    optimised = simplify(fold_constants(graph))
    feeds = random_feeds(graph, positive=True)
    return {
        "same_hash": are_isomorphic(graph, optimised),
        "same_answer": outputs_agree(run(graph, feeds), run(optimised, feeds)),
    }


def collision_rate(count: int = 200, *, seed: int = 0) -> dict:
    """How often two different generated graphs share a hash.

    The measurement that justifies truncating the digest. Eleven percent of two hundred
    generated graphs share a hash with another, which sounds alarming and is not: none of them
    is a collision. They are graphs that differ only in work nobody reads, and the function
    below separates the two.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    hashes: dict[str, int] = {}
    graphs = list(generate_many(count, start=seed))
    for graph in graphs:
        key = structural_hash(graph)
        hashes[key] = hashes.get(key, 0) + 1
    repeats = sum(times - 1 for times in hashes.values() if times > 1)
    return {
        "graphs": len(graphs),
        "distinct_hashes": len(hashes),
        "repeats": repeats,
        "share": round(repeats / len(graphs), 4) if graphs else 0.0,
    }


def repeats_are_real_duplicates(count: int = 200, *, seed: int = 0) -> dict:
    """Whether the graphs that share a hash are the same graph.

    They are, and the way they are is the interesting part. None of the pairs has the same node
    list, because the hash is taken over the outputs and everything a graph computes and does
    not return is invisible to it. Every one of them has the same node list once dead code has
    been removed.

    That is the property that makes this usable as a cache key. Two graphs that differ only in
    work nobody reads compile to the same thing, and a key that told them apart would miss a
    hit it should have had.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    graphs = list(generate_many(count, start=seed))
    by_hash: dict[str, list[Graph]] = {}
    for graph in graphs:
        by_hash.setdefault(structural_hash(graph), []).append(graph)

    identical = 0
    same_once_pruned = 0
    collisions = 0
    for group in by_hash.values():
        for other in group[1:]:
            if _same_node_lists(group[0], other):
                identical += 1
            elif _same_node_lists(eliminate_dead_code(group[0]), eliminate_dead_code(other)):
                same_once_pruned += 1
            else:
                collisions += 1
    return {
        "identical": identical,
        "same_once_dead_code_is_removed": same_once_pruned,
        "collisions": collisions,
    }


def _same_node_lists(left: Graph, right: Graph) -> bool:
    """Whether two graphs hold the same operations in the same order."""
    if len(left.nodes) != len(right.nodes):
        return False
    for first, second in zip(left.nodes, right.nodes, strict=True):
        if first.op is not second.op or first.output.shape != second.output.shape:
            return False
    return True


def dead_code_is_invisible_to_the_hash() -> dict:
    """A graph and the same graph with unread work bolted on.

    The direct statement of what the collision measurement found by accident. The hash is over
    the outputs, so a value nobody returns is not part of the identity, and it should not be:
    two graphs that differ only in work nobody reads compile to the same artefact.
    """
    builder = Builder()
    x = builder.input([8, 8], name="x")
    kept = builder.relu(builder.exp(x))
    lean = builder.finish(kept)

    other = Builder()
    y = other.input([8, 8], name="y")
    result = other.relu(other.exp(y))
    other.tanh(y)
    other.mul(y, y)
    fat = other.finish(result)
    return {
        "same_hash": are_isomorphic(lean, fat),
        "different_node_counts": len(lean.nodes) != len(fat.nodes),
    }


def hash_is_stable_across_calls(graph: Graph | None = None, times: int = 8) -> bool:
    """Whether hashing the same graph twice gives the same answer.

    Worth a line because it would not, if anything in the identity depended on iteration order
    over a set. It does not, and a test is cheaper than remembering that.
    """
    if times < 2:
        raise ConfigError(f"there is nothing to compare, got {times}")
    target = graph if graph is not None else layernorm_graph()
    return len({structural_hash(target) for _ in range(times)}) == 1
