from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.analysis.cost import GPU, Machine
from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, mlp_graph, softmax_graph
from tgc.ir.graph import Graph, Node
from tgc.verify.reference import outputs_agree, random_feeds, run

# Turning several small matrix products into one large one.
#
# A model that applies the same weight to several activations, or several weights to the same
# activation, issues one kernel per product. Each kernel reads the weight from memory, does a
# small amount of arithmetic and stops, and a small matrix product is memory bound: the time is
# the weight divided by the bandwidth and the arithmetic is free. Doing them together reads the
# weight once and the arithmetic is still free, so the time falls by the number of products.
#
# That is the whole argument and the measurement below is where it stops being true, which turns
# out not to be where the argument suggests. Growing the contracted dimension does not stop it
# paying: at eight rows the gain climbs from a third at a contraction of eight to three and a
# half at five hundred, because a product that thin is memory bound at every size.
#
# What stops it is the number of rows. Past sixty four rows the products are compute bound, the
# arithmetic is unchanged by joining them, and the gain is exactly one. So batching is a fix for
# products with few rows rather than for products that are small, and those are different
# conditions: a thin product against an enormous weight is exactly the case it helps most.
#
# The rewrite needs a join and a window, which is why ir/op.py has a concat and a slice. It is
# the second time in this compiler that a transformation has needed operations the forward graph
# never contained, the first being the indicator that reverse mode needed, and the pattern is
# worth naming: an operation set built from what a user writes is not closed under the things a
# compiler does to it.


@dataclass
class Candidate:
    """A group of matrix products that could be issued as one."""

    nodes: tuple[str, ...]
    shared: str
    axis: int

    @property
    def size(self) -> int:
        """Products in the group."""
        return len(self.nodes)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "nodes": list(self.nodes),
            "shared": self.shared,
            "joined_axis": self.axis,
            "size": self.size,
        }


@dataclass
class BatchReport:
    """What batching found and what it would change."""

    candidates: list[Candidate] = field(default_factory=list)

    @property
    def groups(self) -> int:
        """Groups found."""
        return len(self.candidates)

    @property
    def products_merged(self) -> int:
        """Kernels removed if every group were merged."""
        return sum(candidate.size - 1 for candidate in self.candidates)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "groups": self.groups,
            "products_merged": self.products_merged,
            "sizes": [candidate.size for candidate in self.candidates],
        }


def _static(shape) -> list[int]:
    """The sizes of a shape as numbers, refusing symbolic ones."""
    sizes = []
    for size in shape.sizes:
        if not size.is_static:
            raise PassError(f"cannot batch across the symbolic dimension {size.name}")
        sizes.append(size.value)
    return sizes


def independent(graph: Graph, names: Sequence[str]) -> bool:
    """Whether none of a set of values depends on another of them.

    The condition that makes a group safe to issue together. Two products where one reads the
    other cannot run at the same time whatever the shapes say, and checking it is a reachability
    query rather than a shape check, which is why it is separate from the matching below.
    """
    chosen = set(names)
    reached: dict[str, set[str]] = {}
    for node in graph.nodes:
        behind: set[str] = set()
        for operand in node.inputs:
            behind |= reached.get(operand, set())
            if operand in chosen:
                behind.add(operand)
        reached[node.name] = behind
    return all(not (reached.get(name, set()) & (chosen - {name})) for name in names)


def find_candidates(graph: Graph) -> BatchReport:
    """Groups of products that share their right operand and could be joined by rows.

    Only that one pattern. Sharing the left operand and joining by columns is the mirror image
    and is left out deliberately: it is the same rewrite with the axes swapped and adding it
    would double the code to say nothing new. The shared operand has to be the same value rather
    than an equal one, because a subexpression pass has already merged the equal ones.
    """
    by_weight: dict[str, list[Node]] = {}
    for node in graph.nodes:
        if node.op is not ops.MATMUL:
            continue
        by_weight.setdefault(node.inputs[1], []).append(node)

    report = BatchReport()
    for weight, nodes in by_weight.items():
        if len(nodes) < 2:
            continue
        shapes = {tuple(_static(graph.value(node.inputs[0]).shape)[1:]) for node in nodes}
        if len(shapes) != 1:
            continue
        names = tuple(node.name for node in nodes)
        if not independent(graph, [node.inputs[0] for node in nodes]):
            continue
        report.candidates.append(Candidate(nodes=names, shared=weight, axis=0))
    return report


def _positions(graph: Graph) -> dict[str, int]:
    """Where each value sits in the node list, with inputs before everything."""
    positions = {value.name: -1 for value in graph.inputs}
    for index, node in enumerate(graph.nodes):
        positions[node.name] = index
    return positions


def when_a_group_is_ready(graph: Graph, candidate: Candidate) -> int:
    """The earliest position a joined product could be placed at.

    One past the last of the activations it reads. A group whose activations are all graph
    inputs is ready immediately; one whose last activation is computed halfway down the graph
    cannot be issued before that point.
    """
    positions = _positions(graph)
    return max(positions[graph.node(name).inputs[0]] for name in candidate.nodes) + 1


def when_a_group_is_needed(graph: Graph, candidate: Candidate) -> int:
    """The earliest position that reads one of the group's results."""
    members = set(candidate.nodes)
    for index, node in enumerate(graph.nodes):
        if node.name in members:
            continue
        if members & set(node.inputs):
            return index
    return len(graph.nodes)


def is_placeable(graph: Graph, candidate: Candidate) -> bool:
    """Whether a joined product can sit somewhere legal.

    It cannot when one of the activations is computed after something that already reads
    another product in the group. Joining them would mean issuing the product before one of its
    operands exists, and no amount of reordering fixes that without moving the consumer, which
    is a different pass.
    """
    return when_a_group_is_ready(graph, candidate) <= when_a_group_is_needed(graph, candidate)


def batch_matmuls(graph: Graph) -> Graph:
    """Rewrite every group of shareable products into one product with slices after it.

    The activations are joined along their rows, multiplied once, and the results are cut back
    out. Every consumer then reads a slice instead of a product, which is a rename rather than a
    change of meaning, and the check below is that the numbers are identical rather than close.

    The joined product goes one position after the last activation it reads rather than where
    the first product was. Putting it where the first product was reads an activation that does
    not exist yet, which is how the first version of this failed.
    """
    report = find_candidates(graph)
    placeable = [item for item in report.candidates if is_placeable(graph, item)]
    if not placeable:
        return graph

    merged = {name for candidate in placeable for name in candidate.nodes}
    at_index: dict[int, list[Candidate]] = {}
    for candidate in placeable:
        at_index.setdefault(when_a_group_is_ready(graph, candidate), []).append(candidate)

    builder = Builder()
    mapping: dict[str, str] = {}
    for value in graph.inputs:
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=value.name,
        )

    for candidate in at_index.get(0, []):
        _emit_group(graph, builder, mapping, candidate)

    for index, node in enumerate(graph.nodes):
        for candidate in at_index.get(index + 1, []):
            _emit_group(graph, builder, mapping, candidate)
        if node.name in merged:
            continue
        if node.op is ops.CONSTANT:
            mapping[node.name] = builder.constant(
                float(node.attrs["value"]), dtype=node.output.dtype
            )
            continue
        operands = [mapping[name] for name in node.inputs]
        mapping[node.name] = builder.apply(node.op, *operands, **node.attrs)

    return builder.finish(*[mapping[name] for name in graph.outputs])


def _emit_group(
    graph: Graph, builder: Builder, mapping: dict[str, str], candidate: Candidate
) -> None:
    """Emit one joined product and the windows that recover its parts."""
    nodes = [graph.node(name) for name in candidate.nodes]
    joined = mapping[nodes[0].inputs[0]]
    for node in nodes[1:]:
        joined = builder.concat(joined, mapping[node.inputs[0]], candidate.axis)

    product = builder.matmul(joined, mapping[candidate.shared])
    start = 0
    for node in nodes:
        rows = _static(graph.value(node.inputs[0]).shape)[0]
        mapping[node.name] = builder.slice(product, candidate.axis, start, rows)
        start += rows


def parallel_heads(heads: int = 4, rows: int = 8, inner: int = 32, columns: int = 32) -> Graph:
    """Several activations put through one weight, which is the pattern batching is for.

    Written to look like what produces it in practice: a set of independent branches that a
    frontend traced separately because the source wrote them separately, all reaching the same
    parameter. Nothing in the graph says they belong together and the shapes say they could.
    """
    if heads < 2:
        raise ConfigError(f"there is nothing to batch with {heads} branches")
    builder = Builder()
    weight = builder.input([inner, columns], name="w")
    results = []
    for index in range(heads):
        activation = builder.input([rows, inner], name=f"x{index}")
        results.append(builder.relu(builder.matmul(activation, weight)))

    total = results[0]
    for other in results[1:]:
        total = builder.add(total, other)
    return builder.finish(total)


def dependent_chain(length: int = 3, size: int = 16) -> Graph:
    """Products that read each other, which cannot be batched however alike they look.

    Every one of them uses the same weight and the same shapes, so a matcher that only compared
    shapes would merge them and produce a graph that computes something else entirely.
    """
    if length < 2:
        raise ConfigError(f"a chain needs at least two products, got {length}")
    builder = Builder()
    weight = builder.input([size, size], name="w")
    current = builder.input([size, size], name="x")
    for _ in range(length):
        current = builder.matmul(current, weight)
    return builder.finish(current)


def mismatched_shapes(size: int = 16) -> Graph:
    """Two products whose activations have different inner dimensions.

    They cannot share a weight at all, so this is really a check that the matcher groups by the
    operand rather than by the operation, and that a graph with two different weights in it does
    not come back as one group.
    """
    builder = Builder()
    narrow = builder.input([size, size], name="wn")
    wide = builder.input([size * 2, size], name="ww")
    first = builder.matmul(builder.input([size, size], name="a"), narrow)
    second = builder.matmul(builder.input([size, size * 2], name="b"), wide)
    return builder.finish(builder.add(first, second))


def rewrite_preserves_the_answer(graph: Graph | None = None, *, seed: int = 0) -> dict:
    """The batched graph against the original, on the same inputs.

    Bit equality is not expected and is measured anyway. Joining four products of eight rows
    into one of thirty two changes the shape the library sees, and the library picks its
    blocking from the shape, so the answer moves in the last place for the same reason a sharded
    product does.
    """
    target = graph if graph is not None else parallel_heads()
    rewritten = batch_matmuls(target)
    feeds = random_feeds(target, positive=True, seed=seed)
    before = run(target, feeds)
    after = run(rewritten, feeds)
    gap = float((before[0] - after[0]).abs().max())
    scale = float(before[0].abs().max())
    return {
        "identical": outputs_agree(before, after),
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
    }


def kernel_counts(graph: Graph | None = None) -> dict:
    """How many matrix products the graph has before and after."""
    target = graph if graph is not None else parallel_heads()
    return {
        "before": sum(1 for node in target.nodes if node.op is ops.MATMUL),
        "after": sum(1 for node in batch_matmuls(target).nodes if node.op is ops.MATMUL),
        "nodes_before": len(target.nodes),
        "nodes_after": len(batch_matmuls(target).nodes),
    }


def the_rewrite_adds_nodes_while_removing_kernels() -> dict:
    """The trade the rewrite actually makes.

    Fewer products and more nodes, because every join and every window is a node. That is only
    worth it if a product costs more than a copy, which for a memory bound product it does and
    for a copy of the same data it does not, so the joins are the part of this that has to be
    elided by a later pass rather than executed.
    """
    result = kernel_counts()
    return {
        "kernels_removed": result["before"] - result["after"],
        "nodes_added": result["nodes_after"] - result["nodes_before"],
    }


def find_nothing_to_batch() -> list[dict]:
    """Which fixtures offer the rewrite anything, and why the others do not."""
    rows = []
    for label, graph in (
        ("parallel heads", parallel_heads()),
        ("dependent chain", dependent_chain()),
        ("mismatched shapes", mismatched_shapes()),
        ("mlp", mlp_graph()),
        ("softmax", softmax_graph()),
    ):
        report = find_candidates(graph)
        rows.append({"graph": label, **report.as_dict()})
    return rows


def a_dependent_chain_is_refused() -> dict:
    """Whether the matcher notices that products reading each other cannot be merged.

    The check that separates a shape matcher from a correct one. Every product in the chain
    shares a weight and has identical shapes, and merging them would change the answer, so the
    only thing standing between the two is the reachability query.
    """
    graph = dependent_chain()
    return {
        "products": sum(1 for node in graph.nodes if node.op is ops.MATMUL),
        "groups_found": find_candidates(graph).groups,
        "independent": independent(graph, [node.name for node in graph.nodes[:2]]),
    }


def traffic_for(rows: int, inner: int, columns: int, count: int, *, batched: bool) -> int:
    """Bytes a set of products reads, one at a time or joined.

    The weight is the term that changes. Separately, each product reads the whole weight;
    joined, the weight is read once, and everything else is read the same either way.
    """
    if min(rows, inner, columns, count) < 1:
        raise ConfigError("every dimension has to be positive")
    element = 4
    activations = rows * inner * count * element
    outputs = rows * columns * count * element
    weight = inner * columns * element * (1 if batched else count)
    return activations + outputs + weight


def arithmetic_for(rows: int, inner: int, columns: int, count: int) -> float:
    """Multiply adds a set of products performs, which batching does not change."""
    if min(rows, inner, columns, count) < 1:
        raise ConfigError("every dimension has to be positive")
    return 2.0 * rows * inner * columns * count


def batching_gain(
    rows: int = 8, inner: int = 32, columns: int = 32, count: int = 4, machine: Machine = GPU
) -> dict:
    """What the rewrite is worth on one shape, under a roofline model.

    Both the separate and the joined version are timed as the larger of their arithmetic over
    the compute rate and their traffic over the bandwidth. The arithmetic is identical, so the
    gain is entirely in the traffic and it only shows up while the traffic is the larger term.
    """
    work = arithmetic_for(rows, inner, columns, count)
    separate = traffic_for(rows, inner, columns, count, batched=False)
    joined = traffic_for(rows, inner, columns, count, batched=True)

    compute_time = work / machine.flops_per_second
    separate_time = max(compute_time, separate / machine.bytes_per_second)
    joined_time = max(compute_time, joined / machine.bytes_per_second)
    return {
        "arithmetic": work,
        "traffic_separate": separate,
        "traffic_joined": joined,
        "speedup": round(separate_time / joined_time, 4) if joined_time else 1.0,
        "compute_bound": compute_time >= separate / machine.bytes_per_second,
    }


def size_sweep(sizes: Sequence[int] = (8, 32, 128, 512, 2048), count: int = 4) -> list[dict]:
    """The gain against how large the individual products are.

    Climbs rather than falling, which is not what the usual argument for batching predicts. At
    eight rows the product is memory bound at every contraction swept, so a larger weight means
    more traffic to remove and joining them is worth more rather than less.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    return [
        {"size": size, **batching_gain(rows=8, inner=size, columns=size, count=count)}
        for size in sizes
    ]


def row_sweep(rows: Sequence[int] = (1, 8, 64, 256, 1024), size: int = 256) -> list[dict]:
    """The gain against how many rows each product has, which is what really decides it.

    Falls off a cliff. At eight rows every product is memory bound and joining four of them is
    worth three and a half; at sixty four they are compute bound, the arithmetic is unchanged by
    joining them and the gain is exactly one.
    """
    if not rows:
        raise ConfigError("there is nothing to sweep")
    return [
        {"rows": count, **batching_gain(rows=count, inner=size, columns=size)} for count in rows
    ]


def where_batching_stops_paying(threshold: float = 1.1, size: int = 256) -> int:
    """The row count past which joining products is worth less than a tenth.

    Searched over the sweep rather than derived, because the answer depends on the ratio of
    compute to bandwidth on the machine and that is a property of the hardware rather than of
    the rewrite.
    """
    for row in row_sweep(size=size):
        if row["speedup"] < threshold:
            return int(row["rows"])
    return 0


def count_sweep(counts: Sequence[int] = (2, 4, 8, 16, 32), inner: int = 32) -> list[dict]:
    """The gain against how many products are joined.

    Climbs toward the ratio of the whole traffic to the traffic without the repeated weight,
    which is a limit rather than a line: past the point where the weight is a small share of
    what is read, joining more of them adds nothing.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    return [
        {"count": count, **batching_gain(inner=inner, columns=inner, count=count)}
        for count in counts
    ]


def more_products_help_less_and_less() -> dict:
    """Whether that climb really flattens."""
    rows = count_sweep()
    first = rows[1]["speedup"] - rows[0]["speedup"]
    last = rows[-1]["speedup"] - rows[-2]["speedup"]
    return {
        "gain_from_two_to_four": round(first, 4),
        "gain_from_sixteen_to_thirty_two": round(last, 4),
        "flattening": last < first,
    }
