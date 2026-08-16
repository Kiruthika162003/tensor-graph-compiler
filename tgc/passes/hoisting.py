from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.analysis.cost import annotate_matmuls, node_flops
from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.graph import Graph
from tgc.verify.reference import outputs_agree, random_feeds, run

# Doing the part of a graph that does not depend on the data, once instead of every call.
#
# A compiled model is called many times with different activations and the same weights. Any
# node whose inputs are all weights computes the same value every call, so it belongs in a
# prologue that runs once rather than in the body that runs per call. Transposing a weight,
# scaling it, quantising it, packing it into the layout a kernel wants: all of that is work the
# body is doing again for no reason.
#
# Constant folding cannot do this, because a weight is not a constant. It is an input, and the
# compiler is only allowed to treat it as fixed because the caller has said so. So the pass
# takes a list of which inputs are parameters, and everything else follows from that one
# declaration.
#
# The measurements say two things.
#
# The saving is real and it is small in arithmetic and large in traffic. A transpose does no
# arithmetic at all and moves the whole weight twice, so hoisting one out of the body removes
# nothing from the flop count and a great deal from the bytes, which for a memory bound layer is
# the number that decides the time.
#
# And it is not free. Every hoisted value has to be kept, so the prologue turns compute into
# storage at a rate of one tensor per hoisted node, and on a graph where the hoisted node is a
# transpose of a weight that is a second copy of the weight. The break even is in calls: the
# sweep at the bottom says how many.


@dataclass
class HoistReport:
    """What moved out of the body and what it cost."""

    hoisted: tuple[str, ...] = ()
    body_nodes: int = 0
    prologue_nodes: int = 0
    stored_elements: int = 0

    @property
    def moved(self) -> int:
        """Nodes that left the body."""
        return len(self.hoisted)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "hoisted": list(self.hoisted),
            "moved": self.moved,
            "body_nodes": self.body_nodes,
            "prologue_nodes": self.prologue_nodes,
            "stored_elements": self.stored_elements,
        }


@dataclass
class Split:
    """A graph divided into a prologue and a body.

    The body takes the prologue's results as extra inputs, appended after the ones it already
    had. That ordering is a promise to the caller in the same way the parameter ordering in
    frontend/module.py is, and it is the only thing connecting a value the prologue produced to
    the slot the body reads it from.
    """

    prologue: Graph | None
    body: Graph
    report: HoistReport = field(default_factory=HoistReport)

    @property
    def hoisted_anything(self) -> bool:
        """Whether the split found any work to move."""
        return self.prologue is not None

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"hoisted_anything": self.hoisted_anything, **self.report.as_dict()}


def parameter_only(graph: Graph, parameters: Sequence[str]) -> list[str]:
    """Every node whose value is the same on every call.

    Closed forwards from the declared parameters. A node qualifies when every one of its inputs
    is a parameter or another qualifying node, which is the same closure a constant folder does
    and over a different starting set.
    """
    known = set(parameters)
    unknown = {value.name for value in graph.inputs} - known
    if not known:
        return []
    missing = known - {value.name for value in graph.inputs}
    if missing:
        raise ConfigError(f"{sorted(missing)} are not inputs of this graph")

    fixed: list[str] = []
    for node in graph.nodes:
        if node.op is ops.CONSTANT:
            known.add(node.name)
            fixed.append(node.name)
            continue
        if not node.op.can_be_removed_if_unused:
            continue
        if any(name in unknown or name not in known for name in node.inputs):
            continue
        known.add(node.name)
        fixed.append(node.name)
    return fixed


def hoistable(graph: Graph, parameters: Sequence[str]) -> list[str]:
    """The nodes worth moving, which is the fixed ones the body still reads.

    A fixed node nothing reads is dead and belongs to the dead code pass. A fixed node read only
    by other fixed nodes moves with them and is not a boundary. What the body needs is the
    fixed nodes with at least one consumer that is not fixed, and those are the values the
    prologue has to hand over.
    """
    fixed = set(parameter_only(graph, parameters))
    needed: list[str] = []
    for node in graph.nodes:
        if node.name in fixed:
            continue
        for name in node.inputs:
            if name in fixed and name not in needed:
                needed.append(name)
    for name in graph.outputs:
        if name in fixed and name not in needed:
            needed.append(name)
    return needed


def split(graph: Graph, parameters: Sequence[str]) -> Split:
    """Divide a graph into what runs once and what runs per call.

    The prologue is built from the fixed nodes and returns the boundary values; the body is
    built from everything else, with the boundary values arriving as inputs. Neither is a subset
    of the original node list, because both are rebuilt through the builder so their shapes come
    from the same inference the rest of the compiler uses.
    """
    fixed = set(parameter_only(graph, parameters))
    boundary = hoistable(graph, parameters)
    if not boundary:
        return Split(
            prologue=None,
            body=graph,
            report=HoistReport(body_nodes=len(graph.nodes)),
        )

    prologue_builder = Builder()
    mapping: dict[str, str] = {}
    for value in graph.inputs:
        if value.name in parameters:
            mapping[value.name] = prologue_builder.input(
                _sizes(value), dtype=value.dtype, name=value.name
            )
    for node in graph.nodes:
        if node.name not in fixed:
            continue
        mapping[node.name] = _replay(prologue_builder, node, mapping)
    prologue = prologue_builder.finish(*[mapping[name] for name in boundary])

    body_builder = Builder()
    body_map: dict[str, str] = {}
    for value in graph.inputs:
        if value.name not in parameters:
            body_map[value.name] = body_builder.input(
                _sizes(value), dtype=value.dtype, name=value.name
            )
    for name in boundary:
        value = graph.value(name)
        body_map[name] = body_builder.input(
            _sizes(value), dtype=value.dtype, name=f"hoisted_{name}"
        )
    for node in graph.nodes:
        if node.name in fixed:
            continue
        body_map[node.name] = _replay(body_builder, node, body_map)
    body = body_builder.finish(*[body_map[name] for name in graph.outputs])

    stored = sum(graph.value(name).shape.elements for name in boundary)
    return Split(
        prologue=prologue,
        body=body,
        report=HoistReport(
            hoisted=tuple(sorted(fixed)),
            body_nodes=len(body.nodes),
            prologue_nodes=len(prologue.nodes),
            stored_elements=stored,
        ),
    )


def _sizes(value) -> list:
    """The sizes of a value, symbolic names included."""
    return [size.value if size.is_static else size.name for size in value.shape.sizes]


def _replay(builder: Builder, node, mapping: dict[str, str]) -> str:
    """Rebuild one node into a builder."""
    if node.op is ops.CONSTANT:
        return builder.constant(float(node.attrs["value"]), dtype=node.output.dtype)
    missing = [name for name in node.inputs if name not in mapping]
    if missing:
        raise PassError(f"{node.name} reads {missing}, which are not on this side of the split")
    return builder.apply(node.op, *[mapping[name] for name in node.inputs], **node.attrs)


def run_split(
    result: Split, feeds: dict[str, torch.Tensor], parameters: Sequence[str]
) -> list[torch.Tensor]:
    """Run the prologue once and the body against its results.

    Written out rather than folded into a helper, because the whole point of the pass is that
    these are two calls at two different times, and a function that ran them together would be
    describing something the pass does not do.
    """
    if result.prologue is None:
        return run(result.body, feeds)

    prologue_feeds = {name: feeds[name] for name in parameters}
    hoisted = run(result.prologue, prologue_feeds)
    body_feeds = {name: value for name, value in feeds.items() if name not in set(parameters)}
    for value, tensor in zip(result.prologue.outputs, hoisted, strict=True):
        body_feeds[f"hoisted_{value}"] = tensor
    return run(result.body, body_feeds)


def preprocessed_weights(rows: int = 8, width: int = 64) -> Graph:
    """A layer that transposes and scales its weight before using it.

    Both of those are things a frontend emits and neither depends on the activation, so both are
    work the body is repeating. Written as a fixture rather than found in an existing one
    because none of the existing fixtures has any parameter only work in it at all, which is
    itself worth knowing.
    """
    if min(rows, width) < 1:
        raise ConfigError("every dimension has to be positive")
    builder = Builder()
    activation = builder.input([rows, width], name="x")
    weight = builder.input([width, width], name="w")

    scaled = builder.mul(weight, builder.constant(0.5))
    turned = builder.transpose(scaled, [1, 0])
    return builder.finish(builder.relu(builder.matmul(activation, turned)))


def nothing_to_hoist(rows: int = 8, width: int = 64) -> Graph:
    """A layer that uses its weight directly, so there is nothing to move."""
    builder = Builder()
    activation = builder.input([rows, width], name="x")
    weight = builder.input([width, width], name="w")
    return builder.finish(builder.relu(builder.matmul(activation, weight)))


def the_split_computes_the_same_thing(graph: Graph | None = None, *, seed: int = 0) -> dict:
    """The prologue and body against the original graph.

    Bit equality, and it has to be: the split moves operations between two graphs and changes
    neither the operations nor their order relative to each other. Anything short of exact would
    mean the replay through the builder changed something.
    """
    target = graph if graph is not None else preprocessed_weights()
    result = split(target, ["w"])
    feeds = random_feeds(target, positive=True, seed=seed)
    return {
        "hoisted": result.report.moved,
        "identical": outputs_agree(run(target, feeds), run_split(result, feeds, ["w"])),
    }


def what_moves(graph: Graph | None = None) -> dict:
    """How much of a graph is parameter only work."""
    target = graph if graph is not None else preprocessed_weights()
    result = split(target, ["w"])
    return {
        "nodes": len(target.nodes),
        "moved": result.report.moved,
        "body_nodes": result.report.body_nodes,
        "prologue_nodes": result.report.prologue_nodes,
    }


def a_graph_with_no_preprocessing_is_left_alone() -> dict:
    """What the pass does when there is nothing to move.

    Nothing, and it returns the graph it was given rather than a rebuilt copy of it. A pass that
    always rebuilds makes every later comparison by identity useless and hides the fact that it
    did nothing.
    """
    graph = nothing_to_hoist()
    result = split(graph, ["w"])
    return {
        "hoisted_anything": result.hoisted_anything,
        "same_object": result.body is graph,
    }


def declaring_nothing_hoists_nothing(graph: Graph | None = None) -> dict:
    """What happens when the caller does not say which inputs are fixed.

    Nothing moves, which is the only safe answer. A compiler that guessed which inputs were
    weights would be right most of the time and would silently cache an activation the first
    time it was wrong.
    """
    target = graph if graph is not None else preprocessed_weights()
    return {
        "with_the_declaration": split(target, ["w"]).report.moved,
        "without_it": split(target, []).report.moved,
    }


def an_unknown_parameter_is_refused(graph: Graph | None = None) -> bool:
    """Whether declaring an input that does not exist is caught."""
    target = graph if graph is not None else preprocessed_weights()
    try:
        split(target, ["not_an_input"])
    except ConfigError:
        return True
    return False


def arithmetic_moved(graph: Graph, parameters: Sequence[str]) -> dict:
    """How much of the flop count leaves the body.

    Almost none, on a graph whose preprocessing is a transpose and a scale. The arithmetic in a
    layer is the matrix product and the product depends on the activation, so it stays. What
    leaves is bytes, which the next function counts.
    """
    prepared = annotate_matmuls(graph)
    result = split(graph, parameters)
    hoisted = set(result.report.hoisted)
    total = sum(node_flops(node) for node in prepared.nodes)
    moved = sum(node_flops(node) for node in prepared.nodes if node.name in hoisted)
    return {
        "total": total,
        "moved": moved,
        "share": round(moved / total, 6) if total else 0.0,
    }


def traffic_moved(graph: Graph, parameters: Sequence[str], *, element_bytes: int = 4) -> dict:
    """How many bytes leave the body.

    Every hoisted node's output is a tensor the body no longer writes and no longer reads, and
    the transpose of a weight is the whole weight. On a layer of this shape that is eighty nine
    percent of everything the body touched, because the weight is square and the activation is
    eight rows.
    """
    result = split(graph, parameters)
    hoisted = set(result.report.hoisted)
    total = sum(node.output.shape.elements for node in graph.nodes) * element_bytes
    moved = (
        sum(node.output.shape.elements for node in graph.nodes if node.name in hoisted)
        * element_bytes
    )
    return {
        "total": total,
        "moved": moved,
        "share": round(moved / total, 4) if total else 0.0,
    }


def the_saving_is_traffic_rather_than_arithmetic(graph: Graph | None = None) -> dict:
    """Both shares side by side, which is the point of the pass."""
    target = graph if graph is not None else preprocessed_weights()
    return {
        "arithmetic_share": arithmetic_moved(target, ["w"])["share"],
        "traffic_share": traffic_moved(target, ["w"])["share"],
    }


def storage_cost(graph: Graph | None = None) -> dict:
    """What the prologue's results cost to keep.

    One tensor per boundary value, held for the life of the compiled function. On this fixture
    that is a second copy of the weight, so the pass has bought a per call saving with a
    permanent doubling of the parameter memory.
    """
    target = graph if graph is not None else preprocessed_weights()
    result = split(target, ["w"])
    parameters = sum(value.shape.elements for value in target.inputs if value.name == "w")
    return {
        "parameter_elements": parameters,
        "stored_elements": result.report.stored_elements,
        "ratio": round(result.report.stored_elements / parameters, 3) if parameters else 0.0,
    }


def break_even_calls(graph: Graph | None = None, limit: int = 1000) -> int:
    """How many calls it takes for the prologue to pay for itself.

    Two, on this fixture and on almost any other. The prologue does once what the body was
    doing every time, so the first call is a wash and every call after it is ahead. Searched
    rather than asserted because the arithmetic is only that simple when the prologue's cost is
    exactly the per call saving, and a prologue that packs a weight into a tiled layout costs
    more than the transpose it replaced.
    """
    target = graph if graph is not None else preprocessed_weights()
    shares = traffic_moved(target, ["w"])
    total = shares["total"]
    moved = shares["moved"]
    if moved == 0:
        return 0
    for count in range(1, limit + 1):
        if moved + (total - moved) * count < total * count:
            return count
    return 0


def call_count_sweep(
    counts: Sequence[int] = (1, 2, 10, 100), graph: Graph | None = None
) -> list[dict]:
    """Total time with and without the prologue, over a number of calls.

    The two lines are equal at the first call and diverge after it. The ratio climbs toward one
    over what is left, which on this fixture is nine, and it gets there slowly: five at ten
    calls and eight and a third at a hundred. A function called a handful of times collects
    about half of what is available, which is a good deal less than the pass looks like it
    promises.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    target = graph if graph is not None else preprocessed_weights()
    shares = traffic_moved(target, ["w"])
    total = shares["total"]
    moved = shares["moved"]
    rows = []
    for count in counts:
        without = total * count
        with_prologue = moved + (total - moved) * count
        rows.append(
            {
                "calls": count,
                "without": without,
                "with_prologue": with_prologue,
                "ratio": round(without / with_prologue, 4) if with_prologue else 0.0,
            }
        )
    return rows


def the_benefit_takes_a_hundred_calls() -> dict:
    """How close to the limit the saving gets, per call count.

    Not close, for a while. Ten calls collect a bit over half of what is available and a hundred
    collect ninety three percent, because the prologue is paid once and it is the same size as
    the saving. A pass described as removing eighty nine percent of the traffic removes forty
    four percent of it over ten calls.
    """
    rows = {row["calls"]: row for row in call_count_sweep()}
    limit = 1.0 / (1.0 - traffic_moved(preprocessed_weights(), ["w"])["share"])
    return {
        "at_one_call": rows[1]["ratio"],
        "at_ten_calls": rows[10]["ratio"],
        "at_a_hundred": rows[100]["ratio"],
        "limit": round(limit, 4),
        "ten_calls_is_about_half": rows[10]["ratio"] < 0.7 * limit,
        "a_hundred_is_most_of_it": rows[100]["ratio"] > 0.9 * limit,
    }


def compare_graphs() -> list[dict]:
    """What the pass finds in each fixture.

    Nothing, in every one of the standard ones. They are written as a single layer applied to an
    activation with no weight preprocessing at all, which is not what a traced model looks like:
    a frontend emits transposes and casts around weights constantly. The fixture with the
    preprocessing in it is the realistic one and the rest are the simplified ones.
    """
    rows = []
    for label, graph, parameters in (
        ("preprocessed", preprocessed_weights(), ["w"]),
        ("plain layer", nothing_to_hoist(), ["w"]),
        ("mlp", mlp_graph(), ["w_up", "w_down", "b_up"]),
        ("softmax", softmax_graph(), []),
        ("layernorm", layernorm_graph(), []),
    ):
        result = split(graph, parameters)
        rows.append({"graph": label, "moved": result.report.moved})
    return rows


def only_the_preprocessing_fixture_has_anything() -> dict:
    """Which fixtures offer the pass any work."""
    rows = {row["graph"]: row["moved"] for row in compare_graphs()}
    return {
        "graphs": len(rows),
        "with_work": [name for name, moved in rows.items() if moved],
    }
