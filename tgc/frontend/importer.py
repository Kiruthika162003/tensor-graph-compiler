from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, GraphError
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph
from tgc.ir.op import get_op
from tgc.verify.reference import run

# Reading a model written against somebody else's operation set.
#
# A compiler that only accepts graphs its own frontend produced is a compiler nobody can use.
# The graphs that arrive come from an exchange format with a few hundred operations in it, and
# this one has thirty, so importing is mostly a question of what to do about the difference.
#
# Three answers and they are not equally safe. Some operations map one to one and are a rename.
# Some are compositions of operations this compiler does have, and lowering them is real work
# that can be got wrong quietly. The rest have no lowering at all and the only honest thing is
# to refuse the model and say which operation stopped it.
#
# The measurements say the coverage is high and the risk is entirely in the middle group. Every
# node in the sample that carries an attribute is a composite, and the softmax is the clearest
# case: the foreign one takes an axis and this compiler has no softmax at all, so the axis has
# to become a choice of which dimension the two reductions run over. A lowering that ignored it
# would produce a graph with the same shape, the same node count and an output differing by
# almost the whole range, which the comparison at the bottom of this file shows.
#
# The other number worth having is what lowering costs in size. Three foreign nodes become
# eleven, because a softmax is seven operations here once the broadcasts are counted and a
# general product with a bias is three.
# A model that looks small in the exchange format is not small once it has arrived, and every
# pass downstream sees the larger number.

DIRECT = {
    "Relu": "relu",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "Exp": "exp",
    "Log": "log",
    "Sqrt": "sqrt",
    "Abs": "abs",
    "Neg": "neg",
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "Div": "div",
    "MatMul": "matmul",
    "Max": "maximum",
    "Min": "minimum",
}

COMPOSITE = ("Gemm", "Softmax", "Clip", "LeakyRelu", "ReduceMean")

UNSUPPORTED = ("Conv", "LSTM", "Erf", "TopK", "NonMaxSuppression")


@dataclass
class ForeignNode:
    """One operation as the exchange format wrote it."""

    name: str
    op: str
    inputs: tuple[str, ...]
    attrs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"name": self.name, "op": self.op, "inputs": list(self.inputs)}


@dataclass
class ForeignGraph:
    """A whole model in the foreign format."""

    inputs: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    nodes: list[ForeignNode] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    @property
    def op_counts(self) -> dict[str, int]:
        """How many of each operation the model holds."""
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.op] = counts.get(node.op, 0) + 1
        return counts

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "inputs": len(self.inputs),
            "nodes": len(self.nodes),
            "outputs": len(self.outputs),
            "operations": dict(sorted(self.op_counts.items())),
        }


def parse(text: str) -> ForeignGraph:
    """Read the foreign format.

    Deliberately not the format ir/serialize.py writes. An importer that shares a parser with
    the exporter is testing the parser against itself, and the whole point of an importer is to
    read something it did not write.
    """
    if not text.strip():
        raise ConfigError("there is nothing to parse")
    graph = ForeignGraph()
    for number, raw in enumerate(text.strip().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("input "):
            graph.inputs.append(_parse_input(line, number))
            continue
        if line.startswith("output "):
            graph.outputs.extend(part.strip() for part in line[len("output ") :].split(","))
            continue
        graph.nodes.append(_parse_node(line, number))
    if not graph.outputs:
        raise ConfigError("a model with no outputs computes nothing")
    return graph


def _parse_input(line: str, number: int) -> tuple[str, tuple[int, ...]]:
    """One input declaration."""
    body = line[len("input ") :]
    if ":" not in body:
        raise ConfigError(f"line {number}: an input needs a shape")
    name, shape = body.split(":", 1)
    try:
        sizes = tuple(int(part) for part in shape.strip().split("x"))
    except ValueError as error:
        raise ConfigError(f"line {number}: {shape.strip()!r} is not a shape") from error
    return name.strip(), sizes


def _parse_node(line: str, number: int) -> ForeignNode:
    """One operation."""
    if "=" not in line:
        raise ConfigError(f"line {number}: expected an assignment")
    name, body = line.split("=", 1)
    body = body.strip()
    if "(" not in body or not body.endswith(")"):
        raise ConfigError(f"line {number}: expected an operation call")

    head, rest = body.split("(", 1)
    arguments = rest[:-1]
    inputs: list[str] = []
    attrs: dict = {}
    for part in arguments.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "=" in piece:
            key, value = piece.split("=", 1)
            attrs[key.strip()] = _parse_attribute(value.strip())
            continue
        inputs.append(piece)
    return ForeignNode(name=name.strip(), op=head.strip(), inputs=tuple(inputs), attrs=attrs)


def _parse_attribute(text: str) -> object:
    """An attribute value, as an integer or a float or a string."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text.strip('"')


def classify(op: str) -> str:
    """Which of the three groups an operation falls into."""
    if op in DIRECT:
        return "direct"
    if op in COMPOSITE:
        return "composite"
    return "unsupported"


def coverage(graph: ForeignGraph) -> dict:
    """How much of a model this compiler can read.

    Counted in nodes rather than in distinct operations, which is the number that decides
    whether a model imports. A format with three hundred operations in it and a model using
    twenty of them is a coverage question about the twenty.
    """
    counts = graph.op_counts
    groups = {"direct": 0, "composite": 0, "unsupported": 0}
    for op, count in counts.items():
        groups[classify(op)] += count
    total = sum(groups.values())
    return {
        "nodes": total,
        **groups,
        "importable_share": round((groups["direct"] + groups["composite"]) / total, 4)
        if total
        else 0.0,
    }


def unsupported_operations(graph: ForeignGraph) -> list[str]:
    """Which operations stop a model from importing, in the order they appear."""
    seen: list[str] = []
    for node in graph.nodes:
        if classify(node.op) == "unsupported" and node.op not in seen:
            seen.append(node.op)
    return seen


def to_graph(foreign: ForeignGraph) -> Graph:
    """Lower a foreign model into this compiler's operation set.

    Refuses rather than approximates. An operation with no lowering produces an error naming it,
    because the alternative is an importer that drops something and hands back a graph that
    computes most of a model.
    """
    missing = unsupported_operations(foreign)
    if missing:
        raise GraphError(f"no lowering for {missing}")
    if not foreign.inputs:
        raise ConfigError("a model needs at least one input")

    builder = Builder()
    mapping: dict[str, str] = {}
    for name, sizes in foreign.inputs:
        mapping[name] = builder.input(list(sizes), name=name)

    for node in foreign.nodes:
        missing_inputs = [name for name in node.inputs if name not in mapping]
        if missing_inputs:
            raise GraphError(f"{node.name} reads {missing_inputs} before they are defined")
        operands = [mapping[name] for name in node.inputs]
        mapping[node.name] = _lower(builder, node, operands)

    unknown = [name for name in foreign.outputs if name not in mapping]
    if unknown:
        raise GraphError(f"{unknown} are outputs that were never produced")
    return builder.finish(*[mapping[name] for name in foreign.outputs])


def _lower(builder: Builder, node: ForeignNode, operands: Sequence[str]) -> str:
    """One foreign operation as one or more of this compiler's."""
    if node.op in DIRECT:
        return builder.apply(get_op(DIRECT[node.op]), *operands)
    if node.op == "Gemm":
        return _lower_gemm(builder, operands)
    if node.op == "Softmax":
        return _lower_softmax(builder, node, operands)
    if node.op == "Clip":
        return _lower_clip(builder, node, operands)
    if node.op == "LeakyRelu":
        return _lower_leaky(builder, node, operands)
    if node.op == "ReduceMean":
        return _lower_reduce_mean(builder, node, operands)
    raise GraphError(f"no lowering for {node.op}")


def _lower_gemm(builder: Builder, operands: Sequence[str]) -> str:
    """A general matrix product, with the optional bias."""
    if len(operands) not in (2, 3):
        raise GraphError(f"a product takes two or three operands, got {len(operands)}")
    product = builder.matmul(operands[0], operands[1])
    if len(operands) == 2:
        return product
    sizes = [size.value for size in builder.shape_of(product).sizes]
    return builder.add(product, builder.broadcast_to(operands[2], sizes))


def _lower_softmax(builder: Builder, node: ForeignNode, operands: Sequence[str]) -> str:
    """A softmax over the axis the foreign node named.

    The axis is the whole difficulty. This compiler has a maximum, a subtraction, an
    exponential, a sum and a division, and the foreign operation has one attribute saying which
    dimension all of those run over. Getting it wrong produces a graph that runs and computes a
    different function, which is what the comparison at the bottom of this file is for.
    """
    axis = int(node.attrs.get("axis", -1))
    sizes = [size.value for size in builder.shape_of(operands[0]).sizes]
    if axis < 0:
        axis += len(sizes)
    if not 0 <= axis < len(sizes):
        raise GraphError(f"axis {node.attrs.get('axis')} is outside a rank {len(sizes)} value")

    peak = builder.max(operands[0], axes=[axis], keepdims=True)
    shifted = builder.sub(operands[0], builder.broadcast_to(peak, sizes))
    weights = builder.exp(shifted)
    total = builder.sum(weights, axes=[axis], keepdims=True)
    return builder.div(weights, builder.broadcast_to(total, sizes))


def _lower_clip(builder: Builder, node: ForeignNode, operands: Sequence[str]) -> str:
    """A clamp, as a maximum against a minimum."""
    low = float(node.attrs.get("min", 0.0))
    high = float(node.attrs.get("max", 1.0))
    if high < low:
        raise GraphError(f"a clip from {low} to {high} keeps nothing")
    lifted = builder.maximum(operands[0], builder.constant(low))
    return builder.apply(get_op("minimum"), lifted, builder.constant(high))


def _lower_leaky(builder: Builder, node: ForeignNode, operands: Sequence[str]) -> str:
    """A rectifier with a slope below zero."""
    slope = float(node.attrs.get("alpha", 0.01))
    if slope < 0:
        raise GraphError(f"a negative slope of {slope} is not a leak")
    scaled = builder.mul(operands[0], builder.constant(slope))
    return builder.maximum(operands[0], scaled)


def _lower_reduce_mean(builder: Builder, node: ForeignNode, operands: Sequence[str]) -> str:
    """An average over the named axes."""
    axes = node.attrs.get("axes", -1)
    sizes = [size.value for size in builder.shape_of(operands[0]).sizes]
    axis = int(axes)
    if axis < 0:
        axis += len(sizes)
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    return builder.mean(operands[0], axes=[axis], keepdims=keepdims)


def expansion(foreign: ForeignGraph) -> dict:
    """How many nodes the lowering produces from how many it read.

    More, and the ratio is what a composite costs. One softmax becomes seven operations here,
    so a model that looks small in the exchange format is not small once it arrives, and every
    pass downstream sees the larger number.
    """
    lowered = to_graph(foreign)
    return {
        "foreign_nodes": len(foreign.nodes),
        "lowered_nodes": len(lowered.nodes),
        "ratio": round(len(lowered.nodes) / len(foreign.nodes), 3) if foreign.nodes else 0.0,
    }


SAMPLE = """
# a small model in the exchange format
input x : 8x32
input w : 32x32
input b : 32
v0 = Gemm(x, w, b)
v1 = Relu(v0)
v2 = Softmax(v1, axis=1)
output v2
"""

UNSUPPORTED_SAMPLE = """
input x : 1x3x8x8
input k : 4x3x3x3
v0 = Conv(x, k)
output v0
"""


def sample_graph() -> ForeignGraph:
    """The model every measurement here runs over."""
    return parse(SAMPLE)


def the_sample_imports(*, seed: int = 0) -> dict:
    """The imported graph against the same computation written by hand.

    The check that says the lowering means what the foreign operation meant. A lowering that
    reduced over the wrong axis, or put the bias in the wrong place, produces a graph of the
    right shape and this comparison is the only thing that notices.
    """
    graph = to_graph(sample_graph())
    generator = torch.Generator().manual_seed(seed)
    feeds = {
        "x": torch.randn(8, 32, generator=generator),
        "w": torch.randn(32, 32, generator=generator),
        "b": torch.randn(32, generator=generator),
    }
    mine = run(graph, feeds)[0]
    theirs = torch.softmax(torch.relu(feeds["x"] @ feeds["w"] + feeds["b"]), dim=1)
    gap = float((mine - theirs).abs().max())
    return {"largest_gap": gap, "agrees": gap < 1e-5}


def the_axis_is_not_optional(*, seed: int = 0) -> dict:
    """What a lowering that ignored the axis would produce.

    A different function, and one that looks fine. Both graphs have the same shape, the same
    operations and the same node count, and their outputs differ by most of the range, because
    a softmax over rows and a softmax over columns are not close to each other.
    """
    right = to_graph(parse(SAMPLE))
    wrong = to_graph(parse(SAMPLE.replace("axis=1", "axis=0")))
    generator = torch.Generator().manual_seed(seed)
    feeds = {
        "x": torch.randn(8, 32, generator=generator),
        "w": torch.randn(32, 32, generator=generator),
        "b": torch.randn(32, generator=generator),
    }
    over_rows = run(right, feeds)[0]
    over_columns = run(wrong, feeds)[0]
    return {
        "same_shape": list(over_rows.shape) == list(over_columns.shape),
        "same_node_count": len(right.nodes) == len(wrong.nodes),
        "largest_gap": float((over_rows - over_columns).abs().max()),
    }


def coverage_of_the_sample() -> dict:
    """How much of the sample model imports."""
    return coverage(sample_graph())


def an_unsupported_model_is_refused() -> dict:
    """What happens to a model with an operation this compiler does not have.

    It is refused and the operation is named. Dropping it and importing the rest would produce a
    graph that computes most of a model, which is worse than not importing it, because it runs.
    """
    foreign = parse(UNSUPPORTED_SAMPLE)
    try:
        to_graph(foreign)
    except GraphError as error:
        return {"refused": True, "named": "Conv" in str(error)}
    return {"refused": False, "named": False}


def operation_groups() -> dict:
    """How the operation set divides into the three groups."""
    return {
        "direct": len(DIRECT),
        "composite": len(COMPOSITE),
        "unsupported": len(UNSUPPORTED),
    }


def composites_are_where_the_risk_is() -> dict:
    """How many attributes each group carries.

    None of the direct ones and all of the composite ones. A rename cannot be got wrong; a
    lowering that has to read an axis, a slope or a pair of bounds and place them correctly can
    be, and that is where an importer needs its tests.
    """
    graph = sample_graph()
    with_attributes = [node.op for node in graph.nodes if node.attrs]
    return {
        "nodes": len(graph.nodes),
        "with_attributes": len(with_attributes),
        "all_composite": all(classify(op) == "composite" for op in with_attributes),
    }


def the_lowering_grows_the_graph() -> dict:
    """How much larger the model gets on the way in."""
    return expansion(sample_graph())


def parse_errors() -> list[dict]:
    """Every way the parser refuses a line, with what it says.

    Listed rather than tested one at a time, because the useful property of a parser's errors is
    that they name the line and say what was expected, and a table is the readable way to check
    that they all do.
    """
    cases = (
        ("input x\noutput x", "an input needs a shape"),
        ("input x : eight\noutput x", "is not a shape"),
        ("input x : 8x8\nv0 Relu(x)\noutput v0", "expected an assignment"),
        ("input x : 8x8\nv0 = Relu x\noutput v0", "expected an operation call"),
        ("input x : 8x8\nv0 = Relu(x)", "no outputs"),
    )
    rows = []
    for text, expected in cases:
        try:
            parse(text)
        except ConfigError as error:
            rows.append(
                {
                    "expected": expected,
                    "said": str(error),
                    "matched": expected in str(error),
                }
            )
            continue
        rows.append({"expected": expected, "said": "", "matched": False})
    return rows


def every_parse_error_names_the_problem() -> dict:
    """Whether all of those refusals say what was wrong."""
    rows = parse_errors()
    return {"cases": len(rows), "matched": sum(1 for row in rows if row["matched"])}


def a_forward_reference_is_refused() -> bool:
    """Whether a node reading a value defined later is caught.

    The exchange format does not promise topological order and this importer requires it, so the
    check has to be here rather than assumed. A forward reference that slipped through would
    produce a graph the validator refuses much later with a message about the wrong thing.
    """
    text = "input x : 8x8\nv0 = Relu(v1)\nv1 = Tanh(x)\noutput v0"
    try:
        to_graph(parse(text))
    except GraphError:
        return True
    return False


def an_output_that_was_never_produced_is_refused() -> bool:
    """Whether naming an output nothing computes is caught."""
    text = "input x : 8x8\nv0 = Relu(x)\noutput v9"
    try:
        to_graph(parse(text))
    except GraphError:
        return True
    return False


def comments_and_blank_lines_are_ignored() -> dict:
    """Whether the parser reads a file a person wrote rather than a file a program wrote."""
    text = "\n# a comment\n\ninput x : 8x8  # trailing\nv0 = Relu(x)\noutput v0\n"
    graph = parse(text)
    return {"inputs": len(graph.inputs), "nodes": len(graph.nodes)}
