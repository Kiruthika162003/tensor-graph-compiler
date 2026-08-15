from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.dtype import FLOAT32, FLOAT64, DType, can_represent_exactly
from tgc.ir.graph import Graph, Node, Value
from tgc.ir.shape import Shape

# Evaluating what does not depend on the inputs.
#
# The rewrite is easy and the arithmetic is not. Folding happens at compile time in whatever
# type the compiler feels like, and runs at runtime in the type the graph declared. If those
# differ the folded graph computes something the unfolded one did not, and the difference
# arrives as a model that was fine until it was compiled.
#
# Two rules keep it honest. Fold in the declared type, not in the widest available. And
# refuse to fold when the result cannot be represented in the declared type, because a
# constant that rounds on its way into the graph is a different constant.
#
# Overflow is the other refusal. Folding exp of 100 into float32 gives inf, and inf is a
# perfectly good float32 value, so nothing raises. The graph that would have produced inf at
# runtime is the same graph, but a fold that turns a finite intermediate into inf because the
# folding order differed is not.


@dataclass
class FoldReport:
    """What constant folding replaced."""

    folded: dict[str, float] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Nodes replaced by literals."""
        return len(self.folded)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "folded": self.count,
            "refused": len(self.refused),
            "reasons": dict(self.refused),
        }


UNARY_FOLDS = {
    "neg": lambda x: -x,
    "abs": abs,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "tanh": math.tanh,
    "relu": lambda x: max(0.0, x),
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    "reciprocal": lambda x: 1.0 / x,
    "step": lambda x: 1.0 if x > 0 else 0.0,
}

BINARY_FOLDS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "maximum": max,
    "minimum": min,
}


def is_constant(node: Node) -> bool:
    """Whether a node is already a literal."""
    return node.op is ops.CONSTANT


def constant_value(node: Node) -> float:
    """The number a literal holds."""
    if not is_constant(node):
        raise PassError(f"{node.name} is not a constant")
    return float(node.attrs["value"])


def constant_environment(graph: Graph) -> dict[str, float]:
    """Every value the compiler already knows."""
    return {node.name: constant_value(node) for node in graph.nodes if is_constant(node)}


def round_to(value: float, dtype: DType) -> float:
    """The number as the declared type would hold it.

    Folding in float64 and storing in float32 gives a constant the runtime graph would never
    have produced. Rounding through the declared width is what makes the folded graph and the
    original agree.
    """
    if not dtype.is_float:
        return float(int(value))
    if dtype is FLOAT64:
        return value
    code = "e" if dtype.bits == 16 else "f"
    try:
        return struct.unpack(code, struct.pack(code, value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def can_fold_into(value: float, dtype: DType) -> tuple[bool, str]:
    """Whether a computed number survives being stored in a type."""
    if math.isnan(value):
        return False, "the fold produced nan"
    if math.isinf(value):
        return False, "the fold overflowed"
    if dtype.is_integral:
        if value != int(value):
            return False, "the fold produced a fraction for an integer type"
        if not can_represent_exactly(dtype, int(value)):
            return False, f"{int(value)} does not fit in {dtype}"
        return True, ""
    rounded = round_to(value, dtype)
    if math.isinf(rounded):
        return False, f"the fold overflowed {dtype}"
    return True, ""


def evaluate_constant(node: Node, operands: list[float]) -> float:
    """Compute one node from literal operands."""
    name = node.op.name
    if name in UNARY_FOLDS:
        return float(UNARY_FOLDS[name](operands[0]))
    if name in BINARY_FOLDS:
        return float(BINARY_FOLDS[name](operands[0], operands[1]))
    if name == "cast":
        return round_to(operands[0], node.output.dtype)
    raise PassError(f"{name} cannot be folded")


def is_foldable(node: Node, known: dict[str, float]) -> bool:
    """Whether a node's operands are all known and its op can be evaluated.

    Only scalar shaped nodes. Folding a whole tensor is possible and turns a graph into its
    own weights, which is a different transformation with a different tradeoff, and running
    it by accident is how a small graph acquires a hundred megabytes of literals.
    """
    if is_constant(node) or not node.op.pure:
        return False
    foldable_op = (
        node.op.name in UNARY_FOLDS or node.op.name in BINARY_FOLDS or node.op.name == "cast"
    )
    if not foldable_op:
        return False
    if node.output.shape.rank != 0:
        return False
    return all(name in known for name in node.inputs)


def fold_constants(graph: Graph) -> Graph:
    """Replace every computable subexpression with the number it produces."""
    known = constant_environment(graph)
    kept: list[Node] = []

    for node in graph.nodes:
        if not is_foldable(node, known):
            kept.append(node)
            continue
        try:
            value = evaluate_constant(node, [known[name] for name in node.inputs])
        except (ValueError, ZeroDivisionError, OverflowError):
            kept.append(node)
            continue
        allowed, _ = can_fold_into(value, node.output.dtype)
        if not allowed:
            kept.append(node)
            continue
        folded = round_to(value, node.output.dtype)
        known[node.name] = folded
        kept.append(
            Node(op=ops.CONSTANT, inputs=(), output=node.output, attrs={"value": folded})
        )
    return graph.with_nodes(kept)


def report_folding(graph: Graph) -> FoldReport:
    """What folding would replace, and why it declined the rest."""
    known = constant_environment(graph)
    report = FoldReport()

    for node in graph.nodes:
        if not is_foldable(node, known):
            continue
        try:
            value = evaluate_constant(node, [known[name] for name in node.inputs])
        except (ValueError, ZeroDivisionError, OverflowError) as failure:
            report.refused[node.name] = str(failure)
            continue
        allowed, reason = can_fold_into(value, node.output.dtype)
        if not allowed:
            report.refused[node.name] = reason
            continue
        folded = round_to(value, node.output.dtype)
        known[node.name] = folded
        report.folded[node.name] = folded
    return report


def constant_node(name: str, value: float, dtype: DType = FLOAT32) -> Node:
    """A literal node, for tests that need one built by hand."""
    return Node(
        op=ops.CONSTANT,
        inputs=(),
        output=Value(name=name, shape=Shape(), dtype=dtype),
        attrs={"value": float(value)},
    )


def folding_precision_gap(expression: float, dtype: DType = FLOAT32) -> float:
    """How far a fold in double precision lands from one in the declared type.

    The measurement behind the first rule. A compiler that folds in whatever width Python
    hands it and stores the result in float32 has computed something the runtime graph would
    not have, and the gap is real rather than theoretical.
    """
    if dtype is FLOAT64:
        return 0.0
    wide = expression
    narrow = round_to(expression, dtype)
    return abs(wide - narrow)


def chained_fold(values: list[float], dtype: DType = FLOAT32) -> float:
    """Add a list of numbers, rounding to the declared type at every step.

    Which is what the graph would do. Summing in double and rounding once at the end is a
    different number, and it is the number a careless folder produces.
    """
    if not values:
        raise ConfigError("there is nothing to add")
    total = 0.0
    for value in values:
        total = round_to(total + value, dtype)
    return total


def careless_fold(values: list[float], dtype: DType = FLOAT32) -> float:
    """Add a list of numbers in double precision and round once at the end."""
    return round_to(math.fsum(values), dtype)
