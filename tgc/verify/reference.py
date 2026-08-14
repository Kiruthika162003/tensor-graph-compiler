from __future__ import annotations

from collections.abc import Sequence

import torch

from tgc.errors import CompilerError, ConfigError, GraphError
from tgc.ir.dtype import (
    BFLOAT16,
    BOOL,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    INT32,
    INT64,
    DType,
)
from tgc.ir.graph import Graph, Node

# The obvious interpreter, written to be correct rather than fast.
#
# Every optimisation in this compiler is checked against this file. It materialises a full
# tensor for every node, evaluates in the order the nodes are written, and does nothing
# clever, which is the point: it is small enough to read in one sitting and decide by
# inspection that it is right.
#
# It is also the reason the numeric claims elsewhere can be sharp. When a fused kernel and
# this interpreter agree bit for bit, that is a statement about the transformation rather
# than about a tolerance somebody chose.

TORCH_DTYPES: dict[str, torch.dtype] = {
    BOOL.name: torch.bool,
    INT8.name: torch.int8,
    INT32.name: torch.int32,
    INT64.name: torch.int64,
    FLOAT16.name: torch.float16,
    BFLOAT16.name: torch.bfloat16,
    FLOAT32.name: torch.float32,
    FLOAT64.name: torch.float64,
}


def to_torch(dtype: DType) -> torch.dtype:
    """The torch type matching one of ours."""
    if dtype.name not in TORCH_DTYPES:
        raise ConfigError(f"no torch equivalent for {dtype}")
    return TORCH_DTYPES[dtype.name]


UNARY = {
    "neg": torch.neg,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "reciprocal": torch.reciprocal,
    "abs": torch.abs,
}

BINARY = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
}


def evaluate_node(node: Node, operands: Sequence[torch.Tensor]) -> torch.Tensor:
    """Run one operation.

    Reductions widen to the accumulator the type rules chose and cast back afterwards only if
    the inferred output says so. Doing the widening here rather than leaving it to torch is
    what makes the interpreter agree with the compiler about what a float16 sum means.
    """
    name = node.op.name
    if name in UNARY:
        return UNARY[name](operands[0])
    if name in BINARY:
        return BINARY[name](operands[0], operands[1])
    if name == "cast":
        return operands[0].to(to_torch(node.output.dtype))
    if name == "matmul":
        return operands[0] @ operands[1]
    if name in ("sum", "mean", "max"):
        return _reduce(node, operands[0])
    if name == "reshape":
        return operands[0].reshape(tuple(node.attrs["sizes"]))
    if name == "transpose":
        return operands[0].permute(tuple(node.attrs["permutation"]))
    if name == "broadcast_to":
        return operands[0].broadcast_to(_static_sizes(node)).contiguous()
    if name == "constant":
        return torch.full((), float(node.attrs["value"]), dtype=to_torch(node.output.dtype))
    if name in ("print", "assert_finite"):
        return operands[0]
    raise CompilerError(f"the interpreter does not implement {name}")


def _reduce(node: Node, source: torch.Tensor) -> torch.Tensor:
    """Run a reduction in the accumulator type the compiler chose."""
    axes = tuple(node.attrs["axes"])
    keepdims = bool(node.attrs.get("keepdims", False))
    accumulator = to_torch(node.output.dtype)
    if node.op.name == "max":
        result = source
        for axis in sorted(axes, reverse=True):
            result = result.amax(dim=axis, keepdim=True)
        return result if keepdims else result.squeeze(dim=axes)
    widened = source.to(accumulator) if accumulator != source.dtype else source
    if node.op.name == "sum":
        return widened.sum(dim=axes, keepdim=keepdims)
    return widened.mean(dim=axes, keepdim=keepdims)


def _static_sizes(node: Node) -> tuple[int, ...]:
    """The concrete sizes of a node's output."""
    sizes = []
    for size in node.output.shape.sizes:
        if not size.is_static:
            raise GraphError(f"{node.name} has a symbolic shape and cannot be materialised")
        sizes.append(size.value or 0)
    return tuple(sizes)


def interpret(graph: Graph, feeds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Evaluate a whole graph, keeping every intermediate.

    Keeping everything is what makes this useful for checking a transformation: a pass that
    changes a value halfway through a graph can be caught at that value rather than at the
    output, where several errors may have cancelled.
    """
    missing = [value.name for value in graph.inputs if value.name not in feeds]
    if missing:
        raise GraphError(f"no value supplied for {missing}")

    environment: dict[str, torch.Tensor] = {}
    for value in graph.inputs:
        supplied = feeds[value.name]
        expected = to_torch(value.dtype)
        if supplied.dtype != expected:
            raise GraphError(
                f"{value.name} was declared {value.dtype} and was given {supplied.dtype}"
            )
        environment[value.name] = supplied

    for node in graph.nodes:
        operands = [environment[name] for name in node.inputs]
        environment[node.name] = evaluate_node(node, operands)
    return environment


def run(graph: Graph, feeds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Evaluate a graph and return only what it was asked for."""
    environment = interpret(graph, feeds)
    return [environment[name] for name in graph.outputs]


def random_feeds(
    graph: Graph, *, seed: int = 0, scale: float = 1.0, positive: bool = False
) -> dict[str, torch.Tensor]:
    """Inputs of the right shape and type for a graph.

    The positive option exists because logarithms and square roots of negative numbers are
    not a useful way to discover that a transformation is wrong: every comparison becomes nan
    against nan, which compares unequal, and the failure says nothing about the pass.
    """
    generator = torch.Generator().manual_seed(seed)
    feeds = {}
    for value in graph.inputs:
        sizes = []
        for size in value.shape.sizes:
            if not size.is_static:
                raise GraphError(f"{value.name} has a symbolic shape, so it cannot be filled")
            sizes.append(size.value or 0)
        torch_dtype = to_torch(value.dtype)
        if value.dtype.is_float:
            sample = torch.randn(sizes, generator=generator, dtype=torch.float32) * scale
            if positive:
                sample = sample.abs() + 0.5
            feeds[value.name] = sample.to(torch_dtype)
        elif value.dtype is BOOL:
            feeds[value.name] = torch.randint(
                0, 2, sizes, generator=generator, dtype=torch.int64
            ).to(torch.bool)
        else:
            feeds[value.name] = torch.randint(
                -4, 5, sizes, generator=generator, dtype=torch.int64
            ).to(torch_dtype)
    return feeds


def outputs_agree(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor], *, tolerance: float = 0.0
) -> bool:
    """Whether two runs produced the same answers.

    A tolerance of zero means bit equality, which is the right check for a transformation that
    only reorders reads and writes. A pass that changes accumulation order needs a tolerance,
    and having to pass one is a useful reminder that it changed the arithmetic.
    """
    if len(left) != len(right):
        return False
    for first, second in zip(left, right, strict=True):
        if first.shape != second.shape:
            return False
        if tolerance == 0.0:
            if not torch.equal(first, second):
                return False
        elif not torch.allclose(first.float(), second.float(), atol=tolerance, rtol=tolerance):
            return False
    return True


def largest_difference(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    """The worst disagreement between two runs, in absolute terms."""
    if len(left) != len(right):
        raise ConfigError("the two runs produced different numbers of outputs")
    worst = 0.0
    for first, second in zip(left, right, strict=True):
        worst = max(worst, float((first.float() - second.float()).abs().max()))
    return worst


def relative_difference(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    """The worst disagreement, scaled by the size of the values involved."""
    scale = max(
        (float(tensor.float().abs().max()) for tensor in left),
        default=0.0,
    )
    if scale == 0.0:
        return largest_difference(left, right)
    return largest_difference(left, right) / scale
