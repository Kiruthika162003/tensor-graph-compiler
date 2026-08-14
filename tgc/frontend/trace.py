from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, GraphError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.dtype import FLOAT32, DType
from tgc.ir.graph import Graph
from tgc.verify.reference import outputs_agree, random_feeds, run

# Recording what a Python function does to tensors.
#
# Tracing runs the function once with proxy objects standing in for tensors, and every
# operation performed on a proxy becomes a node. That is the whole mechanism, and its
# limitation is the same sentence: the trace records what happened on the one execution it
# saw. A branch on a tensor value takes one side and the other side is gone. A Python loop is
# unrolled to whatever length it ran. A print statement leaves nothing behind.
#
# None of that is fixable inside tracing and all of it is worth being loud about, because the
# failure is silent: the traced graph is a perfectly good graph, it validates, it runs, and it
# computes the function only for inputs that take the same path. The alternative is parsing
# the source, which is a different tool with a different set of things it cannot do.


@dataclass
class Proxy:
    """A stand in for a tensor that records operations instead of performing them."""

    name: str
    builder: Builder
    shape: tuple[int | str, ...] = ()
    dtype: DType = FLOAT32

    def _binary(self, other: Proxy | float, operation: ops.Op, *, flip: bool = False) -> Proxy:
        """Record a two operand operation."""
        if isinstance(other, Proxy):
            right = other.name
            shape = other.shape if len(other.shape) > len(self.shape) else self.shape
        else:
            right = self.builder.constant(float(other))
            shape = self.shape
        left = self.name
        if flip:
            left, right = right, left
        produced = self.builder.apply(operation, left, right)
        return Proxy(name=produced, builder=self.builder, shape=shape, dtype=self.dtype)

    def _unary(self, operation: ops.Op) -> Proxy:
        """Record a one operand operation."""
        produced = self.builder.apply(operation, self.name)
        return Proxy(name=produced, builder=self.builder, shape=self.shape, dtype=self.dtype)

    def __add__(self, other):
        return self._binary(other, ops.ADD)

    def __radd__(self, other):
        return self._binary(other, ops.ADD, flip=True)

    def __sub__(self, other):
        return self._binary(other, ops.SUB)

    def __rsub__(self, other):
        return self._binary(other, ops.SUB, flip=True)

    def __mul__(self, other):
        return self._binary(other, ops.MUL)

    def __rmul__(self, other):
        return self._binary(other, ops.MUL, flip=True)

    def __truediv__(self, other):
        return self._binary(other, ops.DIV)

    def __rtruediv__(self, other):
        return self._binary(other, ops.DIV, flip=True)

    def __neg__(self):
        return self._unary(ops.NEG)

    def __matmul__(self, other):
        if not isinstance(other, Proxy):
            raise GraphError("a matrix product needs two tensors")
        produced = self.builder.matmul(self.name, other.name)
        return Proxy(
            name=produced,
            builder=self.builder,
            shape=(*self.shape[:-1], other.shape[-1]),
            dtype=self.dtype,
        )

    def __bool__(self) -> bool:
        """Refuse to be used as a condition.

        The single most valuable line in the file. Branching on a traced value takes one side
        and silently discards the other, so the graph is correct for the inputs that took that
        path and wrong for the rest. Python calls this to evaluate an if, so raising here turns
        a silent wrong answer into an error at the line that caused it.
        """
        raise GraphError(
            f"cannot branch on {self.name}: tracing sees one execution, so an if on a tensor "
            "records the side it took and loses the other"
        )

    def relu(self) -> Proxy:
        """Record a rectifier."""
        return self._unary(ops.RELU)

    def exp(self) -> Proxy:
        """Record an exponential."""
        return self._unary(ops.EXP)

    def tanh(self) -> Proxy:
        """Record a hyperbolic tangent."""
        return self._unary(ops.TANH)

    def sqrt(self) -> Proxy:
        """Record a square root."""
        return self._unary(ops.SQRT)

    def sigmoid(self) -> Proxy:
        """Record a logistic."""
        return self._unary(ops.SIGMOID)

    def sum(self, dim: int = -1, keepdim: bool = True) -> Proxy:
        """Record a sum over one axis."""
        produced = self.builder.sum(self.name, axes=[dim], keepdims=keepdim)
        return Proxy(name=produced, builder=self.builder, shape=self.shape, dtype=self.dtype)

    def mean(self, dim: int = -1, keepdim: bool = True) -> Proxy:
        """Record a mean over one axis."""
        produced = self.builder.mean(self.name, axes=[dim], keepdims=keepdim)
        return Proxy(name=produced, builder=self.builder, shape=self.shape, dtype=self.dtype)

    def amax(self, dim: int = -1, keepdim: bool = True) -> Proxy:
        """Record a maximum over one axis.

        Named amax rather than max, and taking dim and keepdim rather than axis and keepdims,
        because the proxy has to be spelled exactly like a torch tensor. The same function
        source has to run on both, or the comparison against eager execution is comparing two
        different functions and passing for the wrong reason. That is how the first version of
        this file passed its own test while computing a different softmax.
        """
        produced = self.builder.max(self.name, axes=[dim], keepdims=keepdim)
        return Proxy(name=produced, builder=self.builder, shape=self.shape, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"Proxy({self.name})"


@dataclass
class TraceReport:
    """What one trace recorded and what it could not see."""

    graph: Graph
    unrolled_loops: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def nodes(self) -> int:
        """Operations recorded."""
        return len(self.graph.nodes)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "nodes": self.nodes,
            "inputs": len(self.graph.inputs),
            "notes": list(self.notes),
        }


def trace(
    function: Callable,
    shapes: Sequence[Sequence[int | str]],
    *,
    names: Sequence[str] | None = None,
    dtype: DType = FLOAT32,
) -> Graph:
    """Run a function on proxies and keep what it did.

    The shapes are supplied rather than inferred because the trace has to build inputs before
    the function runs. That makes the specialisation explicit: a graph traced at one shape is a
    graph for that shape, and the guard machinery in runtime/guards.py is what decides when it
    may be reused.
    """
    if not shapes:
        raise ConfigError("a traced function needs at least one input")
    chosen = list(names) if names else [f"in{index}" for index in range(len(shapes))]
    if len(chosen) != len(shapes):
        raise ConfigError("there has to be one name per input")

    builder = Builder()
    proxies = []
    for name, sizes in zip(chosen, shapes, strict=True):
        declared = builder.input(list(sizes), dtype=dtype, name=name)
        proxies.append(Proxy(name=declared, builder=builder, shape=tuple(sizes), dtype=dtype))

    result = function(*proxies)
    if isinstance(result, Proxy):
        result = [result]
    if not result:
        raise GraphError("the traced function returned nothing")
    for item in result:
        if not isinstance(item, Proxy):
            raise GraphError(
                f"the traced function returned {type(item).__name__}, which the trace never "
                "saw being computed"
            )
    return builder.finish(*[item.name for item in result])


def trace_with_report(
    function: Callable, shapes: Sequence[Sequence[int | str]], **kwargs
) -> TraceReport:
    """Trace a function and note what tracing could not capture."""
    graph = trace(function, shapes, **kwargs)
    notes = []
    if len(graph.nodes) > 3 * len(shapes) + 10:
        notes.append("the trace is long, which usually means a Python loop was unrolled")
    return TraceReport(graph=graph, notes=notes)


def unrolled_length(function: Callable, shapes: Sequence[Sequence[int | str]]) -> int:
    """How many nodes a trace produced, which for a loop is the iteration count."""
    return len(trace(function, shapes).nodes)


def compare_unrolling(
    make: Callable[[int], Callable], shapes: Sequence[Sequence[int | str]]
) -> list[dict]:
    """Trace the same function at several loop counts.

    The node count grows with the loop bound, which is the observable consequence of the
    thing tracing cannot do. A function whose loop runs a thousand times produces a thousand
    nodes and compiles slowly, and the graph is still only valid for that count.
    """
    rows = []
    for count in (1, 2, 4, 8):
        rows.append({"iterations": count, "nodes": unrolled_length(make(count), shapes)})
    return rows


def eager(function: Callable, feeds: Sequence[torch.Tensor]) -> torch.Tensor:
    """Run the same function on real tensors, for comparing against the traced graph."""
    return function(*feeds)


def traced_matches_eager(
    function: Callable,
    shapes: Sequence[Sequence[int]],
    *,
    seed: int = 0,
    tolerance: float = 0.0,
) -> bool:
    """Whether the traced graph computes what the function computed.

    Bit equality. The trace records the same operations in the same order, so there is nothing
    for a tolerance to forgive, and needing one would mean the trace had rearranged something.
    """
    graph = trace(function, shapes)
    feeds = random_feeds(graph, seed=seed, positive=True)
    ordered = [feeds[value.name] for value in graph.inputs]
    expected = eager(function, ordered)
    if isinstance(expected, torch.Tensor):
        expected = [expected]
    return outputs_agree(run(graph, feeds), list(expected), tolerance=tolerance)


def softmax(x):
    """A stable softmax, written so the same source runs on a proxy and on a tensor."""
    shifted = x - x.amax(dim=-1, keepdim=True)
    exponentiated = shifted.exp()
    return exponentiated / exponentiated.sum(dim=-1, keepdim=True)


def layernorm(x):
    """Normalisation over the last dimension, in the same dual purpose style."""
    centred = x - x.mean(dim=-1, keepdim=True)
    variance = (centred * centred).mean(dim=-1, keepdim=True)
    return centred / (variance + 1e-5).sqrt()


def feed_forward(x, up, down):
    """A feed forward block."""
    return (x @ up).relu() @ down


def branching(x):
    """A function that branches on a tensor value, which tracing cannot record."""
    if x.sum(dim=-1, keepdim=True):
        return x.relu()
    return x.tanh()


def make_loop(iterations: int) -> Callable:
    """A function with a Python loop of a given length, for showing unrolling."""
    if iterations < 1:
        raise ConfigError(f"the loop has to run, got {iterations}")

    def looped(x):
        current = x
        for _ in range(iterations):
            current = current.tanh()
        return current

    return looped
