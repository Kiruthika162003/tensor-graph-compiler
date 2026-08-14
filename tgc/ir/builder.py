from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, GraphError
from tgc.ir import op as ops
from tgc.ir.dtype import FLOAT32, DType
from tgc.ir.graph import Graph, Node, Value, infer_output, validate
from tgc.ir.shape import Shape, shape

# Building graphs without writing node lists by hand.
#
# The builder is the only thing in the compiler allowed to invent value names, and it does so
# from a counter rather than from the op name. Names derived from ops collide the moment a
# graph has two additions, and a collision in a single assignment representation is not a
# naming annoyance, it silently reroutes a reader to the wrong producer.


@dataclass
class Builder:
    """Accumulates nodes and hands back a validated graph."""

    nodes: list[Node] = field(default_factory=list)
    inputs: list[Value] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    prefix: str = "v"
    _counter: int = field(default=0, repr=False)

    def fresh(self) -> str:
        """The next unused value name."""
        name = f"{self.prefix}{self._counter}"
        self._counter += 1
        return name

    def input(self, sizes: Sequence[int | str], dtype: DType = FLOAT32, name: str = "") -> str:
        """Declare a graph input."""
        chosen = name or self.fresh()
        if chosen in self._defined():
            raise GraphError(f"{chosen!r} is already defined")
        self.inputs.append(Value(name=chosen, shape=shape(*sizes), dtype=dtype))
        return chosen

    def constant(
        self, value: float, sizes: Sequence[int | str] = (), dtype: DType = FLOAT32
    ) -> str:
        """Add a literal."""
        name = self.fresh()
        result = Value(name=name, shape=shape(*sizes) if sizes else Shape(), dtype=dtype)
        self.nodes.append(
            Node(op=ops.CONSTANT, inputs=(), output=result, attrs={"value": float(value)})
        )
        return name

    def apply(self, op: ops.Op, *names: str, **attrs) -> str:
        """Add a node, inferring its output type."""
        operands = [self._lookup(name) for name in names]
        output_name = self.fresh()
        result = infer_output(op, operands, attrs, output_name)
        self.nodes.append(Node(op=op, inputs=tuple(names), output=result, attrs=dict(attrs)))
        return output_name

    def add(self, left: str, right: str) -> str:
        """Elementwise sum."""
        return self.apply(ops.ADD, left, right)

    def sub(self, left: str, right: str) -> str:
        """Elementwise difference."""
        return self.apply(ops.SUB, left, right)

    def mul(self, left: str, right: str) -> str:
        """Elementwise product."""
        return self.apply(ops.MUL, left, right)

    def div(self, left: str, right: str) -> str:
        """Elementwise quotient."""
        return self.apply(ops.DIV, left, right)

    def maximum(self, left: str, right: str) -> str:
        """Elementwise larger of two."""
        return self.apply(ops.MAXIMUM, left, right)

    def neg(self, name: str) -> str:
        """Elementwise negation."""
        return self.apply(ops.NEG, name)

    def exp(self, name: str) -> str:
        """Elementwise exponential."""
        return self.apply(ops.EXP, name)

    def log(self, name: str) -> str:
        """Elementwise logarithm."""
        return self.apply(ops.LOG, name)

    def sqrt(self, name: str) -> str:
        """Elementwise square root."""
        return self.apply(ops.SQRT, name)

    def tanh(self, name: str) -> str:
        """Elementwise hyperbolic tangent."""
        return self.apply(ops.TANH, name)

    def relu(self, name: str) -> str:
        """Elementwise rectifier."""
        return self.apply(ops.RELU, name)

    def sigmoid(self, name: str) -> str:
        """Elementwise logistic."""
        return self.apply(ops.SIGMOID, name)

    def reciprocal(self, name: str) -> str:
        """Elementwise inverse."""
        return self.apply(ops.RECIPROCAL, name)

    def cast(self, name: str, dtype: DType) -> str:
        """Change element type."""
        return self.apply(ops.CAST, name, dtype=dtype)

    def sum(self, name: str, axes: Sequence[int], *, keepdims: bool = False) -> str:
        """Reduce by addition."""
        return self.apply(ops.SUM, name, axes=tuple(axes), keepdims=keepdims)

    def mean(self, name: str, axes: Sequence[int], *, keepdims: bool = False) -> str:
        """Reduce by averaging."""
        return self.apply(ops.MEAN, name, axes=tuple(axes), keepdims=keepdims)

    def max(self, name: str, axes: Sequence[int], *, keepdims: bool = False) -> str:
        """Reduce by taking the largest."""
        return self.apply(ops.MAX, name, axes=tuple(axes), keepdims=keepdims)

    def matmul(self, left: str, right: str) -> str:
        """Matrix product."""
        return self.apply(ops.MATMUL, left, right)

    def reshape(self, name: str, sizes: Sequence[int]) -> str:
        """Reinterpret the same elements with different dimensions."""
        return self.apply(ops.RESHAPE, name, sizes=tuple(sizes))

    def transpose(self, name: str, permutation: Sequence[int]) -> str:
        """Permute dimensions."""
        return self.apply(ops.TRANSPOSE, name, permutation=tuple(permutation))

    def broadcast_to(self, name: str, sizes: Sequence[int | str]) -> str:
        """Expand a tensor against a larger shape."""
        return self.apply(ops.BROADCAST_TO, name, shape=shape(*sizes))

    def emit(self, name: str) -> str:
        """Record a value the caller wants back."""
        if name not in self._defined():
            raise GraphError(f"cannot return {name!r}, which is not defined")
        self.outputs.append(name)
        return name

    def finish(self, *names: str) -> Graph:
        """Produce a validated graph."""
        for name in names:
            self.emit(name)
        graph = Graph(
            nodes=list(self.nodes), inputs=list(self.inputs), outputs=list(self.outputs)
        )
        validate(graph)
        return graph

    def _defined(self) -> set[str]:
        """Every name this builder has handed out."""
        return {value.name for value in self.inputs} | {node.name for node in self.nodes}

    def _lookup(self, name: str) -> Value:
        """Find a value by name."""
        for value in self.inputs:
            if value.name == name:
                return value
        for node in self.nodes:
            if node.name == name:
                return node.output
        raise GraphError(f"no value named {name!r}")


def elementwise_chain(length: int = 6, sizes: Sequence[int | str] = (64, 64)) -> Graph:
    """A run of elementwise operations with nothing else in it.

    The shape fusion exists for. Every intermediate is read once by the next operation and by
    nothing else, so an unfused version writes and reads length minus one whole tensors that
    never needed to exist.
    """
    if length < 1:
        raise ConfigError(f"a chain needs at least one operation, got {length}")
    builder = Builder()
    current = builder.input(sizes, name="x")
    for index in range(length):
        current = builder.relu(current) if index % 2 else builder.exp(current)
    return builder.finish(current)


def diamond_graph(sizes: Sequence[int | str] = (32, 32)) -> Graph:
    """A value read by two consumers and rejoined.

    The shape that catches a fusion pass that duplicates work. Fusing the shared node into
    both branches computes it twice, which is sometimes right and is never free, and a pass
    that does it without noticing has quietly traded compute for memory.
    """
    builder = Builder()
    x = builder.input(sizes, name="x")
    shared = builder.exp(x)
    left = builder.relu(shared)
    right = builder.neg(shared)
    return builder.finish(builder.add(left, right))


def mlp_graph(batch: int | str = 8, hidden: int = 64, expansion: int = 4) -> Graph:
    """A feed forward block: two matrix products with a rectifier between them."""
    if hidden < 1 or expansion < 1:
        raise ConfigError("the widths must be positive")
    builder = Builder()
    x = builder.input([batch, hidden], name="x")
    up = builder.input([hidden, hidden * expansion], name="w_up")
    down = builder.input([hidden * expansion, hidden], name="w_down")
    bias = builder.input([hidden * expansion], name="b_up")

    projected = builder.matmul(x, up)
    biased = builder.add(projected, bias)
    activated = builder.relu(biased)
    return builder.finish(builder.matmul(activated, down))


def softmax_graph(rows: int | str = 8, columns: int = 32) -> Graph:
    """A numerically stable softmax, written as the graph a frontend would trace.

    Kept as a fixture because it exercises the interesting interaction in one place: a
    reduction feeding an elementwise chain that feeds another reduction, which is exactly the
    pattern where fusing everything is wrong and fusing nothing leaves five buffers.
    """
    builder = Builder()
    x = builder.input([rows, columns], name="x")
    largest = builder.max(x, axes=[1], keepdims=True)
    shifted = builder.sub(x, largest)
    exponentiated = builder.exp(shifted)
    total = builder.sum(exponentiated, axes=[1], keepdims=True)
    return builder.finish(builder.div(exponentiated, total))


def layernorm_graph(rows: int | str = 8, columns: int = 32, epsilon: float = 1e-5) -> Graph:
    """Normalisation over the last dimension."""
    builder = Builder()
    x = builder.input([rows, columns], name="x")
    average = builder.mean(x, axes=[1], keepdims=True)
    centred = builder.sub(x, average)
    squared = builder.mul(centred, centred)
    variance = builder.mean(squared, axes=[1], keepdims=True)
    eps = builder.constant(epsilon)
    denominator = builder.sqrt(builder.add(variance, eps))
    return builder.finish(builder.div(centred, denominator))


def branching_graph(branches: int = 4, depth: int = 3, width: int = 64) -> Graph:
    """One input feeding several independent chains that rejoin at the end.

    The fixture that separates execution order from buffer allocation. Running the branches
    one at a time keeps one chain alive; running them in lockstep keeps all of them alive,
    and no allocator can recover the difference because the values genuinely overlap.
    """
    if branches < 1 or depth < 1:
        raise ConfigError("a branching graph needs at least one branch of depth one")
    builder = Builder()
    x = builder.input([width, width], name="x")

    tails = []
    for _ in range(branches):
        current = builder.relu(x)
        for _ in range(depth - 1):
            current = builder.tanh(current)
        tails.append(current)

    joined = tails[0]
    for tail in tails[1:]:
        joined = builder.add(joined, tail)
    return builder.finish(joined)
