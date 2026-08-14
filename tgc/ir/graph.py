from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, GraphError, TypeInferenceError
from tgc.ir import op as ops
from tgc.ir.dtype import BOOL, FLOAT32, DType, accumulator_for, promote_all
from tgc.ir.shape import (
    Shape,
    broadcast_all,
    matmul_shape,
    reduce_shape,
    reshape_shape,
    transpose_shape,
)

# The graph itself.
#
# Single assignment: every node produces exactly one value, and a value has exactly one
# producer. That rules out the whole class of questions about which write a reader sees, and
# it is the reason every pass in this compiler can be written as a rewrite over a value map
# rather than as a walk with a mutable environment.
#
# Nodes are held in a list and the list is kept topologically sorted. Keeping the invariant
# rather than recomputing an order on demand means a pass that breaks it fails at the point
# it broke rather than three passes later.


@dataclass(frozen=True)
class Value:
    """One tensor produced by one node."""

    name: str
    shape: Shape
    dtype: DType

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("a value needs a name")

    @property
    def bytes(self) -> int:
        """Storage the tensor takes, if that is knowable."""
        return self.shape.bytes_for(self.dtype.bytes)

    def __str__(self) -> str:
        return f"{self.name}: {self.dtype}{self.shape}"


@dataclass
class Node:
    """One operation applied to named values."""

    op: ops.Op
    inputs: tuple[str, ...]
    output: Value
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op.arity >= 0 and len(self.inputs) != self.op.arity:
            raise GraphError(f"{self.op} takes {self.op.arity} inputs, got {len(self.inputs)}")

    @property
    def name(self) -> str:
        """The name of the value this node produces."""
        return self.output.name

    def signature(self) -> tuple:
        """A key that two nodes share when they compute the same thing.

        Commutative operands are sorted so that a plus b and b plus a collide. The attributes
        are folded in because a reduction over axis zero and one over axis one are the same
        op on the same input and are not the same value, which is the mistake that makes a
        subexpression pass produce a graph that runs and is wrong.
        """
        inputs = tuple(sorted(self.inputs)) if self.op.commutative else self.inputs
        attrs = tuple(sorted((key, _hashable(value)) for key, value in self.attrs.items()))
        return (self.op.name, inputs, attrs, self.output.dtype.name)

    def replace_inputs(self, mapping: dict[str, str]) -> Node:
        """A copy reading from renamed values."""
        return Node(
            op=self.op,
            inputs=tuple(mapping.get(name, name) for name in self.inputs),
            output=self.output,
            attrs=dict(self.attrs),
        )

    def __str__(self) -> str:
        arguments = ", ".join(self.inputs)
        extra = ""
        if self.attrs:
            extra = " {" + ", ".join(f"{k}={v}" for k, v in sorted(self.attrs.items())) + "}"
        return f"{self.output.name} = {self.op.name}({arguments}){extra}"


def _hashable(value: object) -> object:
    """Turn an attribute into something that can sit in a signature."""
    if isinstance(value, list):
        return tuple(value)
    return value


@dataclass
class Graph:
    """A single assignment graph of tensor operations."""

    nodes: list[Node] = field(default_factory=list)
    inputs: list[Value] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def value_names(self) -> set[str]:
        """Every name defined anywhere in the graph."""
        return {value.name for value in self.inputs} | {node.name for node in self.nodes}

    def value(self, name: str) -> Value:
        """Look up a value by name."""
        for value in self.inputs:
            if value.name == name:
                return value
        for node in self.nodes:
            if node.name == name:
                return node.output
        raise GraphError(f"no value named {name!r}")

    def node(self, name: str) -> Node:
        """The node producing a value."""
        for candidate in self.nodes:
            if candidate.name == name:
                return candidate
        raise GraphError(f"no node produces {name!r}")

    def producer_of(self, name: str) -> Node | None:
        """The node producing a value, or nothing if it is a graph input."""
        for candidate in self.nodes:
            if candidate.name == name:
                return candidate
        return None

    def consumers_of(self, name: str) -> list[Node]:
        """Every node reading a value."""
        return [node for node in self.nodes if name in node.inputs]

    def use_counts(self) -> dict[str, int]:
        """How many times each value is read.

        Counted per argument position rather than per node, so a node squaring a value by
        multiplying it with itself counts as two uses. Fusion and buffer reuse both need the
        distinction: a value used twice cannot have its storage written over by the op
        reading it.
        """
        counts = dict.fromkeys(self.value_names, 0)
        for node in self.nodes:
            for name in node.inputs:
                counts[name] = counts.get(name, 0) + 1
        for name in self.outputs:
            counts[name] = counts.get(name, 0) + 1
        return counts

    def statistics(self) -> ops.OpStats:
        """A tally of what the graph is made of."""
        stats = ops.OpStats()
        for node in self.nodes:
            stats.add(node.op)
        return stats

    def clone(self) -> Graph:
        """A copy that can be rewritten without touching this one."""
        return Graph(
            nodes=[
                Node(op=node.op, inputs=node.inputs, output=node.output, attrs=dict(node.attrs))
                for node in self.nodes
            ],
            inputs=list(self.inputs),
            outputs=list(self.outputs),
        )

    def with_nodes(self, nodes: Sequence[Node]) -> Graph:
        """The same graph with a different node list."""
        return Graph(nodes=list(nodes), inputs=list(self.inputs), outputs=list(self.outputs))

    def __str__(self) -> str:
        header = ", ".join(str(value) for value in self.inputs)
        body = "\n".join(f"  {node}" for node in self.nodes)
        return f"graph({header}) {{\n{body}\n  return {', '.join(self.outputs)}\n}}"


def validate(graph: Graph) -> None:
    """Check every invariant a pass is allowed to rely on.

    Run after every transformation in the test suite rather than only at the end. A pass that
    produces a graph which is subtly wrong is much easier to find when it is the pass that
    fails than when the failure surfaces during code generation four passes later.
    """
    defined: set[str] = set()
    for value in graph.inputs:
        if value.name in defined:
            raise GraphError(f"input {value.name!r} is declared twice")
        defined.add(value.name)

    for node in graph.nodes:
        for name in node.inputs:
            if name not in defined:
                raise GraphError(
                    f"{node.name} reads {name!r} before it is defined, "
                    "so the node list is not in topological order"
                )
        if node.name in defined:
            raise GraphError(f"{node.name!r} is assigned twice")
        defined.add(node.name)

    if not graph.outputs:
        raise GraphError("a graph with no outputs computes nothing")
    for name in graph.outputs:
        if name not in defined:
            raise GraphError(f"output {name!r} is not defined")


def is_valid(graph: Graph) -> bool:
    """Whether the graph passes validation."""
    try:
        validate(graph)
    except GraphError:
        return False
    return True


def infer_output(op: ops.Op, operands: Sequence[Value], attrs: dict, name: str) -> Value:
    """The value an operation produces, given what it reads.

    One place, so that the frontend, constant folding and the verifier cannot disagree about
    what a graph means. Every pass that builds a node goes through here.
    """
    if op.category == ops.ELEMENTWISE:
        return _infer_elementwise(op, operands, attrs, name)
    if op.category == ops.REDUCTION:
        return _infer_reduction(op, operands, attrs, name)
    if op is ops.MATMUL:
        left, right = operands
        return Value(
            name=name,
            shape=matmul_shape(left.shape, right.shape),
            dtype=promote_all([left.dtype, right.dtype]),
        )
    if op.category == ops.VIEW:
        return _infer_view(op, operands, attrs, name)
    if op.category == ops.SIDE_EFFECT:
        return Value(name=name, shape=operands[0].shape, dtype=operands[0].dtype)
    raise TypeInferenceError(f"no inference rule for {op}")


def _infer_elementwise(op: ops.Op, operands: Sequence[Value], attrs: dict, name: str) -> Value:
    """Output type of an elementwise operation."""
    if not operands:
        raise TypeInferenceError(f"{op} needs at least one input")
    result_shape = broadcast_all([operand.shape for operand in operands])
    if op is ops.CAST:
        target = attrs.get("dtype")
        if not isinstance(target, DType):
            raise TypeInferenceError("a cast needs a target dtype")
        return Value(name=name, shape=result_shape, dtype=target)
    dtype = promote_all([operand.dtype for operand in operands])
    if op in (ops.EXP, ops.LOG, ops.SQRT, ops.TANH, ops.SIGMOID, ops.RECIPROCAL, ops.DIV):
        dtype = dtype if dtype.is_float else FLOAT32
    return Value(name=name, shape=result_shape, dtype=dtype)


def _infer_reduction(op: ops.Op, operands: Sequence[Value], attrs: dict, name: str) -> Value:
    """Output type of a reduction.

    The accumulator widening is deliberate and visible. Summing float16 in float16 stalls once
    the running total is large enough that each addend falls below its last bit, and a
    compiler that quietly keeps the narrow type produces a graph whose answer depends on how
    many elements it happened to reduce.
    """
    source = operands[0]
    axes = attrs.get("axes")
    if axes is None:
        raise TypeInferenceError(f"{op} needs axes")
    keepdims = bool(attrs.get("keepdims", False))
    result_shape = reduce_shape(source.shape, axes, keepdims=keepdims)
    if op is ops.MAX:
        return Value(name=name, shape=result_shape, dtype=source.dtype)
    return Value(name=name, shape=result_shape, dtype=accumulator_for(source.dtype))


def _infer_view(op: ops.Op, operands: Sequence[Value], attrs: dict, name: str) -> Value:
    """Output type of a shape changing operation."""
    source = operands[0]
    if op is ops.RESHAPE:
        sizes = attrs.get("sizes")
        if sizes is None:
            raise TypeInferenceError("a reshape needs target sizes")
        return Value(name=name, shape=reshape_shape(source.shape, sizes), dtype=source.dtype)
    if op is ops.TRANSPOSE:
        permutation = attrs.get("permutation")
        if permutation is None:
            raise TypeInferenceError("a transpose needs a permutation")
        return Value(
            name=name, shape=transpose_shape(source.shape, permutation), dtype=source.dtype
        )
    if op is ops.BROADCAST_TO:
        target = attrs.get("shape")
        if not isinstance(target, Shape):
            raise TypeInferenceError("a broadcast needs a target shape")
        return Value(name=name, shape=broadcast_all([source.shape, target]), dtype=source.dtype)
    raise TypeInferenceError(f"no view rule for {op}")


def topological_order(graph: Graph) -> list[Node]:
    """The nodes in an order where every reader follows its producer.

    Deterministic. Two runs of the compiler on the same graph must produce the same order,
    because the order decides the peak memory and an order that varies between runs makes a
    memory regression impossible to bisect. Ties are broken by the position the node already
    held.
    """
    ready = {value.name for value in graph.inputs}
    remaining = list(enumerate(graph.nodes))
    ordered: list[Node] = []

    while remaining:
        progressed = False
        for position, node in list(remaining):
            if all(name in ready for name in node.inputs):
                ordered.append(node)
                ready.add(node.name)
                remaining.remove((position, node))
                progressed = True
                break
        if not progressed:
            stuck = [node.name for _, node in remaining]
            raise GraphError(
                f"these nodes never become ready, so the graph has a cycle: {stuck}"
            )
    return ordered


def reachable_from_outputs(graph: Graph) -> set[str]:
    """Every value an output depends on."""
    wanted = set(graph.outputs)
    changed = True
    while changed:
        changed = False
        for node in reversed(graph.nodes):
            if node.name in wanted:
                for name in node.inputs:
                    if name not in wanted:
                        wanted.add(name)
                        changed = True
    return wanted


def boolean_output(name: str, source: Value) -> Value:
    """A predicate shaped like its input."""
    return Value(name=name, shape=source.shape, dtype=BOOL)
