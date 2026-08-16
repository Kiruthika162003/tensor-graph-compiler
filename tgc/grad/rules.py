from __future__ import annotations

from collections.abc import Callable

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node
from tgc.ir.shape import Shape

# The local derivative of every operation, written once.
#
# A rule takes the cotangent flowing back into a node and returns one cotangent per input. It
# emits into the same builder the forward graph was rebuilt into, so a gradient is an ordinary
# graph that every pass in the compiler can then run over. That is the whole point of doing
# this at the IR level rather than in an interpreter: the backward pass gets constant folding,
# fusion and buffer planning for free, and it needs them more than the forward pass does
# because it is between one and a half and four times the size of the graph it came from.
#
# Two things in here are worth reading rather than skimming.
#
# The first is that a rule for a binary op cannot just return an expression. Elementwise ops
# broadcast, so an operand of shape [8, 1] against one of shape [8, 32] produces a cotangent of
# shape [8, 32] that has to be summed back down before it can be added to the operand's
# gradient. Forgetting that is the single most common way a hand written backward pass is
# wrong, and it is wrong silently, because the shapes only disagree at the point where the
# gradient is finally used.
#
# The second is that every operation with a corner differentiates into an indicator. relu, abs,
# maximum and the max reduction all need a value that is one on one side of a boundary and zero
# on the other, and the forward op set had no such operation because nobody writing a model
# writes one. So ir/op.py grew a step. Reverse mode does not stay inside the operations the
# forward graph used, and an IR designed only around what a user writes will not hold its own
# gradients.

GradRule = Callable[[Builder, "Context"], list[str | None]]


class Context:
    """Everything a rule needs about the node it is differentiating.

    Carries the rebuilt names rather than the original ones. A rule that reached back into the
    original graph would be reading values that may not exist in the graph being built.
    """

    def __init__(
        self,
        node: Node,
        cotangent: str,
        operands: list[str],
        output: str,
        shapes: list[Shape],
    ) -> None:
        if len(operands) != len(shapes):
            raise PassError(
                f"{node.name} has {len(operands)} operands and {len(shapes)} shapes"
            )
        self.node = node
        self.cotangent = cotangent
        self.operands = operands
        self.output = output
        self.shapes = shapes

    @property
    def attrs(self) -> dict:
        """The forward node's attributes."""
        return self.node.attrs

    def operand(self, index: int) -> str:
        """One rebuilt operand by position."""
        if index >= len(self.operands):
            raise PassError(f"{self.node.name} has no operand {index}")
        return self.operands[index]

    def shape(self, index: int) -> Shape:
        """The shape of one operand."""
        if index >= len(self.shapes):
            raise PassError(f"{self.node.name} has no operand {index}")
        return self.shapes[index]


def static_sizes(shape: Shape) -> list[int]:
    """The sizes of a shape as numbers, refusing symbolic ones.

    Differentiation needs to sum a cotangent back to an operand's shape, and doing that needs
    to know which axes were broadcast. A named dimension does not say whether it was one.
    """
    sizes = []
    for size in shape.sizes:
        if not size.is_static:
            raise PassError(f"cannot differentiate through the symbolic dimension {size.name}")
        sizes.append(size.value)
    return sizes


def sum_to_shape(builder: Builder, name: str, source: Shape, target: Shape) -> str:
    """Reduce a cotangent back to the shape of the operand it belongs to.

    The inverse of broadcasting, and the part of a backward pass that is easiest to leave out.
    Broadcasting copies a value across an axis, so the derivative sums across it: an operand of
    shape [8, 1] read against one of shape [8, 32] receives thirty two contributions and its
    gradient is their sum, not any one of them.
    """
    source_sizes = static_sizes(source)
    target_sizes = static_sizes(target)
    if source_sizes == target_sizes:
        return name

    extra = len(source_sizes) - len(target_sizes)
    if extra < 0:
        raise PassError(
            f"a cotangent of rank {len(source_sizes)} cannot fit rank {len(target_sizes)}"
        )

    axes = list(range(extra))
    aligned = target_sizes if extra == 0 else [1] * extra + target_sizes
    for position, (have, want) in enumerate(zip(source_sizes, aligned, strict=True)):
        if position < extra:
            continue
        if want == 1 and have != 1:
            axes.append(position)
        elif want != have:
            raise PassError(f"a cotangent of {source_sizes} does not reduce to {target_sizes}")

    reduced = builder.sum(name, axes=axes, keepdims=True) if axes else name
    if static_sizes(_shape_of(builder, reduced)) == target_sizes:
        return reduced
    return builder.reshape(reduced, target_sizes)


def _shape_of(builder: Builder, name: str) -> Shape:
    """The shape a builder gave a value it emitted."""
    return builder.shape_of(name)


def _grad_add(builder: Builder, context: Context) -> list[str | None]:
    grad = context.cotangent
    out = _shape_of(builder, grad)
    return [
        sum_to_shape(builder, grad, out, context.shape(0)),
        sum_to_shape(builder, grad, out, context.shape(1)),
    ]


def _grad_sub(builder: Builder, context: Context) -> list[str | None]:
    grad = context.cotangent
    out = _shape_of(builder, grad)
    negated = builder.neg(grad)
    return [
        sum_to_shape(builder, grad, out, context.shape(0)),
        sum_to_shape(builder, negated, out, context.shape(1)),
    ]


def _grad_mul(builder: Builder, context: Context) -> list[str | None]:
    left = builder.mul(context.cotangent, context.operand(1))
    right = builder.mul(context.cotangent, context.operand(0))
    return [
        sum_to_shape(builder, left, _shape_of(builder, left), context.shape(0)),
        sum_to_shape(builder, right, _shape_of(builder, right), context.shape(1)),
    ]


def _grad_div(builder: Builder, context: Context) -> list[str | None]:
    # The second one is written through the forward output rather than as a square. It is one
    # multiply cheaper and, more usefully, it reuses a value the forward pass already has.
    left = builder.div(context.cotangent, context.operand(1))
    scaled = builder.mul(context.cotangent, context.output)
    right = builder.neg(builder.div(scaled, context.operand(1)))
    return [
        sum_to_shape(builder, left, _shape_of(builder, left), context.shape(0)),
        sum_to_shape(builder, right, _shape_of(builder, right), context.shape(1)),
    ]


def _grad_neg(builder: Builder, context: Context) -> list[str | None]:
    return [builder.neg(context.cotangent)]


def _grad_exp(builder: Builder, context: Context) -> list[str | None]:
    return [builder.mul(context.cotangent, context.output)]


def _grad_log(builder: Builder, context: Context) -> list[str | None]:
    return [builder.div(context.cotangent, context.operand(0))]


def _grad_sqrt(builder: Builder, context: Context) -> list[str | None]:
    half = builder.constant(0.5, dtype=context.node.output.dtype)
    return [builder.mul(builder.div(context.cotangent, context.output), half)]


def _grad_reciprocal(builder: Builder, context: Context) -> list[str | None]:
    square = builder.mul(context.output, context.output)
    return [builder.neg(builder.mul(context.cotangent, square))]


def _grad_tanh(builder: Builder, context: Context) -> list[str | None]:
    one = builder.constant(1.0, dtype=context.node.output.dtype)
    square = builder.mul(context.output, context.output)
    return [builder.mul(context.cotangent, builder.sub(one, square))]


def _grad_sigmoid(builder: Builder, context: Context) -> list[str | None]:
    one = builder.constant(1.0, dtype=context.node.output.dtype)
    complement = builder.sub(one, context.output)
    return [builder.mul(context.cotangent, builder.mul(context.output, complement))]


def _grad_relu(builder: Builder, context: Context) -> list[str | None]:
    # step is zero at zero, which is what torch does with a relu gradient at the corner. It is
    # a choice rather than a derivative, and both ends of the comparison make the same one.
    return [builder.mul(context.cotangent, builder.step(context.operand(0)))]


def _grad_abs(builder: Builder, context: Context) -> list[str | None]:
    # step(x) - step(-x) is the sign, including the zero at zero that a plain division by the
    # absolute value would turn into a nan.
    positive = builder.step(context.operand(0))
    negative = builder.step(builder.neg(context.operand(0)))
    return [builder.mul(context.cotangent, builder.sub(positive, negative))]


def _grad_maximum(builder: Builder, context: Context) -> list[str | None]:
    one = builder.constant(1.0, dtype=context.node.output.dtype)
    picked = builder.step(builder.sub(context.operand(0), context.operand(1)))
    left = builder.mul(context.cotangent, picked)
    right = builder.mul(context.cotangent, builder.sub(one, picked))
    return [
        sum_to_shape(builder, left, _shape_of(builder, left), context.shape(0)),
        sum_to_shape(builder, right, _shape_of(builder, right), context.shape(1)),
    ]


def _grad_minimum(builder: Builder, context: Context) -> list[str | None]:
    one = builder.constant(1.0, dtype=context.node.output.dtype)
    picked = builder.step(builder.sub(context.operand(1), context.operand(0)))
    left = builder.mul(context.cotangent, picked)
    right = builder.mul(context.cotangent, builder.sub(one, picked))
    return [
        sum_to_shape(builder, left, _shape_of(builder, left), context.shape(0)),
        sum_to_shape(builder, right, _shape_of(builder, right), context.shape(1)),
    ]


def _restore_reduced_axes(context: Context) -> tuple[list[int], list[int]]:
    """The axes a reduction collapsed and the shape to broadcast the cotangent back through."""
    axes = sorted(int(axis) for axis in context.attrs["axes"])
    sizes = static_sizes(context.shape(0))
    keepdims = bool(context.attrs.get("keepdims", False))
    intermediate = list(sizes)
    for axis in axes:
        intermediate[axis] = 1
    return (intermediate if not keepdims else [], sizes)


def _grad_sum(builder: Builder, context: Context) -> list[str | None]:
    intermediate, sizes = _restore_reduced_axes(context)
    grad = context.cotangent
    if intermediate:
        grad = builder.reshape(grad, intermediate)
    return [builder.broadcast_to(grad, sizes)]


def _grad_mean(builder: Builder, context: Context) -> list[str | None]:
    intermediate, sizes = _restore_reduced_axes(context)
    axes = sorted(int(axis) for axis in context.attrs["axes"])
    divisor = 1
    for axis in axes:
        divisor *= sizes[axis]
    grad = context.cotangent
    if intermediate:
        grad = builder.reshape(grad, intermediate)
    scale = builder.constant(1.0 / divisor, dtype=context.node.output.dtype)
    return [builder.broadcast_to(builder.mul(grad, scale), sizes)]


def _grad_max(builder: Builder, context: Context) -> list[str | None]:
    # One minus the step of the gap, so every position that reached the maximum gets a one.
    # torch spreads the gradient evenly across ties and this does not, which is a real
    # divergence and one that no random input will ever produce.
    intermediate, sizes = _restore_reduced_axes(context)
    grad = context.cotangent
    peak = context.output
    if intermediate:
        grad = builder.reshape(grad, intermediate)
        peak = builder.reshape(peak, intermediate)
    one = builder.constant(1.0, dtype=context.node.output.dtype)
    wide_peak = builder.broadcast_to(peak, sizes)
    gap = builder.sub(wide_peak, context.operand(0))
    mask = builder.sub(one, builder.step(gap))
    return [builder.mul(builder.broadcast_to(grad, sizes), mask)]


def _grad_matmul(builder: Builder, context: Context) -> list[str | None]:
    left_shape = static_sizes(context.shape(0))
    right_shape = static_sizes(context.shape(1))
    if len(left_shape) != 2 or len(right_shape) != 2:
        raise PassError(
            f"the matmul rule handles rank two, got {len(left_shape)} and {len(right_shape)}"
        )
    right_t = builder.transpose(context.operand(1), [1, 0])
    left_t = builder.transpose(context.operand(0), [1, 0])
    return [
        builder.matmul(context.cotangent, right_t),
        builder.matmul(left_t, context.cotangent),
    ]


def _grad_transpose(builder: Builder, context: Context) -> list[str | None]:
    permutation = [int(axis) for axis in context.attrs["permutation"]]
    inverse = [0] * len(permutation)
    for position, axis in enumerate(permutation):
        inverse[axis] = position
    return [builder.transpose(context.cotangent, inverse)]


def _grad_reshape(builder: Builder, context: Context) -> list[str | None]:
    return [builder.reshape(context.cotangent, static_sizes(context.shape(0)))]


def _grad_broadcast(builder: Builder, context: Context) -> list[str | None]:
    grad = context.cotangent
    return [sum_to_shape(builder, grad, _shape_of(builder, grad), context.shape(0))]


def _grad_concat(builder: Builder, context: Context) -> list[str | None]:
    # A join splits going backwards. Each operand receives the window of the cotangent that
    # came from it, which is the only rule in this file that is a pure bookkeeping operation
    # with no arithmetic in it at all.
    axis = int(context.attrs["axis"])
    left = static_sizes(context.shape(0))[axis]
    right = static_sizes(context.shape(1))[axis]
    return [
        builder.slice(context.cotangent, axis, 0, left),
        builder.slice(context.cotangent, axis, left, right),
    ]


def _grad_slice(builder: Builder, context: Context) -> list[str | None]:
    # A window pads going backwards. Everything outside the window contributed nothing, so its
    # gradient is zero, and the zeros have to be built rather than assumed because the operand
    # is larger than the cotangent and nothing else in the graph has its shape.
    axis = int(context.attrs["axis"])
    start = int(context.attrs["start"])
    length = int(context.attrs["length"])
    sizes = static_sizes(context.shape(0))
    dtype = context.node.output.dtype

    current = context.cotangent
    if start:
        before = list(sizes)
        before[axis] = start
        zeros = builder.broadcast_to(builder.constant(0.0, dtype=dtype), before)
        current = builder.concat(zeros, current, axis)
    trailing = sizes[axis] - start - length
    if trailing:
        after = list(sizes)
        after[axis] = trailing
        zeros = builder.broadcast_to(builder.constant(0.0, dtype=dtype), after)
        current = builder.concat(current, zeros, axis)
    return [current]


def _grad_identity(_builder: Builder, context: Context) -> list[str | None]:
    return [context.cotangent]


RULES: dict[str, GradRule] = {
    "add": _grad_add,
    "sub": _grad_sub,
    "mul": _grad_mul,
    "div": _grad_div,
    "neg": _grad_neg,
    "exp": _grad_exp,
    "log": _grad_log,
    "sqrt": _grad_sqrt,
    "reciprocal": _grad_reciprocal,
    "tanh": _grad_tanh,
    "sigmoid": _grad_sigmoid,
    "relu": _grad_relu,
    "abs": _grad_abs,
    "maximum": _grad_maximum,
    "minimum": _grad_minimum,
    "sum": _grad_sum,
    "mean": _grad_mean,
    "max": _grad_max,
    "matmul": _grad_matmul,
    "transpose": _grad_transpose,
    "reshape": _grad_reshape,
    "broadcast_to": _grad_broadcast,
    "concat": _grad_concat,
    "slice": _grad_slice,
    "print": _grad_identity,
    "assert_finite": _grad_identity,
}

# Operations a gradient stops at rather than passes through. A leaf has nothing behind it. step
# is a derivative rather than a function anyone differentiates: it is flat everywhere it is
# defined and undefined at the one point that matters, so a rule for it would be a lie. cast is
# left out for a different reason, that its honest gradient is a cast back and a cast down
# followed by a cast up is not the identity, so the rule would quietly lose the low bits of
# every gradient that passed through it.
NON_DIFFERENTIABLE = {"input", "constant", "step", "cast"}


def has_rule(op: ops.Op) -> bool:
    """Whether an operation can be differentiated."""
    return op.name in RULES


def rule_for(op: ops.Op) -> GradRule:
    """The rule for one operation."""
    if op.name not in RULES:
        raise PassError(f"no gradient rule for {op.name}")
    return RULES[op.name]


def coverage() -> dict:
    """Which operations have a rule and which do not.

    Worth reporting rather than assuming. An op added to the forward set without a rule turns
    into a runtime failure the first time somebody differentiates a graph containing it, which
    is much later than it needs to be.
    """
    covered = [op.name for op in ops.ALL_OPS if op.name in RULES]
    missing = [
        op.name
        for op in ops.ALL_OPS
        if op.name not in RULES and op.name not in NON_DIFFERENTIABLE
    ]
    return {
        "ops": len(ops.ALL_OPS),
        "with_rules": len(covered),
        "deliberately_without": len(NON_DIFFERENTIABLE),
        "missing": sorted(missing),
    }


def differentiable_nodes(graph: Graph) -> list[str]:
    """Every node in a graph that a gradient can pass through."""
    return [node.name for node in graph.nodes if has_rule(node.op)]


def blocking_nodes(graph: Graph) -> list[str]:
    """Every node that stops a gradient."""
    return [
        node.name
        for node in graph.nodes
        if not has_rule(node.op) and node.op.name not in ("input", "constant")
    ]


def check_differentiable(graph: Graph) -> None:
    """Raise if a graph holds an operation with no rule."""
    blocking = blocking_nodes(graph)
    if blocking:
        names = sorted({graph.node(name).op.name for name in blocking})
        raise ConfigError(f"these operations have no gradient rule: {names}")
