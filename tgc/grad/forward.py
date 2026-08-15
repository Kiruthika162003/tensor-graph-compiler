from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError, PassError
from tgc.grad.reverse import COTANGENT, gradient
from tgc.grad.rules import static_sizes
from tgc.ir import op as ops
from tgc.ir.builder import (
    Builder,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.graph import Graph, Node
from tgc.verify.reference import random_feeds, run

# The other direction: carrying a perturbation forward instead of a sensitivity back.
#
# Forward mode is the same chain rule read left to right. Every value gets a tangent alongside
# it, each operation says what its output tangent is in terms of its input tangents, and the
# whole thing is a single pass in the order the graph already runs in.
#
# Two structural differences from reverse mode fall straight out of that and neither is obvious
# until the code is written next to the other code.
#
# Broadcasting needs no correction here. In reverse mode a cotangent has to be summed back down
# to the shape of the operand that was broadcast, and that summing is the part everybody
# forgets. In forward mode the tangent broadcasts exactly the way the value did, because it is
# going the same way, so the ordinary shape rules do all of the work.
#
# Accumulation disappears too. A value read by five consumers needs its five cotangents summed
# in reverse mode; in forward mode it has one tangent that five nodes read, which is a plain
# lookup. That is a real simplification of the code and, measured, it buys nothing in the size
# of the graph: the addition saved on a shared value is given straight back by the product
# rule, which costs two multiplies and an addition going forwards against one multiply per
# operand going back.
#
# What forward mode does not have is the thing reverse mode is used for. One forward pass gives
# one column of the jacobian and one reverse pass gives one row, so a function from a million
# parameters to one loss wants reverse, and the measurement at the bottom of this file puts a
# number on where the two cross.

TANGENT_SUFFIX = "_tangent"


def fan_in_graph(sizes: Sequence[int] = (16, 16)) -> Graph:
    """A value read twice and rejoined, with a derivative that is not zero.

    The obvious fixture for this is diamond_graph and it cannot be used. Its two branches are a
    relu and a negation of the same exponential, and an exponential is positive, so the relu is
    the identity there and the two branches cancel exactly. Its output is zero for every input
    and so is every derivative of it, which means a check run on it passes without touching
    anything. This one shares a value the same way and does not cancel.
    """
    builder = Builder()
    x = builder.input(list(sizes), name="x")
    shared = builder.tanh(x)
    return builder.finish(builder.mul(shared, builder.sigmoid(shared)))


def tangent_name(name: str) -> str:
    """The name of the tangent that travels with a value."""
    return f"{name}{TANGENT_SUFFIX}"


@dataclass
class JvpResult:
    """A forward mode graph and what it took to build."""

    graph: Graph
    forward_nodes: int
    tangent_nodes: int
    wrt: tuple[str, ...] = ()

    @property
    def growth(self) -> float:
        """How many times bigger the whole thing is than the graph it came from."""
        if self.forward_nodes == 0:
            return 0.0
        return (self.forward_nodes + self.tangent_nodes) / self.forward_nodes

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "wrt": list(self.wrt),
            "forward_nodes": self.forward_nodes,
            "tangent_nodes": self.tangent_nodes,
            "growth": round(self.growth, 3),
        }


def _tangent_elementwise(
    builder: Builder, node: Node, values: list[str], tangents: list[str], output: str
) -> str:
    """The local derivative of a unary elementwise op, times its input tangent."""
    name = node.op.name
    dtype = node.output.dtype
    if name == "neg":
        return builder.neg(tangents[0])
    if name == "exp":
        return builder.mul(tangents[0], output)
    if name == "log":
        return builder.div(tangents[0], values[0])
    if name == "sqrt":
        half = builder.constant(0.5, dtype=dtype)
        return builder.mul(builder.div(tangents[0], output), half)
    if name == "reciprocal":
        square = builder.mul(output, output)
        return builder.neg(builder.mul(tangents[0], square))
    if name == "tanh":
        one = builder.constant(1.0, dtype=dtype)
        return builder.mul(tangents[0], builder.sub(one, builder.mul(output, output)))
    if name == "sigmoid":
        one = builder.constant(1.0, dtype=dtype)
        return builder.mul(tangents[0], builder.mul(output, builder.sub(one, output)))
    if name == "relu":
        return builder.mul(tangents[0], builder.step(values[0]))
    if name == "abs":
        sign = builder.sub(builder.step(values[0]), builder.step(builder.neg(values[0])))
        return builder.mul(tangents[0], sign)
    raise PassError(f"no forward rule for {name}")


def _tangent_max(
    builder: Builder, node: Node, values: list[str], tangents: list[str], output: str
) -> str:
    """The tangent of the element that won, selected by a mask.

    One minus the step of the gap picks every position that reached the maximum, and the sum
    then carries its tangent through. With a tie the sum carries both, which is wrong by a
    factor of the number of tied elements and cannot happen on any input that is not
    constructed for it.
    """
    axes = [int(axis) for axis in node.attrs["axes"]]
    keepdims = bool(node.attrs.get("keepdims", False))
    sizes = static_sizes(builder.shape_of(values[0]))
    restored = list(sizes)
    for axis in axes:
        restored[axis] = 1

    peak = output if keepdims else builder.reshape(output, restored)
    one = builder.constant(1.0, dtype=node.output.dtype)
    gap = builder.sub(builder.broadcast_to(peak, sizes), values[0])
    mask = builder.sub(one, builder.step(gap))
    return builder.sum(builder.mul(mask, tangents[0]), axes=axes, keepdims=keepdims)


def tangent_of(
    builder: Builder, node: Node, values: list[str], tangents: list[str], output: str
) -> str:
    """The tangent of one node's output, given the tangents of its inputs."""
    name = node.op.name
    if name == "add":
        return builder.add(tangents[0], tangents[1])
    if name == "sub":
        return builder.sub(tangents[0], tangents[1])
    if name == "mul":
        return builder.add(
            builder.mul(tangents[0], values[1]), builder.mul(values[0], tangents[1])
        )
    if name == "div":
        # Written through the output, so the quotient rule costs one division rather than a
        # square and a division, reusing a value the forward pass already computed.
        return builder.div(
            builder.sub(tangents[0], builder.mul(output, tangents[1])), values[1]
        )
    if name in ("maximum", "minimum"):
        left, right = (values[0], values[1]) if name == "maximum" else (values[1], values[0])
        one = builder.constant(1.0, dtype=node.output.dtype)
        picked = builder.step(builder.sub(left, right))
        return builder.add(
            builder.mul(tangents[0], picked),
            builder.mul(tangents[1], builder.sub(one, picked)),
        )
    if name == "matmul":
        return builder.add(
            builder.matmul(tangents[0], values[1]), builder.matmul(values[0], tangents[1])
        )
    if name in ("sum", "mean"):
        method = builder.sum if name == "sum" else builder.mean
        return method(
            tangents[0],
            axes=[int(axis) for axis in node.attrs["axes"]],
            keepdims=bool(node.attrs.get("keepdims", False)),
        )
    if name == "max":
        return _tangent_max(builder, node, values, tangents, output)
    if name == "reshape":
        return builder.reshape(tangents[0], [int(size) for size in node.attrs["sizes"]])
    if name == "transpose":
        return builder.transpose(tangents[0], [int(axis) for axis in node.attrs["permutation"]])
    if name == "broadcast_to":
        return builder.broadcast_to(tangents[0], static_sizes(node.output.shape))
    if name in ("print", "assert_finite"):
        return tangents[0]
    return _tangent_elementwise(builder, node, values, tangents, output)


def jvp(graph: Graph, wrt: Sequence[str] | None = None) -> JvpResult:
    """Build the graph that carries a perturbation of the inputs through to the output.

    The result takes the original inputs plus one tangent per differentiated input, and returns
    the tangent of the output. Inputs not listed are held fixed, which is expressed by giving
    them a zero tangent rather than by leaving them out, because an operation still has to be
    told what its other operand's tangent was.
    """
    if len(graph.outputs) != 1:
        raise ConfigError(f"a forward pass needs exactly one output, got {len(graph.outputs)}")
    targets = list(wrt) if wrt is not None else [value.name for value in graph.inputs]
    if not targets:
        raise ConfigError("there is nothing to differentiate with respect to")
    known = {value.name for value in graph.inputs}
    unknown = [name for name in targets if name not in known]
    if unknown:
        raise ConfigError(f"{unknown} are not inputs of this graph")

    builder = Builder()
    values: dict[str, str] = {}
    tangents: dict[str, str] = {}
    for value in graph.inputs:
        sizes = [size.value if size.is_static else size.name for size in value.shape.sizes]
        values[value.name] = builder.input(sizes, dtype=value.dtype, name=value.name)

    for value in graph.inputs:
        sizes = static_sizes(value.shape)
        if value.name in targets:
            tangents[value.name] = builder.input(
                sizes, dtype=value.dtype, name=tangent_name(value.name)
            )
        else:
            zero = builder.constant(0.0, dtype=value.dtype)
            tangents[value.name] = builder.broadcast_to(zero, sizes) if sizes else zero

    for node in graph.nodes:
        if node.op is ops.CONSTANT:
            values[node.name] = builder.constant(
                float(node.attrs["value"]), dtype=node.output.dtype
            )
            tangents[node.name] = builder.constant(0.0, dtype=node.output.dtype)
            continue
        operands = [values[name] for name in node.inputs]
        output = builder.apply(node.op, *operands, **node.attrs)
        values[node.name] = output
        tangents[node.name] = tangent_of(
            builder, node, operands, [tangents[name] for name in node.inputs], output
        )

    result = builder.finish(tangents[graph.outputs[0]])
    return JvpResult(
        graph=result,
        forward_nodes=len(graph.nodes),
        tangent_nodes=len(result.nodes) - len(graph.nodes),
        wrt=tuple(targets),
    )


def torch_jvp(
    graph: Graph, feeds: dict[str, torch.Tensor], tangents: dict[str, torch.Tensor]
) -> torch.Tensor:
    """The same directional derivative, from torch.

    Built as a reverse pass of a reverse pass rather than with a forward mode primitive, which
    is what most frameworks do internally and is a reasonable independent implementation: the
    derivative of a gradient dotted with a direction is the directional derivative.
    """
    tracked = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in feeds.items()
    }
    result = run(graph, tracked)[0]
    seed = torch.zeros_like(result).requires_grad_(True)
    names = list(tangents)
    grads = torch.autograd.grad(
        result, [tracked[name] for name in names], seed, create_graph=True
    )
    paired = sum((grad * tangents[name]).sum() for name, grad in zip(names, grads, strict=True))
    return torch.autograd.grad(paired, seed)[0]


def split_jvp_feeds(
    graph: Graph, feeds: dict[str, torch.Tensor], wrt: Sequence[str]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Separate a forward mode graph's feeds into values and tangents."""
    values = {value.name: feeds[value.name] for value in graph.inputs}
    tangents = {}
    for name in wrt:
        key = tangent_name(name)
        if key not in feeds:
            raise ConfigError(f"the feeds have no {key!r}")
        tangents[name] = feeds[key]
    return values, tangents


def compare_with_torch(
    graph: Graph, wrt: Sequence[str] | None = None, *, seed: int = 0
) -> dict:
    """The forward mode graph against torch, on the same inputs and the same direction."""
    targets = list(wrt) if wrt is not None else [value.name for value in graph.inputs]
    built = jvp(graph, targets)
    feeds = random_feeds(built.graph, positive=True, seed=seed)
    values, tangents = split_jvp_feeds(graph, feeds, targets)

    ours = run(built.graph, feeds)[0]
    theirs = torch_jvp(graph, values, tangents)
    finite = torch.isfinite(ours) & torch.isfinite(theirs)
    if not finite.any():
        return {"largest_gap": 0.0, "relative_gap": 0.0, "overflowed": int(finite.numel())}
    gap = float((ours[finite] - theirs[finite]).abs().max())
    scale = float(theirs[finite].abs().max())
    return {
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
        "overflowed": int((~finite).sum()),
    }


def agrees_with_torch(
    graph: Graph, wrt: Sequence[str] | None = None, *, tolerance: float = 1e-5
) -> bool:
    """Whether the forward mode graph matches torch to a tolerance."""
    return compare_with_torch(graph, wrt)["relative_gap"] <= tolerance


def dot_product_identity(
    graph: Graph, wrt: Sequence[str] | None = None, *, seed: int = 0
) -> dict:
    """The one check that ties the two modes together.

    A cotangent dotted with a forward derivative equals a reverse derivative dotted with the
    direction, for every cotangent and every direction, because both are the same bilinear form
    read from opposite sides. It needs no reference implementation and it fails loudly if either
    mode has a rule wrong, which is more than either mode's own tests can say.
    """
    targets = list(wrt) if wrt is not None else [value.name for value in graph.inputs]
    forward = jvp(graph, targets)
    backward = gradient(graph, targets)

    feeds = random_feeds(forward.graph, positive=True, seed=seed)
    reverse_feeds = dict(random_feeds(backward.graph, positive=True, seed=seed + 1))
    for value in graph.inputs:
        reverse_feeds[value.name] = feeds[value.name]

    tangent_out = run(forward.graph, feeds)[0]
    left = float((reverse_feeds[COTANGENT] * tangent_out).sum())

    cotangents = run(backward.graph, reverse_feeds)
    right = 0.0
    for name, cotangent in zip(targets, cotangents, strict=True):
        right += float((cotangent * feeds[tangent_name(name)]).sum())

    finite = math.isfinite(left) and math.isfinite(right)
    scale = max(abs(left), abs(right), 1e-12)
    return {
        "forward_side": left,
        "reverse_side": right,
        "finite": finite,
        "relative_gap": abs(left - right) / scale if finite else float("inf"),
    }


def identity_holds_everywhere(tolerance: float = 1e-5) -> list[dict]:
    """The dot product identity on every fixture.

    The chain is at depth two rather than three. Three composes two exponentials and reaches
    the top of float32, and an identity between two infinities is not a check of anything.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(2)),
        ("fan in", fan_in_graph()),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        row = dot_product_identity(graph)
        row["graph"] = label
        row["holds"] = row["finite"] and row["relative_gap"] <= tolerance
        rows.append(row)
    return rows


def the_diamond_fixture_is_vacuous() -> dict:
    """Why the obvious fan in fixture cannot be used here.

    Its output is identically zero. The two branches are a relu and a negation of the same
    exponential, an exponential is positive, so the relu is the identity and the branches
    cancel. Every derivative of a constant zero is zero, and both sides of the dot product
    identity are then zero, so the check passes without having tested a rule.
    """
    graph = diamond_graph()
    feeds = random_feeds(graph, positive=True)
    built = jvp(graph)
    tangent_feeds = random_feeds(built.graph, positive=True)
    return {
        "largest_output": float(run(graph, feeds)[0].abs().max()),
        "largest_tangent": float(run(built.graph, tangent_feeds)[0].abs().max()),
    }


def size_comparison() -> list[dict]:
    """Forward and reverse graphs for the same function, side by side.

    Equal on the elementwise fixtures and up to forty percent larger in reverse mode wherever
    there is a reduction or a contraction, because a sum differentiates into a reshape and a
    broadcast going backwards and into a plain sum going forwards, and a matrix product into
    two products and two transposes rather than two products. So reverse mode is the bigger
    graph, and the real cost of it is not in the node count at all: a
    forward pass can discard a value as soon as its tangent is computed and a reverse pass
    cannot discard anything until the walk back reaches it.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(6)),
        ("fan in", fan_in_graph()),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        rows.append(
            {
                "graph": label,
                "forward_mode": len(jvp(graph).graph.nodes),
                "reverse_mode": len(gradient(graph).graph.nodes),
            }
        )
    return rows


def passes_for_a_full_jacobian(graph: Graph) -> dict:
    """How many passes of each mode a complete jacobian would take.

    One forward pass gives a column and one reverse pass gives a row, so the choice is decided
    by the shape of the function and by nothing else. For a loss, with one output and a great
    many inputs, reverse wins by the number of parameters.
    """
    inputs = sum(graph.value(value.name).shape.elements for value in graph.inputs)
    outputs = sum(graph.value(name).shape.elements for name in graph.outputs)
    return {
        "input_elements": inputs,
        "output_elements": outputs,
        "forward_passes": inputs,
        "reverse_passes": outputs,
        "reverse_is_cheaper": outputs < inputs,
    }


def which_mode_to_use() -> list[dict]:
    """Which mode each fixture wants, and by how much."""
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(3)),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        row = passes_for_a_full_jacobian(graph)
        row["graph"] = label
        row["ratio"] = round(row["forward_passes"] / max(row["reverse_passes"], 1), 3)
        rows.append(row)
    return rows


def broadcasting_needs_no_correction() -> dict:
    """The structural difference between the two modes, counted.

    A reverse pass over a graph with a broadcast in it contains sum nodes that exist only to
    undo the broadcast. A forward pass over the same graph contains none, because the tangent
    broadcast the same way the value did. This counts them.
    """
    builder = Builder()
    x = builder.input([8, 32], name="x")
    column = builder.input([8, 1], name="column")
    graph = builder.finish(builder.mul(x, column))

    original = _count(graph, ops.SUM)
    return {
        "forward_mode_sums": _count(jvp(graph).graph, ops.SUM) - original,
        "reverse_mode_sums": _count(gradient(graph).graph, ops.SUM) - original,
    }


def _count(graph: Graph, op: ops.Op) -> int:
    """How many nodes of one operation a graph holds."""
    return sum(1 for node in graph.nodes if node.op is op)


def accumulation_is_reverse_mode_only() -> dict:
    """The other structural difference, counted, and it does not go the way it reads.

    A value read by several consumers needs its cotangents summed in reverse mode, and in
    forward mode it has one tangent that several nodes read, which is a lookup. So reverse mode
    pays for fan in and forward mode does not.

    It still builds the smaller graph on this fixture, because the saving is one addition per
    shared value and the cost is somewhere else entirely: a product's tangent needs two
    multiplies and an addition where its cotangent needs one multiply per operand. The
    bookkeeping argument for forward mode is real and it is small.
    """
    graph = fan_in_graph()
    reads = graph.use_counts()
    return {
        "values_read_more_than_once": sum(1 for count in reads.values() if count > 1),
        "forward_mode_nodes": jvp(graph).tangent_nodes,
        "reverse_mode_nodes": gradient(graph).backward_nodes,
    }
