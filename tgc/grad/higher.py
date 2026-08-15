from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError, PassError
from tgc.grad.forward import jvp, tangent_name
from tgc.grad.reverse import COTANGENT, gradient
from tgc.ir.builder import Builder, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.graph import Graph
from tgc.verify.reference import random_feeds, run

# Differentiating the gradient, and what it costs to be allowed to.
#
# The gradient of a graph is an ordinary graph, so it can be handed straight back to the
# differentiator. Doing that in forward mode gives a hessian times a vector in one pass, which
# is the only second order quantity anybody computes at scale: the hessian itself is square in
# the number of parameters and nobody wants it.
#
# The interesting part is which graphs allow it, because the answer is fewer than expected and
# for the right reason. Every operation with a corner differentiates into a step, and a step has
# no derivative worth writing: it is flat everywhere it is defined and its derivative at the one
# point that matters is a delta. So a relu network has no second derivative that this compiler
# will build, and it should not, because a relu network is piecewise linear and its second
# derivative really is zero on every piece and undefined between them.
#
# That is not a limitation to work around. A curvature method applied to a relu network is
# working with a hessian that is zero almost everywhere, and the useful thing a compiler can do
# about it is say so at compile time rather than return a tensor of zeros at runtime.

CORNERED = ("relu", "abs", "maximum", "minimum", "max", "step")


def smooth_chain(depth: int = 4, sizes: Sequence[int] = (16, 16)) -> Graph:
    """A chain with no corner in it, so it can be differentiated twice.

    Alternating tanh and sigmoid rather than relu, which is the whole point: both are smooth
    everywhere and both saturate, so the fixture also exercises a second derivative that is
    almost zero over most of its range without ever being undefined.
    """
    if depth < 1:
        raise ConfigError(f"a chain needs at least one operation, got {depth}")
    builder = Builder()
    current = builder.input(list(sizes), name="x")
    for index in range(depth):
        current = builder.sigmoid(current) if index % 2 else builder.tanh(current)
    return builder.finish(current)


def quadratic_graph(sizes: Sequence[int] = (8, 8)) -> Graph:
    """A function whose second derivative is a number rather than a shape.

    The square of an input, elementwise. Its first derivative is twice the input and its second
    is two everywhere, so a hessian vector product against a direction has to come back as twice
    that direction, exactly, in every position. Nothing else in the fixture set gives a second
    derivative that can be checked by reading it.
    """
    builder = Builder()
    x = builder.input(list(sizes), name="x")
    return builder.finish(builder.mul(x, x))


def has_a_corner(graph: Graph) -> bool:
    """Whether a graph holds an operation that differentiates into an indicator."""
    return any(node.op.name in CORNERED for node in graph.nodes)


def corner_operations(graph: Graph) -> list[str]:
    """The nodes that stop a second derivative."""
    return [node.name for node in graph.nodes if node.op.name in CORNERED]


@dataclass
class HvpResult:
    """A hessian vector product graph and what it took to build."""

    graph: Graph
    original_nodes: int
    gradient_nodes: int

    @property
    def total_nodes(self) -> int:
        """Nodes in the finished second order graph."""
        return len(self.graph.nodes)

    @property
    def growth(self) -> float:
        """How many times bigger than the graph it came from."""
        if self.original_nodes == 0:
            return 0.0
        return self.total_nodes / self.original_nodes

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "original_nodes": self.original_nodes,
            "gradient_nodes": self.gradient_nodes,
            "total_nodes": self.total_nodes,
            "growth": round(self.growth, 3),
        }


def hessian_vector_product(graph: Graph, wrt: str | None = None) -> HvpResult:
    """Build the graph that multiplies the hessian by a direction.

    Forward over reverse: differentiate once backwards to get the gradient, then once forwards
    along the direction. One pass each, and the result never materialises the hessian, which for
    any real model would be square in the number of parameters.

    Reverse over reverse computes the same thing and is written out in the next function so the
    two can be compared. On node count the two are close, which is not what the usual advice
    would lead anyone to expect.
    """
    name = wrt if wrt is not None else graph.inputs[0].name
    if has_a_corner(graph):
        raise PassError(
            f"{corner_operations(graph)} differentiate into a step, which has no derivative"
        )
    first = gradient(graph, [name])
    second = jvp(first.graph, [name])
    return HvpResult(
        graph=second.graph,
        original_nodes=len(graph.nodes),
        gradient_nodes=len(first.graph.nodes),
    )


def reverse_over_reverse(graph: Graph, wrt: str | None = None) -> HvpResult:
    """The same product, built by differentiating backwards twice.

    Kept so the two can be compared rather than argued about. It needs the gradient reduced to
    a single output first, which it does by taking the gradient with respect to one input, and
    then differentiates that with respect to the same input.
    """
    name = wrt if wrt is not None else graph.inputs[0].name
    if has_a_corner(graph):
        raise PassError(
            f"{corner_operations(graph)} differentiate into a step, which has no derivative"
        )
    first = gradient(graph, [name])
    renamed = _rename_cotangent(first.graph)
    second = gradient(renamed, [name])
    return HvpResult(
        graph=second.graph,
        original_nodes=len(graph.nodes),
        gradient_nodes=len(first.graph.nodes),
    )


def _rename_cotangent(graph: Graph) -> Graph:
    """Rebuild a gradient graph with its cotangent input under another name.

    Needed only because differentiating a gradient graph a second time would try to add a
    second input called cotangent, and a graph cannot have two inputs with one name.
    """
    builder = Builder()
    mapping: dict[str, str] = {}
    for value in graph.inputs:
        label = "seed" if value.name == COTANGENT else value.name
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=label,
        )
    for node in graph.nodes:
        if node.op.name == "constant":
            mapping[node.name] = builder.constant(
                float(node.attrs["value"]), dtype=node.output.dtype
            )
            continue
        operands = [mapping[name] for name in node.inputs]
        mapping[node.name] = builder.apply(node.op, *operands, **node.attrs)
    return builder.finish(*[mapping[name] for name in graph.outputs])


def which_graphs_admit_a_second_derivative() -> list[dict]:
    """Which fixtures can be differentiated twice, and what stops the rest.

    A relu network cannot, and neither can a softmax, because a max reduction is a corner in
    disguise. That covers most of what people actually train, which is the point: second order
    methods are applied to models whose second derivative is zero almost everywhere, and the
    compiler is in a position to say so before anything runs.
    """
    rows = []
    for label, graph in (
        ("smooth chain", smooth_chain()),
        ("quadratic", quadratic_graph()),
        ("layernorm", layernorm_graph()),
        ("softmax", softmax_graph()),
        ("mlp", mlp_graph()),
    ):
        rows.append(
            {
                "graph": label,
                "twice_differentiable": not has_a_corner(graph),
                "blocked_by": sorted(
                    {graph.node(name).op.name for name in corner_operations(graph)}
                ),
            }
        )
    return rows


def measure_quadratic(seed: int = 0) -> dict:
    """The one case where the answer can be read rather than compared.

    The square of an input has a second derivative of exactly two, so a hessian vector product
    against a direction has to be twice that direction times the seed, in every position. A rule
    that is wrong by a factor or a transpose fails this and cannot fail it quietly.
    """
    graph = quadratic_graph()
    built = hessian_vector_product(graph)
    feeds = random_feeds(built.graph, positive=True, seed=seed)
    result = run(built.graph, feeds)[0]
    expected = 2.0 * feeds[tangent_name("x")] * feeds[COTANGENT]
    return {
        "largest_gap": float((result - expected).abs().max()),
        "largest_value": float(expected.abs().max()),
    }


def torch_hvp(
    graph: Graph,
    feeds: dict[str, torch.Tensor],
    name: str,
    direction: torch.Tensor,
    cotangent: torch.Tensor,
) -> torch.Tensor:
    """The same product, from torch's double backward."""
    tracked = {
        key: tensor.clone().detach().requires_grad_(True) for key, tensor in feeds.items()
    }
    result = run(graph, tracked)[0]
    first = torch.autograd.grad(result, tracked[name], cotangent, create_graph=True)[0]
    return torch.autograd.grad((first * direction).sum(), tracked[name])[0]


def compare_with_torch(graph: Graph, wrt: str | None = None, *, seed: int = 0) -> dict:
    """The second order graph against torch's double backward."""
    name = wrt if wrt is not None else graph.inputs[0].name
    built = hessian_vector_product(graph, name)
    feeds = random_feeds(built.graph, positive=True, seed=seed)
    values = {value.name: feeds[value.name] for value in graph.inputs}

    ours = run(built.graph, feeds)[0]
    theirs = torch_hvp(graph, values, name, feeds[tangent_name(name)], feeds[COTANGENT])
    gap = float((ours - theirs).abs().max())
    scale = float(theirs.abs().max())
    return {"largest_gap": gap, "relative_gap": gap / scale if scale else gap}


def agrees_with_torch(graph: Graph, wrt: str | None = None, *, tolerance: float = 1e-4) -> bool:
    """Whether the second order graph matches torch to a tolerance."""
    return compare_with_torch(graph, wrt)["relative_gap"] <= tolerance


def symmetry_check(graph: Graph, *, seed: int = 0) -> dict:
    """Whether the hessian this builds is symmetric, which it has to be.

    A second derivative does not depend on the order the two directions were taken in, so one
    direction dotted with the hessian times another equals the other dotted with the hessian
    times the first. It needs no reference implementation and it catches a transposed rule,
    which is the mistake a comparison against a single number cannot see.
    """
    built = hessian_vector_product(graph)
    name = graph.inputs[0].name
    first = random_feeds(built.graph, positive=True, seed=seed)
    second = random_feeds(built.graph, positive=True, seed=seed + 1)
    for value in graph.inputs:
        second[value.name] = first[value.name]
    second[COTANGENT] = first[COTANGENT]

    left = float((run(built.graph, first)[0] * second[tangent_name(name)]).sum())
    right = float((run(built.graph, second)[0] * first[tangent_name(name)]).sum())
    scale = max(abs(left), abs(right), 1e-12)
    return {"one_way": left, "the_other": right, "relative_gap": abs(left - right) / scale}


def symmetry_holds_everywhere(tolerance: float = 1e-4) -> list[dict]:
    """The symmetry check on every fixture that admits a second derivative."""
    rows = []
    for label, graph in (
        ("smooth chain", smooth_chain()),
        ("quadratic", quadratic_graph()),
        ("layernorm", layernorm_graph()),
    ):
        row = symmetry_check(graph)
        row["graph"] = label
        row["symmetric"] = row["relative_gap"] <= tolerance
        rows.append(row)
    return rows


def order_of_composition() -> list[dict]:
    """Forward over reverse against reverse over reverse, by size.

    The received advice is that forward over reverse is the cheaper composition. Counted in
    nodes it is not: the two are within two percent of each other on a smooth chain, reverse
    over reverse is twelve percent smaller on a layernorm and almost half the size on a
    quadratic, where its second differentiation finds a graph that folds.

    Which does not make the advice wrong, it makes the usual justification for it wrong. The
    cost of the second reverse pass is not the nodes, it is that it has to keep the whole first
    gradient alive while it walks back over it, and a node count cannot see that. The liveness
    analysis can, and analysis/liveness.py is where that comparison belongs.
    """
    rows = []
    for label, graph in (
        ("smooth chain", smooth_chain()),
        ("quadratic", quadratic_graph()),
        ("layernorm", layernorm_graph()),
    ):
        forward = hessian_vector_product(graph)
        backward = reverse_over_reverse(graph)
        rows.append(
            {
                "graph": label,
                "forward_over_reverse": forward.total_nodes,
                "reverse_over_reverse": backward.total_nodes,
                "ratio": round(backward.total_nodes / forward.total_nodes, 3),
            }
        )
    return rows


def two_compositions_agree(tolerance: float = 1e-4) -> list[dict]:
    """Whether the two ways of composing the passes compute the same product.

    They have to, and the check is worth running because the two take their direction from
    different places: forward over reverse takes it as a tangent and reverse over reverse takes
    it as the cotangent of the second pass. A confusion between the direction and the seed
    produces a plausible tensor of the right shape.
    """
    rows = []
    for label, graph in (
        ("smooth chain", smooth_chain()),
        ("quadratic", quadratic_graph()),
        ("layernorm", layernorm_graph()),
    ):
        name = graph.inputs[0].name
        forward = hessian_vector_product(graph, name)
        backward = reverse_over_reverse(graph, name)

        feeds = random_feeds(forward.graph, positive=True)
        other = {value.name: feeds[value.name] for value in graph.inputs}
        other["seed"] = feeds[COTANGENT]
        other[COTANGENT] = feeds[tangent_name(name)]

        left = run(forward.graph, feeds)[0]
        right = run(backward.graph, other)[0]
        gap = float((left - right).abs().max())
        scale = float(left.abs().max())
        rows.append(
            {
                "graph": label,
                "relative_gap": gap / scale if scale else gap,
                "agree": (gap / scale if scale else gap) <= tolerance,
            }
        )
    return rows


def growth_by_graph() -> list[dict]:
    """How much bigger a second order graph is than the function it came from."""
    rows = []
    for label, graph in (
        ("smooth chain", smooth_chain()),
        ("quadratic", quadratic_graph()),
        ("layernorm", layernorm_graph()),
    ):
        row = hessian_vector_product(graph).as_dict()
        row["graph"] = label
        rows.append(row)
    return rows


def saturation_flattens_the_curvature(
    scales: Sequence[float] = (0.1, 1.0, 4.0, 16.0),
) -> list[dict]:
    """How the curvature of a saturating chain falls away as its input grows.

    A tanh is nearly linear near zero and flat far from it, so the curvature peaks somewhere in
    the middle. It does here, at a scale of about one, and by a scale of sixteen it has fallen
    by seven orders of magnitude. That is the difficulty with curvature methods on saturating
    networks stated as a measurement rather than as a warning: past saturation a method that
    divides by the curvature is dividing by nothing.
    """
    if not scales:
        raise ConfigError("there is nothing to sweep")
    graph = smooth_chain(2)
    built = hessian_vector_product(graph)
    base = random_feeds(built.graph, positive=True)
    rows = []
    for scale in scales:
        feeds = dict(base)
        feeds["x"] = base["x"] * scale
        rows.append(
            {
                "scale": scale,
                "largest_curvature": float(run(built.graph, feeds)[0].abs().max()),
            }
        )
    return rows
