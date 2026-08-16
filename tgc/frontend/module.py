from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, GraphError
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph
from tgc.verify.reference import run

# Building graphs out of layers rather than out of nodes.
#
# The tracer in frontend/trace.py takes a Python function and watches what it does. This takes a
# description of a model and builds the graph directly. Both are frontends and they answer
# different questions: a tracer works on code somebody already wrote, and a module system works
# when the model is being described rather than executed, which is where the shapes and the
# parameter names are known before anything runs.
#
# The part that turns out to need care is the names. Every parameter becomes a graph input, and
# a graph cannot have two inputs with one name, so a stack of four identical layers needs four
# distinct names for what the source called one thing. Deriving them from the position in the
# tree is the only scheme that survives a layer being reused, and getting it wrong does not
# produce a bad name, it produces a graph where two layers silently share a weight.
#
# Everything here is checked against the equivalent written in torch, with the same weights fed
# into both. That comparison is the reason a module system is worth having in a compiler at all:
# it is the only place where what the compiler thinks a layer means can be checked against what
# everybody else thinks it means.


@dataclass
class Parameter:
    """One learned tensor, and the name it will have in the graph."""

    name: str
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("a parameter needs a name")
        if any(size < 1 for size in self.sizes):
            raise ConfigError(f"{self.name} cannot have shape {list(self.sizes)}")

    @property
    def elements(self) -> int:
        """Numbers the parameter holds."""
        total = 1
        for size in self.sizes:
            total *= size
        return total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"name": self.name, "sizes": list(self.sizes), "elements": self.elements}


class Module:
    """Something that contributes parameters and nodes to a graph.

    Deliberately not a class hierarchy with hooks. A module declares what it needs and says how
    to build itself, and the two are separate methods because every parameter has to be declared
    before any of them is built: a graph's inputs are fixed when it is finished, and a layer
    that discovered a parameter halfway through building would be too late.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

    def parameters(self, prefix: str = "") -> list[Parameter]:
        """Every learned tensor this module needs, named by its position."""
        raise NotImplementedError

    def build(self, builder: Builder, source: str, names: dict[str, str]) -> str:
        """Emit this module's nodes and return the value it produces."""
        raise NotImplementedError

    def output_sizes(self, sizes: Sequence[int]) -> list[int]:
        """The shape this module produces from a given input shape."""
        return list(sizes)

    def path(self, prefix: str, leaf: str) -> str:
        """The name a parameter gets, given where it sits."""
        return f"{prefix}.{leaf}" if prefix else leaf


class Linear(Module):
    """A matrix product and an optional offset."""

    def __init__(self, width: int, output_width: int, *, bias: bool = True) -> None:
        super().__init__(name="linear")
        if min(width, output_width) < 1:
            raise ConfigError(f"a linear cannot be {width} by {output_width}")
        self.width = width
        self.output_width = output_width
        self.bias = bias

    def parameters(self, prefix: str = "") -> list[Parameter]:
        found = [Parameter(self.path(prefix, "weight"), (self.width, self.output_width))]
        if self.bias:
            found.append(Parameter(self.path(prefix, "bias"), (self.output_width,)))
        return found

    def build(self, builder: Builder, source: str, names: dict[str, str]) -> str:
        product = builder.matmul(source, names["weight"])
        if not self.bias:
            return product
        rows = builder.shape_of(product).sizes[0].value
        wide = builder.broadcast_to(names["bias"], [rows, self.output_width])
        return builder.add(product, wide)

    def output_sizes(self, sizes: Sequence[int]) -> list[int]:
        return [sizes[0], self.output_width]


class LayerNorm(Module):
    """Centre, scale by the standard deviation, then apply a gain and an offset.

    The epsilon goes inside the square root. Putting it outside is the mistake this layer is
    most often written with, it produces a number that is close for ordinary inputs and wrong by
    a factor for a row that is nearly constant, and the divergence is measured below rather than
    asserted away.
    """

    def __init__(self, width: int, epsilon: float = 1e-5) -> None:
        super().__init__(name="layernorm")
        if width < 1:
            raise ConfigError(f"a layernorm cannot be {width} wide")
        if epsilon <= 0:
            raise ConfigError(f"the epsilon has to be positive, got {epsilon}")
        self.width = width
        self.epsilon = epsilon

    def parameters(self, prefix: str = "") -> list[Parameter]:
        return [
            Parameter(self.path(prefix, "gain"), (self.width,)),
            Parameter(self.path(prefix, "offset"), (self.width,)),
        ]

    def build(self, builder: Builder, source: str, names: dict[str, str]) -> str:
        rows = builder.shape_of(source).sizes[0].value
        sizes = [rows, self.width]

        average = builder.broadcast_to(builder.mean(source, axes=[1], keepdims=True), sizes)
        centred = builder.sub(source, average)
        squared = builder.mul(centred, centred)
        variance = builder.broadcast_to(builder.mean(squared, axes=[1], keepdims=True), sizes)
        spread = builder.sqrt(builder.add(variance, builder.constant(self.epsilon)))
        normalised = builder.div(centred, spread)

        gain = builder.broadcast_to(names["gain"], sizes)
        offset = builder.broadcast_to(names["offset"], sizes)
        return builder.add(builder.mul(normalised, gain), offset)


class Mlp(Module):
    """A widening product, a rectifier and a narrowing product."""

    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__(name="mlp")
        if width < 1 or expansion < 1:
            raise ConfigError(f"an mlp cannot be {width} wide by {expansion}")
        self.width = width
        self.expansion = expansion
        self.up = Linear(width, width * expansion)
        self.down = Linear(width * expansion, width, bias=False)

    def parameters(self, prefix: str = "") -> list[Parameter]:
        return self.up.parameters(self.path(prefix, "up")) + self.down.parameters(
            self.path(prefix, "down")
        )

    def build(self, builder: Builder, source: str, names: dict[str, str]) -> str:
        widened = self.up.build(
            builder, source, {"weight": names["up.weight"], "bias": names["up.bias"]}
        )
        activated = builder.relu(widened)
        return self.down.build(builder, activated, {"weight": names["down.weight"]})


class Sequential(Module):
    """Several modules applied in order.

    The container is where the naming happens. Each child gets a prefix from its position, so
    two copies of the same layer object still produce two sets of parameters, which is the
    behaviour a stack of identical layers needs and the opposite of what sharing a name would
    give.
    """

    def __init__(self, modules: Sequence[Module]) -> None:
        super().__init__(name="sequential")
        if not modules:
            raise ConfigError("a sequence needs something in it")
        self.modules = list(modules)

    def parameters(self, prefix: str = "") -> list[Parameter]:
        found: list[Parameter] = []
        for index, module in enumerate(self.modules):
            found.extend(module.parameters(self.path(prefix, f"{index}")))
        return found

    def build(self, builder: Builder, source: str, names: dict[str, str]) -> str:
        current = source
        for index, module in enumerate(self.modules):
            local = {
                key[len(f"{index}.") :]: value
                for key, value in names.items()
                if key.startswith(f"{index}.")
            }
            current = module.build(builder, current, local)
        return current

    def output_sizes(self, sizes: Sequence[int]) -> list[int]:
        current = list(sizes)
        for module in self.modules:
            current = module.output_sizes(current)
        return current


def compile_module(module: Module, sizes: Sequence[int], name: str = "x") -> Graph:
    """Turn a module into a graph, with its parameters as inputs.

    The parameters come after the activation in the input list, and they come in the order the
    module declared them. Both of those are choices a caller has to be able to rely on, because
    the only thing connecting a weight in the caller's hands to a name in the graph is the
    position it was declared at.
    """
    if len(sizes) != 2:
        raise ConfigError(f"a module takes a matrix, got rank {len(sizes)}")
    builder = Builder()
    source = builder.input(list(sizes), name=name)

    declared = module.parameters()
    seen = {parameter.name for parameter in declared}
    if len(seen) != len(declared):
        raise GraphError("two parameters share a name, so one would shadow the other")

    names = {
        parameter.name: builder.input(list(parameter.sizes), name=parameter.name)
        for parameter in declared
    }
    return builder.finish(module.build(builder, source, names))


def parameter_count(module: Module) -> int:
    """Numbers a module has to be given."""
    return sum(parameter.elements for parameter in module.parameters())


def torch_linear(
    source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """The same layer in torch, for the comparison."""
    result = source @ weight
    return result if bias is None else result + bias


def torch_layernorm(
    source: torch.Tensor, gain: torch.Tensor, offset: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """The same normalisation in torch, written out rather than called.

    Written out because torch.nn.functional.layer_norm is the thing being checked against, and a
    comparison against the library that also uses the library is not a comparison. The two are
    checked against each other separately below.
    """
    average = source.mean(dim=-1, keepdim=True)
    centred = source - average
    variance = centred.pow(2).mean(dim=-1, keepdim=True)
    return centred / (variance + epsilon).sqrt() * gain + offset


def random_weights(module: Module, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """Values for every parameter a module declared."""
    generator = torch.Generator().manual_seed(seed)
    return {
        parameter.name: torch.randn(list(parameter.sizes), generator=generator)
        for parameter in module.parameters()
    }


def check_linear(rows: int = 8, width: int = 16, output_width: int = 32) -> dict:
    """The compiled linear against the torch one, on the same weights."""
    module = Linear(width, output_width)
    graph = compile_module(module, [rows, width])
    weights = random_weights(module)
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(rows, width, generator=generator)

    from_graph = run(graph, {"x": source, **weights})[0]
    from_torch = torch_linear(source, weights["weight"], weights["bias"])
    return {
        "identical": bool(torch.equal(from_graph, from_torch)),
        "largest_gap": float((from_graph - from_torch).abs().max()),
    }


def check_layernorm(rows: int = 8, width: int = 16, epsilon: float = 1e-5) -> dict:
    """The compiled normalisation against a hand written one and against torch.

    Three implementations rather than two. The hand written one is what the graph is checked
    against, and torch's own is checked against that, so a shared misunderstanding between the
    graph and the reference would show up as a disagreement with the library rather than as
    agreement all round.
    """
    module = LayerNorm(width, epsilon)
    graph = compile_module(module, [rows, width])
    weights = random_weights(module)
    generator = torch.Generator().manual_seed(12)
    source = torch.randn(rows, width, generator=generator)

    from_graph = run(graph, {"x": source, **weights})[0]
    by_hand = torch_layernorm(source, weights["gain"], weights["offset"], epsilon)
    from_library = torch.nn.functional.layer_norm(
        source, (width,), weight=weights["gain"], bias=weights["offset"], eps=epsilon
    )
    return {
        "graph_against_hand": float((from_graph - by_hand).abs().max()),
        "hand_against_library": float((by_hand - from_library).abs().max()),
    }


def epsilon_outside_the_root(
    source: torch.Tensor, gain: torch.Tensor, offset: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """The version with the epsilon in the wrong place."""
    average = source.mean(dim=-1, keepdim=True)
    centred = source - average
    variance = centred.pow(2).mean(dim=-1, keepdim=True)
    return centred / (variance.sqrt() + epsilon) * gain + offset


def where_the_epsilon_goes(rows: int = 8, width: int = 16, epsilon: float = 1e-5) -> dict:
    """What putting the epsilon outside the root costs, on ordinary and degenerate rows.

    Nothing worth noticing on ordinary data, which is why the mistake survives review. On a row
    that is nearly constant the variance is tiny, its root is much larger than the variance, and
    the two versions differ by a factor rather than in the last place.
    """
    generator = torch.Generator().manual_seed(13)
    gain = torch.ones(width)
    offset = torch.zeros(width)

    ordinary = torch.randn(rows, width, generator=generator)
    flat = torch.ones(rows, width) + torch.randn(rows, width, generator=generator) * 1e-4

    rows_out = {}
    for label, source in (("ordinary", ordinary), ("nearly constant", flat)):
        right = torch_layernorm(source, gain, offset, epsilon)
        wrong = epsilon_outside_the_root(source, gain, offset, epsilon)
        gap = float((right - wrong).abs().max())
        scale = float(right.abs().max())
        rows_out[label] = {
            "largest_gap": gap,
            "relative_gap": gap / scale if scale else gap,
        }
    return rows_out


def check_mlp(rows: int = 8, width: int = 16, expansion: int = 4) -> dict:
    """The compiled mlp against the torch one."""
    module = Mlp(width, expansion)
    graph = compile_module(module, [rows, width])
    weights = random_weights(module)
    generator = torch.Generator().manual_seed(14)
    source = torch.randn(rows, width, generator=generator)

    from_graph = run(graph, {"x": source, **weights})[0]
    widened = torch_linear(source, weights["up.weight"], weights["up.bias"])
    from_torch = torch_linear(torch.relu(widened), weights["down.weight"], None)
    return {
        "identical": bool(torch.equal(from_graph, from_torch)),
        "largest_gap": float((from_graph - from_torch).abs().max()),
    }


def stack(depth: int = 4, width: int = 16) -> Sequential:
    """A stack of normalisation and mlp, which is a transformer block without attention."""
    if depth < 1:
        raise ConfigError(f"a stack needs at least one layer, got {depth}")
    layers: list[Module] = []
    for _ in range(depth):
        layers.append(LayerNorm(width))
        layers.append(Mlp(width))
    return Sequential(layers)


def check_stack(depth: int = 3, rows: int = 8, width: int = 16) -> dict:
    """A whole stack against the same thing composed in torch."""
    module = stack(depth, width)
    graph = compile_module(module, [rows, width])
    weights = random_weights(module)
    generator = torch.Generator().manual_seed(15)
    source = torch.randn(rows, width, generator=generator)

    current = source
    for index in range(depth):
        norm = index * 2
        mlp = norm + 1
        current = torch_layernorm(
            current, weights[f"{norm}.gain"], weights[f"{norm}.offset"], 1e-5
        )
        widened = torch_linear(current, weights[f"{mlp}.up.weight"], weights[f"{mlp}.up.bias"])
        current = torch_linear(torch.relu(widened), weights[f"{mlp}.down.weight"], None)

    from_graph = run(graph, {"x": source, **weights})[0]
    gap = float((from_graph - current).abs().max())
    scale = float(current.abs().max())
    return {
        "identical": bool(torch.equal(from_graph, current)),
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
    }


def names_are_unique(depth: int = 8, width: int = 16) -> dict:
    """Whether a stack of identical layers gets distinct parameter names.

    The failure this guards against does not look like a failure. Two layers sharing a name
    produce a graph that builds, runs, and computes a model with tied weights, which is a
    different model that trains and converges to something.
    """
    module = stack(depth, width)
    declared = [parameter.name for parameter in module.parameters()]
    return {
        "parameters": len(declared),
        "distinct": len(set(declared)),
        "all_unique": len(declared) == len(set(declared)),
    }


def a_repeated_layer_object_still_gets_two_names(width: int = 16) -> dict:
    """The same module instance used twice in a sequence.

    The names come from the position rather than from the object, so reusing an instance gives
    two independent sets of parameters. Whether that is what a caller wants is a separate
    question, and the answer here is that the compiler cannot know, so it does the thing that
    cannot silently tie two layers together.
    """
    layer = LayerNorm(width)
    module = Sequential([layer, layer])
    declared = [parameter.name for parameter in module.parameters()]
    return {"names": declared, "distinct": len(set(declared))}


def graph_size(depth: int = 4, width: int = 16, rows: int = 8) -> dict:
    """Nodes and parameters for a stack of a given depth."""
    module = stack(depth, width)
    graph = compile_module(module, [rows, width])
    return {
        "depth": depth,
        "nodes": len(graph.nodes),
        "inputs": len(graph.inputs),
        "parameter_elements": parameter_count(module),
    }


def size_by_depth(depths: Sequence[int] = (1, 2, 4, 8)) -> list[dict]:
    """How the graph grows with the stack.

    Linearly in both, which is the answer that says the frontend is not doing anything clever.
    A frontend that shared structure between identical layers would show a graph that grows more
    slowly than the parameters, and this one does not, because sharing structure is what the
    outlining pass is for and doing it here would put the same decision in two places.
    """
    if not depths:
        raise ConfigError("there is nothing to sweep")
    return [graph_size(depth) for depth in depths]


def a_duplicate_name_is_refused(width: int = 16) -> bool:
    """Whether a module that declares two parameters with one name is caught.

    Built by hand, because none of the modules here can produce it. It is the check that would
    have caught the naming failure if the naming had been done any other way, so it is worth
    having even though nothing currently trips it.
    """

    class Broken(Module):
        def parameters(self, _prefix: str = "") -> list[Parameter]:
            return [Parameter("shared", (width,)), Parameter("shared", (width,))]

        def build(self, builder: Builder, source: str, names: dict[str, str]) -> str:
            return builder.add(source, names["shared"])

    try:
        compile_module(Broken(), [8, width])
    except GraphError:
        return True
    return False


def parameter_share_of_the_graph(depth: int = 4, width: int = 64) -> dict:
    """How much of a compiled model's inputs are weights rather than data.

    Almost all of them, which is the fact that decides how a compiler should treat inputs. Four
    layers give one activation and twenty parameters, so the interesting question about an input
    is not how to pass it but whether it changes between calls, and for twenty out of twenty one
    it does not.
    """
    module = stack(depth, width)
    graph = compile_module(module, [8, width])
    return {
        "inputs": len(graph.inputs),
        "parameters": len(module.parameters()),
        "activations": len(graph.inputs) - len(module.parameters()),
    }


@dataclass
class ModuleReport:
    """What a module compiles to."""

    label: str
    nodes: int = 0
    parameters: int = 0
    elements: int = 0
    checks: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "module": self.label,
            "nodes": self.nodes,
            "parameters": self.parameters,
            "elements": self.elements,
        }


def compare_modules(rows: int = 8, width: int = 16) -> list[dict]:
    """Every module in this file, compiled and sized.

    The layernorm is the one worth looking at. It has the fewest parameter elements and by far
    the most nodes, fourteen against three for a linear, because a normalisation is two
    reductions and a dozen elementwise operations where a linear is a product and an offset.
    Seven nodes per parameter against one and a half is why fusion matters more to a
    normalisation than to anything else in a model.
    """
    modules: list[tuple[str, Module]] = [
        ("linear", Linear(width, width)),
        ("layernorm", LayerNorm(width)),
        ("mlp", Mlp(width)),
        ("stack of two", stack(2, width)),
    ]
    rows_out = []
    for label, module in modules:
        graph = compile_module(module, [rows, width])
        rows_out.append(
            ModuleReport(
                label=label,
                nodes=len(graph.nodes),
                parameters=len(module.parameters()),
                elements=parameter_count(module),
            ).as_dict()
        )
    return rows_out


def the_normalisation_is_the_expensive_one_to_write(width: int = 16) -> dict:
    """Nodes per parameter, per module."""
    rows = {row["module"]: row for row in compare_modules(width=width)}
    return {
        label: round(row["nodes"] / max(row["parameters"], 1), 3) for label, row in rows.items()
    }


def epsilon_that_is_not_positive_is_refused() -> bool:
    """Whether a normalisation with a zero epsilon is caught.

    It is, and the reason is not tidiness. A zero epsilon divides by the root of the variance,
    which for a constant row is a division of zero by zero, and the answer is nan rather than
    the offset the layer is supposed to produce.
    """
    try:
        LayerNorm(16, epsilon=0.0)
    except ConfigError:
        return True
    return False


def what_a_zero_epsilon_would_do(width: int = 16) -> dict:
    """The nan that the check above prevents, produced directly so it can be seen."""
    source = torch.ones(4, width)
    gain = torch.ones(width)
    offset = torch.zeros(width)
    return {
        "with_epsilon": bool(torch.isfinite(torch_layernorm(source, gain, offset, 1e-5)).all()),
        "without": bool(torch.isfinite(torch_layernorm(source, gain, offset, 0.0)).all()),
    }


def scale_matches_the_convention(width: int = 64) -> dict:
    """Whether the layernorm here divides by the population deviation or the sample one.

    The population one, matching torch. It is one number in the denominator and the two
    conventions differ by the square root of the width over the width minus one, which at a
    width of sixty four is under a percent and at a width of four is thirteen.
    """
    generator = torch.Generator().manual_seed(16)
    source = torch.randn(8, width, generator=generator)
    centred = source - source.mean(dim=-1, keepdim=True)
    population = centred.pow(2).mean(dim=-1, keepdim=True).sqrt()
    sample = centred.pow(2).sum(dim=-1, keepdim=True).div(width - 1).sqrt()
    return {
        "difference": float((population - sample).abs().max()),
        "predicted_ratio": round(math.sqrt(width / (width - 1)), 6),
        "measured_ratio": round(float((sample / population).mean()), 6),
    }
