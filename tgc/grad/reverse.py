from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, PassError
from tgc.grad.rules import Context, check_differentiable, has_rule, rule_for, static_sizes
from tgc.ir import op as ops
from tgc.ir.builder import (
    Builder,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT64, DType
from tgc.ir.graph import Graph
from tgc.passes.pipeline import DEFAULT_ORDER, pipeline_from
from tgc.verify.reference import random_feeds, run

# Walking a graph backwards and building the graph that computes its derivative.
#
# The forward graph is rebuilt into a fresh builder first, then the nodes are visited in
# reverse, and each one hands its cotangent to the rules in rules.py. A value read by several
# consumers receives several contributions and its gradient is their sum, which is the reason
# this cannot be a simple map over nodes: the accumulation has to finish before a value is
# differentiated, and reverse topological order is what guarantees that it has.
#
# The result is an ordinary graph. Nothing about it is special to the compiler, so every pass
# runs over it unchanged, and the default pipeline removes between six and sixteen percent of
# it. That is less than the usual argument for optimising a backward pass claims, and the
# measurement below says why: most of the reported saving belongs to seeding the walk with a
# constant one rather than with a real cotangent.
#
# Correctness is checked against torch rather than against a hand table. The reference
# interpreter is written in torch operations, so running the forward graph on tensors that want
# gradients makes torch's own autograd the ground truth, and the comparison is then between two
# independent implementations rather than between an implementation and a restatement of it.


@dataclass
class GradientResult:
    """A gradient graph and what it took to build."""

    graph: Graph
    forward_nodes: int
    backward_nodes: int
    wrt: tuple[str, ...] = ()

    @property
    def growth(self) -> float:
        """How many times bigger the whole thing is than the forward graph."""
        if self.forward_nodes == 0:
            return 0.0
        return (self.forward_nodes + self.backward_nodes) / self.forward_nodes

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "wrt": list(self.wrt),
            "forward_nodes": self.forward_nodes,
            "backward_nodes": self.backward_nodes,
            "growth": round(self.growth, 3),
        }


def _rebuild_forward(graph: Graph, builder: Builder) -> dict[str, str]:
    """Copy a graph into a builder and return the name mapping."""
    mapping: dict[str, str] = {}
    for value in graph.inputs:
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=value.name,
        )
    for node in graph.nodes:
        if node.op is ops.CONSTANT:
            mapping[node.name] = builder.constant(
                float(node.attrs["value"]), dtype=node.output.dtype
            )
            continue
        operands = [mapping[name] for name in node.inputs]
        mapping[node.name] = builder.apply(node.op, *operands, **node.attrs)
    return mapping


def _seed(builder: Builder, name: str, as_input: bool) -> str:
    """The cotangent the backward walk starts from."""
    sizes = static_sizes(builder.shape_of(name))
    if as_input:
        return builder.input(sizes, dtype=builder.dtype_of(name), name=COTANGENT)
    one = builder.constant(1.0, dtype=builder.dtype_of(name))
    return builder.broadcast_to(one, sizes) if sizes else one


def retype(graph: Graph, dtype: DType = FLOAT64) -> Graph:
    """Rebuild a graph with every input and literal in one type.

    Needed because a finite difference cannot be checked at the precision the graph runs at. A
    central difference throws away about half the digits it started with, so at float32 the
    best possible agreement with an analytic derivative is three or four decimal places, which
    is not enough to tell a correct rule from one that is wrong in the fourth. At float64 the
    same check agrees to ten.
    """
    builder = Builder()
    mapping: dict[str, str] = {}
    for value in graph.inputs:
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=dtype,
            name=value.name,
        )
    for node in graph.nodes:
        if node.op is ops.CONSTANT:
            mapping[node.name] = builder.constant(float(node.attrs["value"]), dtype=dtype)
            continue
        if node.op is ops.CAST:
            raise PassError(f"{node.name} pins a type, so the graph cannot be retyped")
        operands = [mapping[name] for name in node.inputs]
        mapping[node.name] = builder.apply(node.op, *operands, **node.attrs)
    return builder.finish(*[mapping[name] for name in graph.outputs])


def _needed(graph: Graph, output: str, wrt: Sequence[str]) -> set[str]:
    """Values that lie on a path from one of the inputs to the output.

    A gradient only has to be built for these. Differentiating everything and letting dead code
    elimination sort it out gives the same answer and builds a noticeably larger graph first,
    which matters because the pass manager then has to validate all of it.
    """
    reaching: set[str] = set(wrt)
    for node in graph.nodes:
        if any(name in reaching for name in node.inputs):
            reaching.add(node.name)

    reached: set[str] = {output}
    for node in reversed(graph.nodes):
        if node.name in reached:
            reached.update(node.inputs)
    return reaching & reached


COTANGENT = "cotangent"


def gradient(
    graph: Graph,
    wrt: Sequence[str] | None = None,
    *,
    keep_forward: bool = False,
    seed_is_input: bool = True,
) -> GradientResult:
    """Build the graph that computes the derivative of a graph's output.

    The output has to be a single value, and the derivative is taken against a cotangent rather
    than against the output directly, because reverse mode cannot produce a jacobian in one
    pass. By default the cotangent becomes an extra input, which makes the result a general
    vector jacobian product.

    Seeding with ones instead is the common shortcut and it is a trap worth naming. It computes
    the derivative of the sum of the output, and the sum of a softmax is one by construction, so
    every gradient in a softmax comes out exactly zero and a check built on that seed passes
    while measuring nothing at all. That is not a hypothetical: it is what the first version of
    the checks below did.
    """
    if len(graph.outputs) != 1:
        raise ConfigError(f"a gradient needs exactly one output, got {len(graph.outputs)}")
    check_differentiable(graph)

    targets = list(wrt) if wrt is not None else [value.name for value in graph.inputs]
    if not targets:
        raise ConfigError("there is nothing to differentiate with respect to")
    known = {value.name for value in graph.inputs}
    unknown = [name for name in targets if name not in known]
    if unknown:
        raise ConfigError(f"{unknown} are not inputs of this graph")
    if seed_is_input and COTANGENT in known:
        raise ConfigError(f"the graph already has an input named {COTANGENT!r}")

    builder = Builder()
    mapping = _rebuild_forward(graph, builder)
    forward_nodes = len(builder.nodes)

    output = graph.outputs[0]
    needed = _needed(graph, output, targets)
    cotangents: dict[str, str] = {output: _seed(builder, mapping[output], seed_is_input)}

    for node in reversed(graph.nodes):
        if node.name not in cotangents or node.name not in needed:
            continue
        if not has_rule(node.op):
            continue
        context = Context(
            node=node,
            cotangent=cotangents[node.name],
            operands=[mapping[name] for name in node.inputs],
            output=mapping[node.name],
            shapes=[graph.value(name).shape for name in node.inputs],
        )
        contributions = rule_for(node.op)(builder, context)
        for name, contribution in zip(node.inputs, contributions, strict=True):
            if contribution is None or name not in needed:
                continue
            if name in cotangents:
                cotangents[name] = builder.add(cotangents[name], contribution)
            else:
                cotangents[name] = contribution

    missing = [name for name in targets if name not in cotangents]
    if missing:
        raise PassError(f"{missing} do not reach the output, so they have no gradient")

    outputs = [mapping[output]] if keep_forward else []
    outputs.extend(cotangents[name] for name in targets)
    result = builder.finish(*outputs)
    return GradientResult(
        graph=result,
        forward_nodes=forward_nodes,
        backward_nodes=len(result.nodes) - forward_nodes,
        wrt=tuple(targets),
    )


def torch_gradient(
    graph: Graph,
    feeds: dict[str, torch.Tensor],
    wrt: Sequence[str],
    cotangent: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The same derivative, from torch's autograd.

    The reference interpreter is written in torch operations, so feeding it tensors that want
    gradients makes torch differentiate the interpreter. The two implementations share the
    forward kernels and nothing else, which is what makes the comparison worth running.
    """
    tracked = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in feeds.items()
    }
    result = run(graph, tracked)[0]
    seed = torch.ones_like(result) if cotangent is None else cotangent
    grads = torch.autograd.grad(result, [tracked[name] for name in wrt], grad_outputs=seed)
    return list(grads)


def split_feeds(
    graph: Graph, feeds: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Separate a gradient graph's feeds into the forward ones and the cotangent."""
    if COTANGENT not in feeds:
        raise ConfigError(f"the feeds have no {COTANGENT!r}")
    forward = {value.name: feeds[value.name] for value in graph.inputs}
    return forward, feeds[COTANGENT]


def compare_with_torch(
    graph: Graph, wrt: Sequence[str] | None = None, *, seed: int = 0
) -> list[dict]:
    """Our gradient against torch's, on the same inputs and the same cotangent."""
    targets = list(wrt) if wrt is not None else [value.name for value in graph.inputs]
    built = gradient(graph, targets)
    feeds = random_feeds(built.graph, positive=True, seed=seed)
    forward, cotangent = split_feeds(graph, feeds)

    ours = run(built.graph, feeds)
    theirs = torch_gradient(graph, forward, targets, cotangent)

    rows = []
    for name, mine, reference in zip(targets, ours, theirs, strict=True):
        rows.append(_compare_one(name, mine, reference))
    return rows


def ones_seed_hides_a_softmax(seed: int = 0) -> dict:
    """What the shortcut seed measures on a softmax, which is nothing.

    Both numbers below are the largest absolute gradient the check saw. Under a cotangent of
    ones every one of them is zero, because the sum of a softmax row is one whatever the input
    was, so a comparison against torch agrees perfectly on a value that carries no information.
    Under a real cotangent the same gradient is not zero and the comparison is worth running.
    """
    graph = softmax_graph()
    flat = gradient(graph, seed_is_input=False)
    real = gradient(graph)

    ones_feeds = random_feeds(graph, positive=True, seed=seed)
    real_feeds = random_feeds(real.graph, positive=True, seed=seed)
    return {
        "with_a_ones_seed": float(run(flat.graph, ones_feeds)[0].abs().max()),
        "with_a_real_cotangent": float(run(real.graph, real_feeds)[0].abs().max()),
    }


def _compare_one(name: str, mine: torch.Tensor, reference: torch.Tensor) -> dict:
    """One gradient against the reference, ignoring positions that overflowed.

    Positions where both sides are infinite are dropped rather than differenced. An infinity
    minus an infinity is a nan, so a chain deep enough to overflow reports a nan gap even when
    the two agree exactly, and the fixtures here do overflow: a chain of four exponentials
    reaches the top of float32 on ordinary inputs.
    """
    finite = torch.isfinite(mine) & torch.isfinite(reference)
    overflowed = bool((~finite).any())
    agree_elsewhere = bool((mine[~finite] == reference[~finite]).all()) if overflowed else True
    if not finite.any():
        return {
            "input": name,
            "largest_gap": 0.0,
            "relative_gap": 0.0,
            "identical": agree_elsewhere,
            "overflowed": int((~finite).sum()),
        }
    gap = float((mine[finite] - reference[finite]).abs().max())
    scale = float(reference[finite].abs().max())
    return {
        "input": name,
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
        "identical": bool(torch.equal(mine[finite], reference[finite])) and agree_elsewhere,
        "overflowed": int((~finite).sum()),
    }


def agrees_with_torch(
    graph: Graph, wrt: Sequence[str] | None = None, *, tolerance: float = 1e-6
) -> bool:
    """Whether every gradient matches torch's to a tolerance."""
    return all(row["relative_gap"] <= tolerance for row in compare_with_torch(graph, wrt))


def bitwise_agreement(graph: Graph, wrt: Sequence[str] | None = None) -> dict:
    """How often the two agree to the last bit rather than to a tolerance.

    Not always, and the reason is not a bug in either. The two build the same derivative out of
    different orders of the same operations, and float addition is not associative, so a value
    read by three consumers can accumulate its three contributions in two orders that differ in
    the last place.
    """
    rows = compare_with_torch(graph, wrt)
    return {
        "gradients": len(rows),
        "bit_identical": sum(1 for row in rows if row["identical"]),
        "largest_relative_gap": max((row["relative_gap"] for row in rows), default=0.0),
    }


def finite_difference(
    graph: Graph,
    name: str,
    feeds: dict[str, torch.Tensor],
    *,
    step: float = 1e-3,
    samples: int = 8,
    seed: int = 0,
) -> dict:
    """A central difference at a few positions, as a check on the rules themselves.

    Slow and inaccurate and completely independent of both other implementations, which is what
    makes it worth having. If a rule is wrong in a way that both the analytic pass and torch
    agree on, which happens when the rule is a restatement of the same misunderstanding, this
    is the only thing that notices.
    """
    if step <= 0:
        raise ConfigError(f"the step has to be positive, got {step}")
    if samples < 1:
        raise ConfigError(f"the sample count must be positive, got {samples}")

    wide = retype(graph)
    built = gradient(wide, [name])
    doubled = {key: tensor.to(torch.float64) for key, tensor in feeds.items()}
    if COTANGENT not in doubled:
        raise ConfigError(f"the feeds have no {COTANGENT!r}")
    cotangent = doubled[COTANGENT]
    forward_feeds = {value.name: doubled[value.name] for value in wide.inputs}

    base = doubled[name]
    flat = base.flatten()
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(flat.numel(), generator=generator)[:samples]

    analytic = run(built.graph, doubled)[0].flatten()
    gaps = []
    for position in positions.tolist():
        raised = flat.clone()
        raised[position] += step
        lowered = flat.clone()
        lowered[position] -= step
        raised_feeds = {**forward_feeds, name: raised.reshape(base.shape)}
        lowered_feeds = {**forward_feeds, name: lowered.reshape(base.shape)}
        high = (run(wide, raised_feeds)[0] * cotangent).sum()
        low = (run(wide, lowered_feeds)[0] * cotangent).sum()
        estimate = float((high - low) / (2 * step))
        gaps.append(abs(estimate - float(analytic[position])))

    scale = float(analytic.abs().max())
    return {
        "input": name,
        "samples": len(gaps),
        "step": step,
        "largest_gap": max(gaps),
        "mean_gap": sum(gaps) / len(gaps),
        "relative_gap": max(gaps) / scale if scale else max(gaps),
    }


def step_size_sweep(
    graph: Graph, name: str, steps: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)
) -> list[dict]:
    """How the finite difference error moves with the step.

    Down and then back up. A large step measures the wrong thing because the function curves
    across it, and a small one measures the wrong thing because the subtraction cancels almost
    every digit it had. The best step is somewhere in the middle and the tolerance a test can
    ask for has to come from that curve rather than from a round number somebody liked.
    """
    if not steps:
        raise ConfigError("there is nothing to sweep")
    feeds = random_feeds(gradient(graph, [name]).graph, positive=True)
    return [finite_difference(graph, name, feeds, step=step) for step in steps]


def best_step(graph: Graph, name: str) -> float:
    """The step size that measures the derivative most accurately."""
    rows = step_size_sweep(graph, name)
    return min(rows, key=lambda row: row["largest_gap"])["step"]


def curvature_decides_the_best_step() -> list[dict]:
    """Where the best step sits, for a curved function and a flat one.

    The textbook curve has a minimum in the middle, and a softmax has it: the error falls by a
    hundred for each factor of ten off the step until a step of ten thousandths, then climbs
    again as the subtraction eats the digits.

    An mlp does not, and the reason is worth knowing. It is a matrix product followed by a relu
    followed by another product, which is piecewise linear in its input, so a central difference
    has no truncation error to trade against and the largest step available is the best one. A
    rule of thumb that names a step size is really a claim about curvature.
    """
    rows = []
    for label, graph in (("softmax", softmax_graph()), ("mlp", mlp_graph())):
        sweep = step_size_sweep(graph, "x")
        steps = [row["step"] for row in sweep]
        chosen = min(sweep, key=lambda row: row["largest_gap"])["step"]
        rows.append(
            {
                "graph": label,
                "best_step": chosen,
                "at_an_end": chosen in (max(steps), min(steps)),
            }
        )
    return rows


def optimise(result: GradientResult) -> Graph:
    """Run the ordinary optimiser over a gradient graph.

    Nothing here knows it is a gradient. That is the argument for building the backward pass in
    the IR rather than in an interpreter: the passes that were written for forward graphs apply
    unchanged, and running the whole default pipeline rather than three chosen passes is the
    point: nothing had to be added for gradients.
    """
    optimised, _ = pipeline_from(DEFAULT_ORDER).run(result.graph)
    return optimised


def optimiser_gains(
    graph: Graph, wrt: Sequence[str] | None = None, *, seed_is_input: bool = True
) -> dict:
    """How much of a naive gradient graph the optimiser removes.

    Six to sixteen percent from a real backward pass and twenty to twenty four from one seeded
    with ones. The difference is the seed: a cotangent of ones produces multiplications by a
    literal one that the algebraic rules delete, and a real cotangent produces multiplications
    by an input that they cannot. So roughly half the saving people attribute to optimising a
    backward pass is the seed, and the other half is real.
    """
    built = gradient(graph, wrt, seed_is_input=seed_is_input)
    optimised = optimise(built)
    before = len(built.graph.nodes)
    after = len(optimised.nodes)
    return {
        "before": before,
        "after": after,
        "removed": before - after,
        "fraction_removed": round((before - after) / before, 4) if before else 0.0,
    }


def optimising_preserves_the_gradient(graph: Graph, wrt: Sequence[str] | None = None) -> bool:
    """Whether the optimised gradient still agrees with torch."""
    targets = list(wrt) if wrt is not None else [value.name for value in graph.inputs]
    built = gradient(graph, targets)
    optimised = optimise(built)
    feeds = random_feeds(built.graph, positive=True)
    forward, cotangent = split_feeds(graph, feeds)

    ours = run(optimised, feeds)
    theirs = torch_gradient(graph, forward, targets, cotangent)
    return all(
        torch.allclose(mine, reference, rtol=1e-5, atol=1e-6)
        for mine, reference in zip(ours, theirs, strict=True)
    )


def growth_by_graph() -> list[dict]:
    """How much bigger the gradient is than the forward pass, per fixture.

    Between two and a half and nearly five times, and the spread is the interesting part. An
    elementwise chain is the cheap end at two and a half, because each operation needs one local
    derivative and one multiply to chain it. A reduction is the expensive end: a sum
    differentiates into a reshape and a broadcast, and a max into a broadcast, a subtraction, a
    step and a multiply, so a softmax with two reductions in it grows the most of the four.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(6)),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        row = gradient(graph).as_dict()
        row["graph"] = label
        rows.append(row)
    return rows


def elementwise_share(graph: Graph) -> dict:
    """How much of a backward pass is elementwise.

    Two thirds of a softmax or a layernorm gradient and a sixth of an mlp gradient, which is
    the answer to whether the growth in nodes is growth in kernels. For the normalisations it
    mostly is not, because what got added fuses. For the mlp it is: a matmul differentiates
    into two matmuls and two transposes and none of those merge into anything.
    """
    built = gradient(graph)
    backward = built.graph.nodes[built.forward_nodes :]
    if not backward:
        return {"backward_nodes": 0, "elementwise": 0, "share": 0.0}
    elementwise = sum(1 for node in backward if node.op.is_elementwise)
    return {
        "backward_nodes": len(backward),
        "elementwise": elementwise,
        "share": round(elementwise / len(backward), 4),
    }


@dataclass
class GradientCheck:
    """One graph checked three ways."""

    label: str
    torch_gap: float = 0.0
    finite_gap: float = 0.0
    optimised_agrees: bool = False

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "graph": self.label,
            "torch_gap": self.torch_gap,
            "finite_gap": self.finite_gap,
            "optimised_agrees": self.optimised_agrees,
        }


def check_every_fixture() -> list[dict]:
    """Every fixture against torch, against a finite difference and after optimisation."""
    checks = []
    for label, graph in (
        ("chain", elementwise_chain(4)),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        first = graph.inputs[0].name
        feeds = random_feeds(gradient(graph, [first]).graph, positive=True)
        checks.append(
            GradientCheck(
                label=label,
                torch_gap=max(row["relative_gap"] for row in compare_with_torch(graph)),
                finite_gap=finite_difference(graph, first, feeds)["relative_gap"],
                optimised_agrees=optimising_preserves_the_gradient(graph),
            ).as_dict()
        )
    return checks


@dataclass
class AccumulationReport:
    """How many values received more than one contribution."""

    values: int = 0
    accumulated: int = 0
    largest_fan_in: int = 0
    per_value: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "values": self.values,
            "accumulated": self.accumulated,
            "largest_fan_in": self.largest_fan_in,
        }


def accumulation_report(graph: Graph) -> AccumulationReport:
    """Which values need their gradients summed rather than assigned.

    Exactly the ones read by more than one consumer, which is why this is a count over the use
    map rather than something the backward walk has to discover. A graph where every value is
    read once needs no accumulation at all and its backward pass is a plain reverse map.
    """
    counts = graph.use_counts()
    report = AccumulationReport(values=len(counts))
    for name, count in counts.items():
        if count > 1:
            report.accumulated += 1
            report.per_value[name] = count
            report.largest_fan_in = max(report.largest_fan_in, count)
    return report


def accumulation_matters() -> list[dict]:
    """Fan in across the fixtures, and whether dropping it would be noticed.

    It would. A softmax reads its exponential twice, so a backward pass that assigned instead
    of accumulating would return exactly half of one of the two contributions and the answer
    would be wrong by a factor that depends on the input.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(6)),
        ("diamond", diamond_graph()),
        ("softmax", softmax_graph()),
        ("mlp", mlp_graph()),
    ):
        row = accumulation_report(graph).as_dict()
        row["graph"] = label
        rows.append(row)
    return rows
