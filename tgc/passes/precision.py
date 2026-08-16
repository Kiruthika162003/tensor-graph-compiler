from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.dtype import BFLOAT16, FLOAT16, FLOAT32, DType
from tgc.ir.graph import Graph
from tgc.verify.reference import random_feeds, run

# Deciding which parts of a graph can run in half the bits.
#
# Half precision halves the traffic and roughly doubles the arithmetic rate, and it has eleven
# bits of mantissa against twenty four and a top of sixty five thousand against ten to the
# thirty eighth. Both of those limits are reachable by ordinary models, so the question is never
# whether to use it, it is which operations to leave alone.
#
# The IR already refuses one of the answers. A reduction over float16 accumulates in float32,
# because ir/dtype.py says so, and that is not an optimisation this pass gets to undo. Added one
# at a time, a half precision running total of ones stalls at two thousand and forty eight and
# never moves again, and no policy above the type rules can put that back.
#
# What is left is the elementwise work and the products. Narrowing the products costs five ten
# thousandths of relative error on an mlp and is the policy worth having. Narrowing everything
# survives on these fixtures, which was not what I expected: the exponential of a score of
# twelve is past the top of float16, and a softmax gets away with it only because it subtracts
# the row maximum first. Remove that subtraction and the same numbers overflow.
#
# The memory result came out backwards and it is the most useful thing here. Counting every
# value in the rewritten graph, the narrow policies use three times more memory than the wide
# one, because every cast materialises a copy of what it converted. The saving is real only for
# a compiler that folds the conversion into the consumer, and the function that counts bytes
# without the casts is what shows the saving arriving.
#
# The third policy is bfloat16, which has the range of a float32 and eight bits of mantissa. It
# does not overflow where float16 does and it is eight times less accurate everywhere else.

POLICIES = ("everything wide", "everything narrow", "products only", "brain float")


@dataclass
class PrecisionPlan:
    """Which type each node's inputs should be cast to."""

    policy: str
    narrow: DType = FLOAT16
    assignments: dict[str, DType] = field(default_factory=dict)

    @property
    def narrow_nodes(self) -> int:
        """Nodes running in the narrow type."""
        return sum(1 for value in self.assignments.values() if value is self.narrow)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "policy": self.policy,
            "narrow": self.narrow.name,
            "nodes": len(self.assignments),
            "narrow_nodes": self.narrow_nodes,
        }


def plan_for(graph: Graph, policy: str, narrow: DType = FLOAT16) -> PrecisionPlan:
    """Assign a type to every node under a named policy.

    Four policies rather than a knob. A knob suggests the space in between is meaningful and it
    is not: what matters is which categories of operation are allowed to narrow, and there are
    only a few groupings of those worth having.
    """
    if policy not in POLICIES:
        raise ConfigError(f"unknown policy {policy!r}, expected one of {list(POLICIES)}")
    plan = PrecisionPlan(policy=policy, narrow=BFLOAT16 if policy == "brain float" else narrow)

    for node in graph.nodes:
        if policy == "everything wide":
            plan.assignments[node.name] = FLOAT32
        elif policy == "products only":
            plan.assignments[node.name] = plan.narrow if node.op is ops.MATMUL else FLOAT32
        else:
            plan.assignments[node.name] = plan.narrow
    return plan


def apply_plan(graph: Graph, plan: PrecisionPlan) -> Graph:
    """Rebuild a graph with casts inserted where the type changes.

    A cast goes in wherever an operand is not already the type the node wants. That is more
    casts than a real compiler would emit, because a real one would propagate a type through a
    run of nodes that agree, and counting them here is the point: the cast count is what the
    propagation would be saving.
    """
    builder = Builder()
    mapping: dict[str, str] = {}
    kinds: dict[str, DType] = {}

    for value in graph.inputs:
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=value.name,
        )
        kinds[value.name] = value.dtype

    for node in graph.nodes:
        wanted = plan.assignments.get(node.name, FLOAT32)
        if node.op is ops.CONSTANT:
            mapping[node.name] = builder.constant(float(node.attrs["value"]), dtype=wanted)
            kinds[node.name] = wanted
            continue

        operands = []
        for name in node.inputs:
            current = mapping[name]
            if kinds[name] is not wanted:
                current = builder.cast(current, wanted)
            operands.append(current)
        produced = builder.apply(node.op, *operands, **node.attrs)
        mapping[node.name] = produced
        kinds[node.name] = builder.dtype_of(produced)

    outputs = []
    for name in graph.outputs:
        current = mapping[name]
        if kinds[name] is not FLOAT32:
            current = builder.cast(current, FLOAT32)
        outputs.append(current)
    return builder.finish(*outputs)


def compile_with(graph: Graph, policy: str) -> Graph:
    """A graph rewritten under one policy."""
    return apply_plan(graph, plan_for(graph, policy))


def cast_count(graph: Graph) -> int:
    """How many conversions a rewritten graph holds."""
    return sum(1 for node in graph.nodes if node.op is ops.CAST)


def bytes_of_intermediates(graph: Graph) -> int:
    """What every value in a graph would cost to hold at once.

    A ceiling rather than a plan. The memory planner produces the real number and this one is
    what the precision decision moves, so it is the right thing to compare across policies even
    though nothing would ever allocate it.
    """
    total = 0
    for node in graph.nodes:
        shape = node.output.shape
        if not shape.is_static:
            raise PassError(f"{node.name} has a symbolic shape and cannot be sized")
        total += shape.elements * node.output.dtype.bytes
    return total


def measure_policy(graph: Graph, policy: str, *, seed: int = 0) -> dict:
    """One policy's error and memory against running everything wide."""
    reference = compile_with(graph, "everything wide")
    rewritten = compile_with(graph, policy)
    feeds = random_feeds(graph, positive=True, seed=seed)

    exact = run(reference, feeds)[0]
    result = run(rewritten, feeds)[0]
    finite = torch.isfinite(result).all()
    gap = float((result - exact).abs().max()) if finite else float("inf")
    scale = float(exact.abs().max())
    return {
        "policy": policy,
        "finite": bool(finite),
        "relative_gap": gap / scale if scale and finite else gap,
        "bytes": bytes_of_intermediates(rewritten),
        "casts": cast_count(rewritten),
    }


def compare_policies(graph: Graph | None = None) -> list[dict]:
    """Every policy on one graph.

    Run on a softmax by default, because a softmax is where the range limit of half precision
    would show if anything showed it. It does not show: the row maximum is subtracted before the
    exponential, every shifted score is at most zero, and the exponential of a negative number
    is comfortably inside float16. The protection a softmax already has against overflow in
    float32 is the same protection it needs in float16.
    """
    target = graph if graph is not None else softmax_graph()
    return [measure_policy(target, policy) for policy in POLICIES]


def half_precision_breaks_an_unshifted_softmax(scale: float = 1.0) -> dict:
    """The overflow, produced rather than described.

    The largest value float16 can hold is sixty five thousand and the exponential of twelve is
    a hundred and sixty thousand. float32 holds it without trouble. So an exponential of raw
    scores is finite in one type and infinite in the other, and the subtraction of the row
    maximum that every softmax does is what keeps the difference from mattering.
    """
    values = torch.tensor([0.0, 12.0, 24.0]) * scale
    narrow = values.to(torch.float16).exp()
    wide = values.exp()
    shifted = (values - values.max()).to(torch.float16).exp()
    return {
        "largest_input": float(values.max()),
        "narrow_is_finite": bool(torch.isfinite(narrow).all()),
        "wide_is_finite": bool(torch.isfinite(wide).all()),
        "shifted_is_finite": bool(torch.isfinite(shifted).all()),
        "float16_ceiling": float(torch.finfo(torch.float16).max),
    }


def where_float16_runs_out(
    values: Sequence[float] = (4.0, 8.0, 11.0, 12.0, 16.0),
) -> list[dict]:
    """The input at which an exponential stops fitting in half precision.

    Between eleven and twelve, which is a number worth carrying around. Any graph that
    exponentiates something a model produced without subtracting a maximum first is one score of
    twelve away from an infinity, and models produce scores of twelve.
    """
    if not values:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for value in values:
        narrow = torch.tensor([value], dtype=torch.float16).exp()
        rows.append(
            {
                "input": value,
                "finite": bool(torch.isfinite(narrow).all()),
                "result": float(narrow) if torch.isfinite(narrow).all() else float("inf"),
            }
        )
    return rows


def brain_float_does_not_overflow(
    values: Sequence[float] = (4.0, 12.0, 40.0, 88.0),
) -> list[dict]:
    """The same sweep in bfloat16, which has the range of a float32.

    It never overflows over the range where float16 does, and it is much less accurate at every
    point. Eight bits of mantissa is about two decimal digits, so the exponential of four comes
    back with an error of half a percent where float16 gives four thousandths.
    """
    if not values:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for value in values:
        wide = torch.tensor([value], dtype=torch.float32).exp()
        brain = torch.tensor([value], dtype=torch.bfloat16).exp().to(torch.float32)
        half = torch.tensor([value], dtype=torch.float16).exp().to(torch.float32)
        rows.append(
            {
                "input": value,
                "brain_float_finite": bool(torch.isfinite(brain).all()),
                "float16_finite": bool(torch.isfinite(half).all()),
                "brain_float_error": float((brain - wide).abs() / wide),
            }
        )
    return rows


def the_range_and_the_mantissa_trade(count: int = 4096, *, seed: int = 0) -> dict:
    """What each narrow type costs on values neither of them overflows on.

    float16 has three more bits of mantissa than bfloat16 and eight fewer bits of exponent. On
    ordinary values that makes it eight times more accurate; on values a model actually produces
    at the wrong moment it makes it infinite. The two numbers here are the whole choice.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(count, generator=generator)
    half = (values.to(torch.float16).to(torch.float32) - values).abs()
    brain = (values.to(torch.bfloat16).to(torch.float32) - values).abs()
    scale = values.abs().mean()
    return {
        "float16_error": round(float(half.mean() / scale), 6),
        "brain_float_error": round(float(brain.mean() / scale), 6),
        "ratio": round(float(brain.mean() / half.mean()), 3),
    }


def reductions_stay_wide_whatever_the_policy(graph: Graph | None = None) -> dict:
    """Whether a policy can force a reduction into half precision, which it cannot.

    A sum cannot, a maximum can, and the difference is exactly right. ir/dtype.py widens an
    accumulator because a sum of many narrow values stalls; a maximum has no accumulator and
    never grows past the largest thing it read, so there is nothing to widen it for. The
    aggressive policy asks for both to be narrow and gets one.
    """
    target = graph if graph is not None else softmax_graph()
    rewritten = compile_with(target, "everything narrow")
    reductions = [node for node in rewritten.nodes if node.op.category == ops.REDUCTION]
    return {
        "reductions": len(reductions),
        "wide_outputs": sum(1 for node in reductions if node.output.dtype is FLOAT32),
    }


def what_summing_in_half_precision_would_do(count: int = 4096) -> dict:
    """The failure the widening prevents, produced by adding one value at a time.

    Torch does not reproduce it with a plain sum, because its reduction is blocked and the
    blocking is itself a fix for this. Adding sequentially is what a naive kernel does, and the
    running total stalls: past two thousand and forty eight, one is below the last bit of the
    accumulator and the sum stops moving entirely.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    running = torch.zeros((), dtype=torch.float16)
    one = torch.ones((), dtype=torch.float16)
    stalled_at = 0
    for index in range(count):
        before = float(running)
        running = running + one
        if float(running) == before and not stalled_at:
            stalled_at = index
    return {
        "expected": count,
        "sequential_in_half_precision": float(running),
        "blocked_in_half_precision": float(torch.ones(count, dtype=torch.float16).sum()),
        "stalled_at": stalled_at,
    }


def where_the_accumulator_stalls() -> dict:
    """The exact value at which adding one to a half precision total does nothing.

    Two thousand and forty eight, which is two to the eleventh, which is one more than the
    mantissa has bits. Above it the gap between representable numbers is two, so adding one
    rounds to the number that was already there.
    """
    running = torch.tensor(2048.0, dtype=torch.float16)
    one = torch.ones((), dtype=torch.float16)
    below = torch.tensor(1024.0, dtype=torch.float16)
    return {
        "two_thousand_and_forty_eight_plus_one": float(running + one),
        "one_thousand_and_twenty_four_plus_one": float(below + one),
        "mantissa_bits": 11,
    }


def bytes_without_the_casts(graph: Graph) -> int:
    """What the values would cost if every conversion were folded into its consumer.

    A cast is a copy in this IR and a fused operand read in any real backend. Counting it as a
    materialised value is honest about what the graph says and wrong about what would run, so
    both numbers are here and the difference between them is what the folding is worth.
    """
    total = 0
    for node in graph.nodes:
        if node.op is ops.CAST:
            continue
        shape = node.output.shape
        if not shape.is_static:
            raise PassError(f"{node.name} has a symbolic shape and cannot be sized")
        total += shape.elements * node.output.dtype.bytes
    return total


def memory_saved(graph: Graph | None = None) -> dict:
    """What each policy is worth in bytes, counted both ways.

    Counting the casts, every narrow policy costs more than the wide one, by a factor of three
    on an mlp. Not counting them, the mixed policy saves a third and the aggressive one saves
    close to half. Which of those a compiler gets depends entirely on whether it folds a
    conversion into the operation that reads it, and that is a backend decision rather than a
    policy decision.
    """
    target = graph if graph is not None else mlp_graph()
    rows = {}
    wide_with = bytes_of_intermediates(compile_with(target, "everything wide"))
    wide_without = bytes_without_the_casts(compile_with(target, "everything wide"))
    for policy in POLICIES:
        rewritten = compile_with(target, policy)
        rows[policy] = {
            "with_casts": round(wide_with / bytes_of_intermediates(rewritten), 4),
            "without_casts": round(wide_without / bytes_without_the_casts(rewritten), 4),
        }
    return rows


def the_casts_eat_the_saving(graph: Graph | None = None) -> dict:
    """The two accountings side by side, for the policy worth having."""
    rows = memory_saved(graph)
    return {
        "counting_the_casts": rows["products only"]["with_casts"],
        "folding_them_in": rows["products only"]["without_casts"],
        "folding_is_what_saves": rows["products only"]["without_casts"] > 1.0,
    }


def products_only_is_the_useful_policy(graph: Graph | None = None) -> dict:
    """The policy comparison reduced to the two numbers that decide it.

    Narrowing only the products keeps the answer to within five ten thousandths on an mlp, and
    it is the only policy that touches the weights, which are most of what a model holds. Run on
    an mlp rather than a softmax, because a softmax has no product in it and the policy would be
    measured doing nothing.
    """
    target = graph if graph is not None else mlp_graph()
    rows = {row["policy"]: row for row in compare_policies(target)}
    return {
        "products_only_gap": rows["products only"]["relative_gap"],
        "products_only_finite": rows["products only"]["finite"],
        "everything_narrow_finite": rows["everything narrow"]["finite"],
        "brain_float_gap": rows["brain float"]["relative_gap"],
    }


def casts_inserted(graph: Graph | None = None) -> list[dict]:
    """How many conversions each policy adds.

    The mixed policy adds the most, which is the cost of mixing: every boundary between a narrow
    node and a wide one is a conversion, and a policy that narrows one op in a graph creates two
    boundaries around it. A uniform policy has boundaries only at the edges.
    """
    target = graph if graph is not None else mlp_graph()
    return [
        {"policy": row["policy"], "casts": row["casts"]} for row in compare_policies(target)
    ]


def mixing_costs_the_most_conversions(graph: Graph | None = None) -> dict:
    """Whether the mixed policy really adds more casts than either uniform one."""
    rows = {row["policy"]: row["casts"] for row in casts_inserted(graph)}
    return {
        "wide": rows["everything wide"],
        "narrow": rows["everything narrow"],
        "mixed": rows["products only"],
        "mixed_is_the_most": rows["products only"]
        > max(rows["everything wide"], rows["everything narrow"]),
    }


def compare_graphs() -> list[dict]:
    """The mixed policy on each fixture, so the answer is not one graph's answer.

    A layernorm is the one to watch. It divides by a standard deviation, and a standard
    deviation of a nearly constant row is a small number, so a graph that narrows anything near
    that division has a chance of dividing by zero that the wide version does not.
    """
    rows = []
    for label, graph in (
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        row = measure_policy(graph, "products only")
        row["graph"] = label
        rows.append(row)
    return rows


def every_fixture_survives_the_mixed_policy(tolerance: float = 1e-2) -> bool:
    """Whether narrowing the products is safe on all of them."""
    return all(row["finite"] and row["relative_gap"] <= tolerance for row in compare_graphs())
