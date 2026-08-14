from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, layernorm_graph
from tgc.ir.dtype import BFLOAT16, FLOAT16, FLOAT32, FLOAT64, DType
from tgc.ir.graph import Graph, Node
from tgc.verify.reference import interpret, random_feeds, run, to_torch

# How far a small error at the input has travelled by the time it reaches the output.
#
# Every operation either shrinks a relative error, leaves it alone, or multiplies it. Addition
# of same signed values leaves it alone. Multiplication adds the two relative errors.
# Subtraction of two nearly equal values divides by how much they cancel, which is unbounded,
# and is the only one of these that turns a rounding error into a wrong answer.
#
# The point of writing it down is to decide where precision can be dropped. A node whose
# amplification is one is a node whose output is no worse than its input, and narrowing it to
# half precision costs one rounding. A node sitting on a cancellation is a node where the
# rounding it introduces gets multiplied by whatever the cancellation factor is, and half
# precision there produces an answer with no correct digits at all.
#
# Both the predicted amplification and the measured one are here, because a predicted
# condition number nobody checks is a comment with arithmetic in it.

EPSILON = {
    FLOAT16.name: 2.0**-11,
    BFLOAT16.name: 2.0**-8,
    FLOAT32.name: 2.0**-24,
    FLOAT64.name: 2.0**-53,
}


def machine_epsilon(dtype: DType) -> float:
    """The largest relative gap between neighbouring values in a type."""
    if dtype.name not in EPSILON:
        raise ConfigError(f"no epsilon known for {dtype}")
    return EPSILON[dtype.name]


def significant_digits(dtype: DType) -> float:
    """Roughly how many decimal digits a type carries."""
    return -math.log10(machine_epsilon(dtype))


@dataclass
class Amplification:
    """How much one operation multiplies a relative error."""

    node: str
    factor: float
    reason: str = ""

    def __post_init__(self) -> None:
        if self.factor < 0:
            raise ConfigError(f"{self.node} cannot amplify by {self.factor}")

    @property
    def is_dangerous(self) -> bool:
        """Whether the operation makes the error meaningfully worse."""
        return self.factor > 10.0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "node": self.node,
            "factor": round(self.factor, 4),
            "dangerous": self.is_dangerous,
            "reason": self.reason,
        }


def cancellation_factor(left: torch.Tensor, right: torch.Tensor) -> float:
    """How much a subtraction magnifies the relative error of its operands.

    The ratio of the larger operand to the difference. Two values agreeing in their first
    seven digits produce a difference whose relative error is ten million times theirs, and
    float32 only has seven digits to give.
    """
    difference = (left - right).abs()
    largest = torch.maximum(left.abs(), right.abs())
    safe = torch.where(difference > 0, difference, torch.ones_like(difference))
    ratio = torch.where(difference > 0, largest / safe, torch.ones_like(difference))
    return float(ratio.max())


def node_amplification(node: Node, environment: dict[str, torch.Tensor]) -> Amplification:
    """The factor one node applies to the relative error arriving at it."""
    name = node.op.name
    if name == "sub":
        left = environment[node.inputs[0]]
        right = environment[node.inputs[1]]
        factor = cancellation_factor(left, right)
        return Amplification(
            node=node.name, factor=factor, reason="cancellation between nearby values"
        )
    if name == "add":
        left = environment[node.inputs[0]]
        right = environment[node.inputs[1]]
        opposite = (left.sign() != right.sign()) & (left != 0) & (right != 0)
        if bool(opposite.any()):
            return Amplification(
                node=node.name,
                factor=cancellation_factor(left, -right),
                reason="addition of opposite signs is a subtraction",
            )
        return Amplification(node=node.name, factor=1.0, reason="same signs, no cancellation")
    if name in ("mul", "div"):
        return Amplification(node=node.name, factor=2.0, reason="the relative errors add")
    if name == "exp":
        magnitude = float(environment[node.inputs[0]].abs().max())
        return Amplification(
            node=node.name, factor=max(1.0, magnitude), reason="the exponent scales the error"
        )
    if name in ("sum", "mean"):
        source = environment[node.inputs[0]]
        total = source.sum().abs()
        absolute = source.abs().sum()
        factor = float(absolute / total) if float(total) > 0 else 1.0
        return Amplification(
            node=node.name,
            factor=factor,
            reason="a sum that cancels amplifies what it cancelled",
        )
    if name in ("sqrt", "log"):
        return Amplification(node=node.name, factor=0.5, reason="the error is halved")
    return Amplification(node=node.name, factor=1.0, reason="no amplification")


def predicted_amplifications(
    graph: Graph, feeds: dict[str, torch.Tensor] | None = None
) -> list[Amplification]:
    """The predicted factor for every node in a graph."""
    supplied = feeds if feeds is not None else random_feeds(graph, positive=True)
    environment = interpret(graph, supplied)
    return [
        node_amplification(node, environment) for node in graph.nodes if not node.op.is_leaf
    ]


def worst_amplification(graph: Graph, feeds: dict[str, torch.Tensor] | None = None) -> float:
    """The largest factor any node applies."""
    factors = predicted_amplifications(graph, feeds)
    return max((item.factor for item in factors), default=1.0)


def dangerous_nodes(graph: Graph, feeds: dict[str, torch.Tensor] | None = None) -> list[str]:
    """Nodes where a rounding error gets multiplied rather than carried."""
    return [item.node for item in predicted_amplifications(graph, feeds) if item.is_dangerous]


def measured_amplification(
    graph: Graph, feeds: dict[str, torch.Tensor] | None = None, *, relative: float = 1e-6
) -> float:
    """Perturb one input at a time and see how much the output moves.

    The empirical version, and the perturbation has to be independent per input. Scaling every
    input by the same factor passes straight through a subtraction: both operands move
    together, the difference moves by the same relative amount, and the measurement reports an
    amplification of one on a graph whose true condition number is eight million. Nudging one
    operand and leaving the other alone is what a rounding error actually does.
    """
    if relative <= 0:
        raise ConfigError(f"the perturbation has to be positive, got {relative}")
    supplied = feeds if feeds is not None else random_feeds(graph, positive=True)
    base = run(graph, supplied)

    worst = 0.0
    for target in supplied:
        nudged = dict(supplied)
        nudged[target] = supplied[target] * (1.0 + relative)
        moved = run(graph, nudged)
        for before, after in zip(base, moved, strict=True):
            scale = float(before.abs().max())
            if scale == 0:
                continue
            worst = max(worst, float((after - before).abs().max()) / scale)
    return worst / relative


def cancelling_graph(gap: float = 1e-7) -> tuple[Graph, dict[str, torch.Tensor]]:
    """Two values that agree to seven digits, subtracted.

    The shape that turns a rounding error into a wrong answer. Nothing here is contrived: a
    variance computed as the mean of squares minus the square of the mean does exactly this,
    which is why nobody computes it that way twice.
    """
    if gap <= 0:
        raise ConfigError(f"the gap has to be positive, got {gap}")
    builder = Builder()
    left = builder.input([4], name="left")
    right = builder.input([4], name="right")
    graph = builder.finish(builder.sub(left, right))
    feeds = {
        "left": torch.full((4,), 1.0),
        "right": torch.full((4,), 1.0 - gap),
    }
    return graph, feeds


def benign_graph() -> tuple[Graph, dict[str, torch.Tensor]]:
    """The same shape with operands that do not cancel."""
    builder = Builder()
    left = builder.input([4], name="left")
    right = builder.input([4], name="right")
    graph = builder.finish(builder.sub(left, right))
    feeds = {"left": torch.full((4,), 5.0), "right": torch.full((4,), 1.0)}
    return graph, feeds


def compare_conditioning() -> list[dict]:
    """The cancelling graph against the benign one, predicted and measured.

    Same three nodes, same operation, two sets of numbers. The cancelling one amplifies a
    relative error by ten million and the benign one by one and a quarter, and both the
    prediction and the measurement say so.
    """
    rows = []
    for label, (graph, feeds) in (
        ("cancelling", cancelling_graph()),
        ("benign", benign_graph()),
    ):
        rows.append(
            {
                "graph": label,
                "predicted": round(worst_amplification(graph, feeds), 2),
                "measured": round(measured_amplification(graph, feeds), 2),
            }
        )
    return rows


@dataclass
class PrecisionPlan:
    """Which nodes may be computed in a narrower type."""

    narrowed: list[str] = field(default_factory=list)
    kept_wide: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Nodes narrowed."""
        return len(self.narrowed)

    @property
    def share(self) -> float:
        """Fraction of the considered nodes that were narrowed."""
        total = self.count + len(self.kept_wide)
        if total == 0:
            return 0.0
        return self.count / total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "narrowed": self.count,
            "kept_wide": len(self.kept_wide),
            "share": round(self.share, 4),
        }


def plan_precision(
    graph: Graph,
    feeds: dict[str, torch.Tensor] | None = None,
    *,
    narrow: DType = FLOAT16,
    budget: float = 1e-3,
) -> PrecisionPlan:
    """Decide which nodes can drop to a narrower type without breaking the budget.

    A node narrows when the rounding it introduces, multiplied by the amplification it sits
    on, stays inside the budget. Sitting on a cancellation of ten million means half precision
    contributes an error of five, which is not a rounding detail, and the plan refuses it with
    the reason attached.
    """
    if budget <= 0:
        raise ConfigError(f"the budget has to be positive, got {budget}")
    supplied = feeds if feeds is not None else random_feeds(graph, positive=True)
    rounding = machine_epsilon(narrow)

    plan = PrecisionPlan()
    for item in predicted_amplifications(graph, supplied):
        contributed = rounding * item.factor
        if contributed <= budget:
            plan.narrowed.append(item.node)
        else:
            plan.kept_wide[item.node] = (
                f"{item.reason}, so {narrow.name} would contribute {contributed:.3g}"
            )
    return plan


def compare_precision_budgets(
    graph: Graph | None = None, budgets: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
) -> list[dict]:
    """How much of a graph can be narrowed at each error budget."""
    target = graph or layernorm_graph()
    if not budgets:
        raise ConfigError("there is nothing to compare")
    feeds = random_feeds(target, positive=True)
    rows = []
    for budget in budgets:
        plan = plan_precision(target, feeds, budget=budget)
        row = plan.as_dict()
        row["budget"] = budget
        rows.append(row)
    return rows


def compare_narrow_types(graph: Graph | None = None, budget: float = 1e-3) -> list[dict]:
    """The same budget under float16 and bfloat16.

    Same width and eight times the rounding, because bfloat16 spends its bits on range rather
    than on precision. A graph that narrows cleanly to float16 may not narrow at all to
    bfloat16, which is the trade that choice actually makes.
    """
    target = graph or layernorm_graph()
    feeds = random_feeds(target, positive=True)
    rows = []
    for narrow in (FLOAT16, BFLOAT16):
        plan = plan_precision(target, feeds, narrow=narrow, budget=budget)
        row = plan.as_dict()
        row["type"] = narrow.name
        row["epsilon"] = machine_epsilon(narrow)
        rows.append(row)
    return rows


def accumulate_in(values: torch.Tensor, dtype: DType) -> float:
    """Sum a tensor one element at a time in a given type.

    Written out rather than delegated, because torch widens its accumulator for sum and the
    whole question here is what happens when nobody does.
    """
    total = torch.zeros((), dtype=to_torch(dtype))
    for value in values.to(to_torch(dtype)):
        total = total + value
    return float(total)


def accumulator_comparison(count: int = 4096) -> list[dict]:
    """Summing the same values in three widths, one element at a time.

    The running total grows until each new addend falls below its last bit and the sum stops
    moving. In float16 that happens well before the end, which is why the type rules widen
    every reduction and why the interpreter had to agree with them.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    values = torch.full((count,), 1.0)
    rows = []
    for dtype in (FLOAT16, BFLOAT16, FLOAT32):
        total = accumulate_in(values, dtype)
        rows.append(
            {
                "dtype": dtype.name,
                "sum": total,
                "expected": float(count),
                "relative_error": round(abs(total - count) / count, 6),
            }
        )
    return rows


def variance_two_ways(scale: float = 1e4, count: int = 1000) -> dict:
    """Variance computed by cancellation and by centring first.

    The textbook example, and it is textbook because it keeps happening. The mean of squares
    minus the square of the mean subtracts two large nearly equal numbers, so at a scale of ten
    thousand it loses every significant digit and can even return a negative variance.
    """
    if count < 2:
        raise ConfigError("a variance needs at least two values")
    generator = torch.Generator().manual_seed(0)
    values = (torch.randn(count, generator=generator) + scale).to(torch.float32)

    naive = float((values * values).mean() - values.mean() ** 2)
    centred = values - values.mean()
    stable = float((centred * centred).mean())
    exact = float(((values.double() - values.double().mean()) ** 2).mean())
    return {
        "naive": naive,
        "stable": stable,
        "exact": exact,
        "naive_error": abs(naive - exact) / exact,
        "stable_error": abs(stable - exact) / exact,
        "naive_is_negative": naive < 0,
    }


def elementwise_ops_are_benign(graph: Graph) -> bool:
    """Whether every elementwise node in a graph has an amplification of one."""
    feeds = random_feeds(graph, positive=True)
    environment = interpret(graph, feeds)
    for node in graph.nodes:
        if node.op is not ops.RELU and node.op is not ops.TANH:
            continue
        if node_amplification(node, environment).factor != 1.0:
            return False
    return True
