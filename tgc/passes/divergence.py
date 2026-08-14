from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph
from tgc.passes.algebraic import ALL_RULES, EXACT_RULES, get_rule, simplify
from tgc.verify.reference import run

# Measuring what the fast rules actually cost.
#
# Every rule in the fast list comes with a note saying how it changes the answer. A note is a
# claim, and a claim in a compiler is worth what somebody can check. This file builds the
# smallest graph each rule fires on, runs it with the rule and without, and reports the gap
# on inputs chosen to make the gap visible.
#
# The inputs are adversarial on purpose, and the result is not a prediction about real data.
# It is an upper bound with a worked example attached, which is what an engineer deciding
# whether to turn the flag on actually needs.


@dataclass
class Divergence:
    """How far one rule moved the answer."""

    rule: str
    baseline: float
    rewritten: float
    produced_nan: bool = False

    @property
    def absolute_gap(self) -> float:
        """How far apart the two answers are."""
        if self.produced_nan:
            return math.inf
        return abs(self.baseline - self.rewritten)

    @property
    def relative_gap(self) -> float:
        """The gap scaled by the size of the answer."""
        if self.produced_nan:
            return math.inf
        scale = max(abs(self.baseline), abs(self.rewritten))
        if scale == 0.0:
            return 0.0
        return self.absolute_gap / scale

    @property
    def is_exact(self) -> bool:
        """Whether the rewrite left the answer alone."""
        return not self.produced_nan and self.baseline == self.rewritten

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rule": self.rule,
            "baseline": self.baseline,
            "rewritten": self.rewritten,
            "relative_gap": self.relative_gap,
            "exact": self.is_exact,
        }


def _mul_by_zero_case() -> tuple[Graph, dict[str, torch.Tensor]]:
    """An infinity multiplied by a literal zero.

    The value is not contrived. An overflowed intermediate reaches a masked position in every
    attention implementation ever written, and the mask is a multiply by zero.
    """
    builder = Builder()
    x = builder.input([1], name="x")
    zero = builder.constant(0.0)
    graph = builder.finish(builder.mul(x, zero))
    return graph, {"x": torch.tensor([float("inf")])}


def _add_then_subtract_case() -> tuple[Graph, dict[str, torch.Tensor]]:
    """A small value plus a large one, then the large one taken away again.

    Catastrophic cancellation in three nodes. The addition rounds the small operand out of
    existence and the subtraction cannot bring it back, so the honest answer is zero and the
    rewritten one is the original value.
    """
    builder = Builder()
    small = builder.input([1], name="small")
    large = builder.input([1], name="large")
    graph = builder.finish(builder.sub(builder.add(small, large), large))
    return graph, {
        "small": torch.tensor([1.0]),
        "large": torch.tensor([1e8]),
    }


def _double_reciprocal_case() -> tuple[Graph, dict[str, torch.Tensor]]:
    """One over one over a value whose two roundings do not cancel.

    Seven, and the choice matters. The round trip is exact for most float32 values and fails
    for about one in seven of them, so a rule tested on the first number somebody typed comes
    back clean and is still not exact.
    """
    builder = Builder()
    x = builder.input([1], name="x")
    graph = builder.finish(builder.reciprocal(builder.reciprocal(x)))
    return graph, {"x": torch.tensor([7.0])}


def _sqrt_squared_case() -> tuple[Graph, dict[str, torch.Tensor]]:
    """A square root multiplied by itself."""
    builder = Builder()
    x = builder.input([1], name="x")
    root = builder.sqrt(x)
    graph = builder.finish(builder.mul(root, root))
    return graph, {"x": torch.tensor([2.0])}


CASES = {
    "mul_by_zero": _mul_by_zero_case,
    "add_then_subtract": _add_then_subtract_case,
    "double_reciprocal": _double_reciprocal_case,
    "sqrt_squared": _sqrt_squared_case,
}


def measure_rule(name: str) -> Divergence:
    """Run one rule's worked example with the rewrite and without it."""
    if name not in CASES:
        raise ConfigError(f"no worked example for {name!r}, expected one of {sorted(CASES)}")
    rule = get_rule(name)
    graph, feeds = CASES[name]()

    baseline = run(graph, feeds)[0]
    rewritten_graph = simplify(graph, rules=(*EXACT_RULES, rule))
    rewritten = run(rewritten_graph, feeds)[0]

    first = float(baseline.reshape(-1)[0])
    second = float(rewritten.reshape(-1)[0])
    return Divergence(
        rule=name,
        baseline=first,
        rewritten=second,
        produced_nan=math.isnan(first) != math.isnan(second),
    )


def measure_divergence(names: Sequence[str] = ()) -> list[dict]:
    """Every fast rule against its worked example.

    Not one of them is exact, which is the point of the file. The magnitudes span everything
    from a last bit disagreement to an answer that is wrong by its entire value, and treating
    those as one category is how a fast math flag gets turned on for the whole compiler.
    """
    wanted = list(names) if names else sorted(CASES)
    return [measure_rule(name).as_dict() for name in wanted]


def worst_relative_gap() -> float:
    """The largest relative disagreement any fast rule produces on its example."""
    gaps = [row["relative_gap"] for row in measure_divergence()]
    finite = [gap for gap in gaps if math.isfinite(gap)]
    return max(finite, default=0.0)


def exact_rules_are_exact() -> list[dict]:
    """The same measurement for the rules that claim to be exact.

    A claim of exactness that nobody checks is a comment. Each of these fires on its own
    worked example and has to come back bit identical.
    """
    rows = []
    for graph, feeds, name in _exact_cases():
        baseline = run(graph, feeds)[0]
        rewritten = run(simplify(graph, rules=EXACT_RULES), feeds)[0]
        rows.append(
            {
                "rule": name,
                "identical": bool(torch.equal(baseline, rewritten)),
                "gap": float((baseline - rewritten).abs().max()),
            }
        )
    return rows


def _exact_cases() -> list[tuple[Graph, dict[str, torch.Tensor], str]]:
    """One graph per exact rule, with an input that would expose a mistake."""
    cases = []

    builder = Builder()
    x = builder.input([4], name="x")
    one = builder.constant(1.0)
    cases.append((builder.finish(builder.mul(x, one)), None, "mul_by_one"))

    builder = Builder()
    x = builder.input([4], name="x")
    one = builder.constant(1.0)
    cases.append((builder.finish(builder.div(x, one)), None, "div_by_one"))

    builder = Builder()
    x = builder.input([4], name="x")
    zero = builder.constant(0.0)
    cases.append((builder.finish(builder.add(x, zero)), None, "add_zero"))

    builder = Builder()
    x = builder.input([4], name="x")
    zero = builder.constant(0.0)
    cases.append((builder.finish(builder.sub(x, zero)), None, "sub_zero"))

    builder = Builder()
    x = builder.input([4], name="x")
    cases.append((builder.finish(builder.neg(builder.neg(x))), None, "double_negation"))

    sample = torch.tensor([1.0, -2.5, 1e-8, 1e8])
    return [(graph, {"x": sample}, name) for graph, _, name in cases]


def reciprocal_round_trip_failure_rate(samples: int = 200_000, seed: int = 0) -> float:
    """How often one over one over x fails to give back x, on ordinary values.

    About one in seven. Which is why the rule is inexact and why testing it on a single
    convenient number says nothing: most values do round back exactly.
    """
    if samples < 1:
        raise ConfigError(f"the sample count must be positive, got {samples}")
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(samples, generator=generator)
    values = values[values.abs() > 1e-3]
    round_tripped = torch.reciprocal(torch.reciprocal(values))
    return float((round_tripped != values).float().mean())


def rule_table() -> list[dict]:
    """Every rule with its exactness claim, for printing."""
    return [rule.as_dict() for rule in ALL_RULES]
