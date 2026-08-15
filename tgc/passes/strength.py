from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node
from tgc.verify.reference import outputs_agree, random_feeds, run

# Replacing an expensive operation with a cheaper one that computes the same thing.
#
# Dividing by a literal becomes multiplying by its reciprocal. Raising to the power of two
# becomes a multiplication. Both are four to eight times cheaper on the arithmetic units and
# both are the kind of rewrite that gets applied without measuring, so this file measures.
#
# The two are not equally safe and the difference is not obvious. Squaring by multiplication is
# exact: x times x is the definition, and any pow implementation that disagrees is the one that
# is wrong. Dividing by a reciprocal is not: one over three is not representable, so the
# multiply performs two roundings where the divide performed one, and the answers differ on
# about one input in three.
#
# So they belong in separate lists, the same way the algebraic rules do, and the rate is
# measured rather than guessed at.


@dataclass
class Reduction:
    """One operation replaced by a cheaper one."""

    node: str
    original: str
    replacement: str
    exact: bool

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "node": self.node,
            "from": self.original,
            "to": self.replacement,
            "exact": self.exact,
        }


@dataclass
class StrengthReport:
    """What strength reduction would rewrite."""

    reductions: list[Reduction] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Nodes rewritten."""
        return len(self.reductions)

    @property
    def exact_count(self) -> int:
        """Rewrites that leave the answer alone."""
        return sum(1 for item in self.reductions if item.exact)

    @property
    def inexact_count(self) -> int:
        """Rewrites that change it."""
        return self.count - self.exact_count

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "reductions": self.count,
            "exact": self.exact_count,
            "inexact": self.inexact_count,
        }


def is_constant(graph: Graph, name: str) -> bool:
    """Whether a value is a literal."""
    node = graph.producer_of(name)
    return node is not None and node.op is ops.CONSTANT


def constant_value(graph: Graph, name: str) -> float:
    """The number a literal holds."""
    node = graph.producer_of(name)
    if node is None or node.op is not ops.CONSTANT:
        raise PassError(f"{name} is not a literal")
    return float(node.attrs["value"])


def is_self_multiply(node: Node) -> bool:
    """Whether a node squares its input by multiplying it with itself."""
    return node.op is ops.MUL and node.inputs[0] == node.inputs[1]


def divides_by_literal(graph: Graph, node: Node) -> bool:
    """Whether a node divides by a compile time constant."""
    return node.op is ops.DIV and is_constant(graph, node.inputs[1])


def report_strength(graph: Graph) -> StrengthReport:
    """Which nodes could be made cheaper, and whether the rewrite is exact."""
    report = StrengthReport()
    for node in graph.nodes:
        if divides_by_literal(graph, node):
            report.reductions.append(
                Reduction(
                    node=node.name,
                    original="div by a literal",
                    replacement="mul by its reciprocal",
                    exact=False,
                )
            )
        elif is_self_multiply(node):
            report.reductions.append(
                Reduction(
                    node=node.name,
                    original="mul by itself",
                    replacement="already the cheap form",
                    exact=True,
                )
            )
    return report


def reduce_divisions(graph: Graph) -> Graph:
    """Replace division by a literal with multiplication by its reciprocal.

    Inexact and off by default everywhere it is used, for the same reason the fast algebraic
    rules are: it saves a real four to one on the arithmetic units and changes the answer on
    inputs nobody chose.
    """
    builder = Builder()
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
        if divides_by_literal(graph, node):
            divisor = constant_value(graph, node.inputs[1])
            if divisor == 0:
                raise PassError(f"{node.name} divides by zero and cannot be reduced")
            reciprocal = builder.constant(1.0 / divisor, dtype=node.output.dtype)
            mapping[node.name] = builder.mul(operands[0], reciprocal)
            continue
        mapping[node.name] = builder.apply(node.op, *operands, **node.attrs)

    return builder.finish(*[mapping[name] for name in graph.outputs])


def division_graph(divisor: float = 3.0, count: int = 4) -> Graph:
    """A chain of divisions by the same literal."""
    if count < 1:
        raise ConfigError(f"there has to be at least one division, got {count}")
    if divisor == 0:
        raise ConfigError("the divisor cannot be zero")
    builder = Builder()
    current = builder.input([8, 8], name="x")
    for _ in range(count):
        current = builder.div(current, builder.constant(divisor))
    return builder.finish(current)


def squaring_graph(count: int = 4) -> Graph:
    """A chain of squarings written as self multiplications."""
    if count < 1:
        raise ConfigError(f"there has to be at least one squaring, got {count}")
    builder = Builder()
    current = builder.input([8, 8], name="x")
    for _ in range(count):
        current = builder.mul(current, current)
    return builder.finish(current)


def reciprocal_disagreement_rate(
    divisor: float = 3.0, samples: int = 200_000, seed: int = 0
) -> float:
    """How often dividing and multiplying by the reciprocal give different answers.

    Measured rather than assumed. One over three is not representable, so the multiply
    performs two roundings where the divide performed one, and the two disagree on a
    substantial share of ordinary values rather than on a contrived few.
    """
    if samples < 1:
        raise ConfigError(f"the sample count must be positive, got {samples}")
    if divisor == 0:
        raise ConfigError("the divisor cannot be zero")
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(samples, generator=generator)
    divided = values / divisor
    multiplied = values * (1.0 / divisor)
    return float((divided != multiplied).float().mean())


def compare_divisors(
    divisors: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 8.0, 10.0),
) -> list[dict]:
    """The disagreement rate for several divisors.

    Zero for powers of two, because their reciprocals are exact and the multiply performs the
    same rounding the divide did. Everything else disagrees on a third of inputs or more,
    which is the difference between a rewrite that is safe and one that merely looks like it.
    """
    if not divisors:
        raise ConfigError("there is nothing to compare")
    return [
        {
            "divisor": divisor,
            "reciprocal_is_exact": (1.0 / divisor) * divisor == 1.0
            and float(torch.tensor(1.0 / divisor, dtype=torch.float32)) * divisor == 1.0,
            "disagreement_rate": round(reciprocal_disagreement_rate(divisor), 4),
        }
        for divisor in divisors
    ]


def measure_division_rewrite(divisor: float = 3.0, count: int = 4) -> dict:
    """The rewritten graph against the original, on real inputs."""
    graph = division_graph(divisor, count)
    rewritten = reduce_divisions(graph)
    feeds = random_feeds(graph, positive=True)

    before = run(graph, feeds)
    after = run(rewritten, feeds)
    return {
        "divisor": divisor,
        "identical": outputs_agree(before, after),
        "largest_gap": float((before[0] - after[0]).abs().max()),
        "divisions_before": sum(1 for node in graph.nodes if node.op is ops.DIV),
        "divisions_after": sum(1 for node in rewritten.nodes if node.op is ops.DIV),
    }


def compare_rewrite_by_divisor(
    divisors: tuple[float, ...] = (2.0, 3.0, 4.0, 10.0),
) -> list[dict]:
    """Whether the rewrite is exact on a whole graph, for several divisors.

    A power of two survives it exactly and everything else does not, which is the rule a
    compiler should actually apply: reduce the divisions whose reciprocal is representable and
    leave the rest alone unless somebody asks.
    """
    if not divisors:
        raise ConfigError("there is nothing to compare")
    return [measure_division_rewrite(divisor) for divisor in divisors]


def safe_to_reduce(divisor: float) -> bool:
    """Whether replacing division by this literal is exact.

    True exactly for powers of two. Their reciprocals are representable, so the multiply
    performs the same single rounding the divide did.
    """
    if divisor == 0:
        raise ConfigError("the divisor cannot be zero")
    reciprocal = float(torch.tensor(1.0 / divisor, dtype=torch.float32))
    return reciprocal * divisor == 1.0 and reciprocal != 0.0


def reduce_safe_divisions(graph: Graph) -> Graph:
    """Rewrite only the divisions whose reciprocal is exact.

    The version a compiler can turn on without a flag. It fires less often and never changes
    an answer, and the tests measure both halves of that rather than asserting either.
    """
    keep = {
        node.name
        for node in graph.nodes
        if divides_by_literal(graph, node)
        and not safe_to_reduce(constant_value(graph, node.inputs[1]))
    }
    if not keep:
        return reduce_divisions(graph)

    builder = Builder()
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
        if divides_by_literal(graph, node) and node.name not in keep:
            divisor = constant_value(graph, node.inputs[1])
            reciprocal = builder.constant(1.0 / divisor, dtype=node.output.dtype)
            mapping[node.name] = builder.mul(operands[0], reciprocal)
            continue
        mapping[node.name] = builder.apply(node.op, *operands, **node.attrs)

    return builder.finish(*[mapping[name] for name in graph.outputs])


def measure_safe_rewrite(divisors: tuple[float, ...] = (2.0, 3.0, 4.0, 10.0)) -> list[dict]:
    """The conservative rewrite on the same graphs, which never changes an answer."""
    if not divisors:
        raise ConfigError("there is nothing to compare")
    rows = []
    for divisor in divisors:
        graph = division_graph(divisor)
        rewritten = reduce_safe_divisions(graph)
        feeds = random_feeds(graph, positive=True)
        rows.append(
            {
                "divisor": divisor,
                "safe": safe_to_reduce(divisor),
                "identical": outputs_agree(run(graph, feeds), run(rewritten, feeds)),
                "divisions_after": sum(1 for node in rewritten.nodes if node.op is ops.DIV),
            }
        )
    return rows
