from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph
from tgc.verify.reference import outputs_agree, random_feeds, run

# Writing rewrites down as patterns rather than as code.
#
# passes/algebraic.py spells each simplification out: a loop over the nodes, a check on the
# operation, a check on the operands, a rebuild. That is fine for a dozen rules and it stops
# being fine somewhere after that, because every rule repeats the same walk and the same rebuild
# and the differences between them get buried in it.
#
# So this file separates the two. A pattern says what a rewrite matches, a replacement says what
# it produces, and one engine does the walking. The rules that come out are two lines each and
# read as what they mean, which is the point: a rewrite that is hard to read is a rewrite nobody
# audits.
#
# Two things fall out of building it that are worth more than the rules themselves.
#
# A pattern language makes unsafe rewrites easy to write. The exponential of a logarithm is the
# identity in exact arithmetic and is not in floating point, and it is two lines here just like
# the safe ones. So every rule carries a flag saying whether it preserves the answer, and the
# engine will not apply an inexact one unless it is asked to. The measurement below says what
# that one costs.
#
# And an engine that runs to a fixed point can be given rules that never reach one. Doubling
# written as a sum and a sum of equal terms written as a doubling are both reasonable rewrites
# and together they are an infinite loop. The round limit catches it and names the rules, which
# is the only useful thing to do about a pair of rules that undo each other.

WILDCARD = "_"


@dataclass(frozen=True)
class Pattern:
    """A shape of expression, as an operation and its operands.

    An operand is either another pattern or a name that binds to whatever is there. A name
    appearing twice has to bind to the same value, which is what expresses a rule like the
    subtraction of a value from itself without a separate equality check.
    """

    op: str
    operands: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not self.op:
            raise ConfigError("a pattern needs an operation")
        for operand in self.operands:
            if not isinstance(operand, (Pattern, str)):
                raise ConfigError(f"{operand!r} is not a pattern or a name")

    @property
    def names(self) -> tuple[str, ...]:
        """Every binding name the pattern uses, in order."""
        found: list[str] = []
        for operand in self.operands:
            if isinstance(operand, str):
                if operand not in found:
                    found.append(operand)
                continue
            for name in operand.names:
                if name not in found:
                    found.append(name)
        return tuple(found)

    def __str__(self) -> str:
        parts = [str(operand) for operand in self.operands]
        return f"{self.op}({', '.join(parts)})"


def literal(value: float) -> Pattern:
    """A pattern matching a constant of a given value."""
    return Pattern(op="constant", operands=(str(value),))


def match(graph: Graph, name: str, pattern: Pattern) -> dict[str, str] | None:
    """Try a pattern against a value, returning the bindings or nothing.

    Nothing rather than an exception, because a failed match is the ordinary case: an engine
    trying twelve rules against forty nodes fails four hundred and seventy times per round, and
    a failure has to be cheap and quiet.
    """
    node = graph.producer_of(name)
    if node is None or node.op.name != pattern.op:
        return None
    if pattern.op == "constant":
        wanted = pattern.operands[0]
        checkable = isinstance(wanted, str) and wanted != WILDCARD
        if checkable and float(node.attrs["value"]) != float(wanted):
            return None
        return {}
    if len(pattern.operands) != len(node.inputs):
        return None

    bindings: dict[str, str] = {}
    for operand, value in zip(pattern.operands, node.inputs, strict=True):
        if isinstance(operand, str):
            if operand == WILDCARD:
                continue
            if operand in bindings and bindings[operand] != value:
                return None
            bindings[operand] = value
            continue
        inner = match(graph, value, operand)
        if inner is None:
            return None
        for key, bound in inner.items():
            if key in bindings and bindings[key] != bound:
                return None
            bindings[key] = bound
    return bindings


Replacement = Callable[[Builder, dict[str, str]], str]


@dataclass
class Rule:
    """One rewrite: a pattern, a replacement, and whether it changes the answer."""

    name: str
    pattern: Pattern
    replacement: Replacement
    exact: bool = True

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"rule": self.name, "pattern": str(self.pattern), "exact": self.exact}


def _keep(_builder: Builder, bindings: dict[str, str]) -> str:
    """The replacement for a rule that returns one of its own operands."""
    return bindings["x"]


ADD_ZERO = Rule(
    name="x plus zero",
    pattern=Pattern("add", ("x", literal(0.0))),
    replacement=_keep,
)

MUL_ONE = Rule(
    name="x times one",
    pattern=Pattern("mul", ("x", literal(1.0))),
    replacement=_keep,
)

DOUBLE_NEGATION = Rule(
    name="two negations",
    pattern=Pattern("neg", (Pattern("neg", ("x",)),)),
    replacement=_keep,
)

SUB_SELF = Rule(
    name="x minus itself",
    pattern=Pattern("sub", ("x", "x")),
    replacement=lambda builder, bindings: builder.broadcast_to(
        builder.constant(0.0), _static(builder, bindings["x"])
    ),
)

EXP_OF_LOG = Rule(
    name="exp of log",
    pattern=Pattern("exp", (Pattern("log", ("x",)),)),
    replacement=_keep,
    exact=False,
)

SUM_TO_DOUBLE = Rule(
    name="x plus x is twice x",
    pattern=Pattern("add", ("x", "x")),
    replacement=lambda builder, bindings: builder.mul(bindings["x"], builder.constant(2.0)),
)

DOUBLE_TO_SUM = Rule(
    name="twice x is x plus x",
    pattern=Pattern("mul", ("x", literal(2.0))),
    replacement=lambda builder, bindings: builder.add(bindings["x"], bindings["x"]),
)

EXACT_RULES = (ADD_ZERO, MUL_ONE, DOUBLE_NEGATION, SUB_SELF)
INEXACT_RULES = (EXP_OF_LOG,)
LOOPING_RULES = (SUM_TO_DOUBLE, DOUBLE_TO_SUM)


def _static(builder: Builder, name: str) -> list[int]:
    """The sizes of a value the builder already holds."""
    sizes = []
    for size in builder.shape_of(name).sizes:
        if not size.is_static:
            raise PassError(f"cannot build a constant the shape of {name}")
        sizes.append(size.value)
    return sizes


@dataclass
class RewriteReport:
    """What one run of the engine did."""

    rounds: int = 0
    applied: dict[str, int] = field(default_factory=dict)
    settled: bool = True

    @property
    def total(self) -> int:
        """Rewrites performed."""
        return sum(self.applied.values())

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rounds": self.rounds,
            "applied": dict(sorted(self.applied.items())),
            "total": self.total,
            "settled": self.settled,
        }


def apply_once(
    graph: Graph, rules: Sequence[Rule], *, allow_inexact: bool = False
) -> tuple[Graph, dict[str, int]]:
    """One pass over the graph, rewriting whatever matches.

    The rewrite is done during a rebuild rather than in place, so a replacement can emit several
    nodes and the shapes come from the same inference everything else uses. A node whose value
    was rewritten is left out and its consumers are pointed at the replacement.
    """
    usable = [rule for rule in rules if rule.exact or allow_inexact]
    if not usable:
        return graph, {}

    builder = Builder()
    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    for value in graph.inputs:
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=value.name,
        )

    for node in graph.nodes:
        rewritten = None
        for rule in usable:
            bindings = match(graph, node.name, rule.pattern)
            if bindings is None:
                continue
            translated = {key: mapping[value] for key, value in bindings.items()}
            rewritten = rule.replacement(builder, translated)
            counts[rule.name] = counts.get(rule.name, 0) + 1
            break
        if rewritten is not None:
            mapping[node.name] = rewritten
            continue
        if node.op is ops.CONSTANT:
            mapping[node.name] = builder.constant(
                float(node.attrs["value"]), dtype=node.output.dtype
            )
            continue
        mapping[node.name] = builder.apply(
            node.op, *[mapping[name] for name in node.inputs], **node.attrs
        )
    return builder.finish(*[mapping[name] for name in graph.outputs]), counts


def rewrite(
    graph: Graph,
    rules: Sequence[Rule] = EXACT_RULES,
    *,
    allow_inexact: bool = False,
    max_rounds: int = 8,
) -> tuple[Graph, RewriteReport]:
    """Rewrite to a fixed point, or to the round limit.

    The limit is not a safety net, it is the answer for a rule set that does not converge. Two
    rules that undo each other will run forever, and the useful thing to report is which rules
    fired in the last round rather than a stack overflow.
    """
    if max_rounds < 1:
        raise ConfigError(f"there has to be at least one round, got {max_rounds}")
    current = graph
    report = RewriteReport()
    for _ in range(max_rounds):
        report.rounds += 1
        rewritten, counts = apply_once(current, rules, allow_inexact=allow_inexact)
        for name, count in counts.items():
            report.applied[name] = report.applied.get(name, 0) + count
        if not counts:
            return rewritten, report
        current = rewritten
    report.settled = False
    return current, report


def messy_graph(size: int = 8) -> Graph:
    """A graph holding something for every exact rule.

    Built so each rule fires exactly once, which makes the counts in the report readable. A
    fixture where one rule fires four times and another never would make a report that is
    correct and says nothing.
    """
    if size < 1:
        raise ConfigError(f"the size has to be positive, got {size}")
    builder = Builder()
    x = builder.input([size, size], name="x")

    added = builder.add(x, builder.constant(0.0))
    scaled = builder.mul(added, builder.constant(1.0))
    flipped = builder.neg(builder.neg(scaled))
    cancelled = builder.sub(flipped, flipped)
    return builder.finish(builder.add(flipped, cancelled))


def logarithm_graph(size: int = 8) -> Graph:
    """A graph with an exponential of a logarithm in it."""
    builder = Builder()
    x = builder.input([size, size], name="x")
    return builder.finish(builder.exp(builder.log(x)))


def looping_graph(size: int = 8) -> Graph:
    """A graph the two looping rules will pass back and forth forever."""
    builder = Builder()
    x = builder.input([size, size], name="x")
    return builder.finish(builder.add(x, x))


def every_rule_fires(graph: Graph | None = None) -> dict:
    """Which rules matched, and how often."""
    target = graph if graph is not None else messy_graph()
    _, report = rewrite(target)
    return {
        "rules": len(EXACT_RULES),
        "fired": len(report.applied),
        "counts": dict(sorted(report.applied.items())),
    }


def the_rewrite_preserves_the_answer(graph: Graph | None = None, *, seed: int = 0) -> dict:
    """The rewritten graph against the original.

    Bit equality for the exact rules, which is what exact means. A rule that changed the answer
    and was labelled exact would be the worst kind of mistake in this file, because the whole
    design rests on the label.
    """
    target = graph if graph is not None else messy_graph()
    rewritten, report = rewrite(target)
    feeds = random_feeds(target, positive=True, seed=seed)
    return {
        "applied": report.total,
        "nodes_before": len(target.nodes),
        "nodes_after": len(rewritten.nodes),
        "identical": outputs_agree(run(target, feeds), run(rewritten, feeds)),
    }


def the_inexact_rule_is_not_applied_by_default() -> dict:
    """Whether the engine leaves an inexact rewrite alone unless asked.

    It does, and the flag is the whole mechanism. A pattern language makes an unsafe rewrite as
    easy to write as a safe one, so the safety has to live somewhere the writer cannot forget
    it, which is the rule rather than the reviewer.
    """
    graph = logarithm_graph()
    without = rewrite(graph, EXACT_RULES + INEXACT_RULES)[1]
    with_it = rewrite(graph, EXACT_RULES + INEXACT_RULES, allow_inexact=True)[1]
    return {"without_permission": without.total, "with_permission": with_it.total}


def what_the_inexact_rule_costs(*, seed: int = 0) -> dict:
    """The exponential of a logarithm against the value it claims to be.

    Not the same number. The logarithm rounds, the exponential rounds again, and the round trip
    lands within a few parts in ten million of where it started rather than on it. That is small
    and it is not zero, and a rule that removes both operations is removing two roundings the
    program had.
    """
    graph = logarithm_graph()
    rewritten, _ = rewrite(graph, INEXACT_RULES, allow_inexact=True)
    feeds = random_feeds(graph, positive=True, seed=seed)
    before = run(graph, feeds)[0]
    after = run(rewritten, feeds)[0]
    gap = float((before - after).abs().max())
    scale = float(before.abs().max())
    return {
        "identical": bool(torch.equal(before, after)),
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
    }


def rules_that_undo_each_other() -> dict:
    """What the engine does with a pair of rules that never settle.

    It stops at the limit and says so. The report names both rules and the round count, which is
    everything needed to find the pair, and it returns the graph it had rather than raising,
    because a graph rewritten eight times is still a correct graph.
    """
    graph = looping_graph()
    rewritten, report = rewrite(graph, LOOPING_RULES, max_rounds=8)
    feeds = random_feeds(graph, positive=True)
    return {
        "settled": report.settled,
        "rounds": report.rounds,
        "rules_involved": sorted(report.applied),
        "still_correct": outputs_agree(run(graph, feeds), run(rewritten, feeds)),
    }


def either_rule_alone_settles() -> dict:
    """Whether the problem is the pair rather than either rule.

    It is the pair. Each of them on its own reaches a fixed point in two rounds, and together
    they do not reach one at all, which is why a rule set has to be checked as a set and a rule
    that is fine in isolation says nothing about the set it joins.
    """
    graph = looping_graph()
    first = rewrite(graph, (SUM_TO_DOUBLE,), max_rounds=8)[1]
    second = rewrite(graph, (DOUBLE_TO_SUM,), max_rounds=8)[1]
    together = rewrite(graph, LOOPING_RULES, max_rounds=8)[1]
    return {
        "first_settles": first.settled,
        "second_settles": second.settled,
        "together_settles": together.settled,
    }


def a_repeated_name_binds_once() -> dict:
    """Whether a pattern with a name used twice really requires the same value.

    It does, and it is what expresses the subtraction of a value from itself without a separate
    check. A pattern language that allowed the two positions to bind independently would match
    every subtraction and rewrite them all to zero.
    """
    builder = Builder()
    x = builder.input([4, 4], name="x")
    y = builder.input([4, 4], name="y")
    same = builder.sub(x, x)
    different = builder.sub(x, y)
    graph = builder.finish(builder.add(same, different))
    return {
        "matches_the_same_value": match(graph, same, SUB_SELF.pattern) is not None,
        "refuses_two_values": match(graph, different, SUB_SELF.pattern) is None,
    }


def a_wildcard_matches_anything() -> dict:
    """Whether the underscore binds nothing and accepts everything."""
    builder = Builder()
    x = builder.input([4, 4], name="x")
    y = builder.input([4, 4], name="y")
    graph = builder.finish(builder.add(x, y))
    pattern = Pattern("add", (WILDCARD, WILDCARD))
    return {
        "matched": match(graph, graph.outputs[0], pattern) is not None,
        "bound_nothing": match(graph, graph.outputs[0], pattern) == {},
    }


def a_pattern_of_the_wrong_arity_never_matches() -> dict:
    """Whether a pattern with too many operands is refused rather than misread."""
    builder = Builder()
    x = builder.input([4, 4], name="x")
    graph = builder.finish(builder.neg(x))
    return {
        "right_arity": match(graph, graph.outputs[0], Pattern("neg", ("x",))) is not None,
        "wrong_arity": match(graph, graph.outputs[0], Pattern("neg", ("x", "y"))) is None,
    }


def a_pattern_needs_an_operation() -> bool:
    """Whether an empty operation name is refused."""
    try:
        Pattern(op="")
    except ConfigError:
        return True
    return False


def a_pattern_operand_has_to_be_one() -> bool:
    """Whether an operand that is neither a pattern nor a name is refused."""
    try:
        Pattern(op="add", operands=(3,))
    except ConfigError:
        return True
    return False


def rules_are_shorter_than_the_pass() -> dict:
    """How much a rule costs to write, against the hand written version.

    Two lines each here, against the ten or so a rule takes in passes/algebraic.py once the walk
    and the rebuild are counted. That ratio is the argument for the engine and it is not the
    only consideration: the hand written pass can express conditions a pattern cannot, and this
    file has no way to say that a rewrite applies only when a dimension divides by four.
    """
    return {
        "exact_rules": len(EXACT_RULES),
        "inexact_rules": len(INEXACT_RULES),
        "names": [rule.name for rule in EXACT_RULES + INEXACT_RULES],
    }


def the_engine_reaches_a_fixed_point(graph: Graph | None = None) -> dict:
    """How many rounds the exact rules take to settle.

    Two, on a graph built to need chaining: removing the addition of zero exposes the
    multiplication by one, and removing that exposes nothing further. The second round is the
    one that proves the first round finished, which is why a fixed point costs one round more
    than the work does.
    """
    target = graph if graph is not None else messy_graph()
    _, report = rewrite(target)
    return {"rounds": report.rounds, "settled": report.settled, "applied": report.total}


def a_zero_round_limit_is_refused() -> bool:
    """Whether asking for no rounds at all is caught."""
    try:
        rewrite(messy_graph(), max_rounds=0)
    except ConfigError:
        return True
    return False
