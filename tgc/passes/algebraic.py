from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.graph import Graph, Node

# Rewriting arithmetic into cheaper arithmetic, and being honest about the difference.
#
# Almost every rule people call an algebraic identity is an identity over the reals and not
# over floating point. Multiplying by zero gives zero unless the other operand was infinite,
# in which case it gives nan. Subtracting a value from itself gives zero on the same
# condition. Adding b and then subtracting it does not give back a, it gives back a with
# however many low bits the addition destroyed.
#
# None of that means the rules are useless. It means they belong behind a flag, and the flag
# has to be off by default, because a compiler that silently swaps exact arithmetic for
# approximate arithmetic has changed the program without telling anybody.
#
# The rules are split into two lists here, and measure_divergence runs both versions on
# inputs chosen to make the difference visible. The claim that a fast rule is fine on real
# data is a claim somebody can check.


@dataclass
class Context:
    """What a rule needs to know about the graph around a node."""

    constants: dict[str, float] = field(default_factory=dict)
    producers: dict[str, Node] = field(default_factory=dict)

    def constant(self, name: str) -> float | None:
        """The literal value behind a name, if it is one."""
        return self.constants.get(name)

    def is_constant(self, name: str, value: float) -> bool:
        """Whether a name holds a particular literal."""
        found = self.constants.get(name)
        return found is not None and found == value

    def producer(self, name: str) -> Node | None:
        """The node that computed a value."""
        return self.producers.get(name)


Match = Callable[[Node, Context], str | None]


@dataclass
class Rule:
    """One rewrite, and whether it changes the answer."""

    name: str
    match: Match
    exact: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("a rule needs a name")
        if not callable(self.match):
            raise ConfigError(f"{self.name} is not callable")
        if not self.exact and not self.note:
            raise ConfigError(f"{self.name} changes the answer and does not say how")

    def as_dict(self) -> dict[str, str | bool]:
        """Flat mapping for logging."""
        return {"rule": self.name, "exact": self.exact, "note": self.note}


def _mul_by_one(node: Node, context: Context) -> str | None:
    """x times one is x."""
    if node.op is not ops.MUL:
        return None
    left, right = node.inputs
    if context.is_constant(right, 1.0):
        return left
    if context.is_constant(left, 1.0):
        return right
    return None


def _div_by_one(node: Node, context: Context) -> str | None:
    """x divided by one is x."""
    if node.op is not ops.DIV:
        return None
    left, right = node.inputs
    return left if context.is_constant(right, 1.0) else None


def _add_zero(node: Node, context: Context) -> str | None:
    """x plus zero is x.

    Exact for every finite and infinite value, and not quite for negative zero: adding
    positive zero to negative zero gives positive zero. Nothing downstream in this compiler
    distinguishes the two, and a rule that claims exactness has to say where it stops.
    """
    if node.op is not ops.ADD:
        return None
    left, right = node.inputs
    if context.is_constant(right, 0.0):
        return left
    if context.is_constant(left, 0.0):
        return right
    return None


def _sub_zero(node: Node, context: Context) -> str | None:
    """x minus zero is x."""
    if node.op is not ops.SUB:
        return None
    left, right = node.inputs
    return left if context.is_constant(right, 0.0) else None


def _double_negation(node: Node, context: Context) -> str | None:
    """Negating twice gives back the original, exactly, including for zero and nan payloads."""
    if node.op is not ops.NEG:
        return None
    inner = context.producer(node.inputs[0])
    if inner is not None and inner.op is ops.NEG:
        return inner.inputs[0]
    return None


def _mul_by_zero(node: Node, context: Context) -> str | None:
    """x times zero is zero, unless x was infinite or nan, in which case it is nan."""
    if node.op is not ops.MUL:
        return None
    left, right = node.inputs
    if context.is_constant(right, 0.0):
        return right
    if context.is_constant(left, 0.0):
        return left
    return None


def _add_then_subtract(node: Node, context: Context) -> str | None:
    """Adding b then subtracting it gives back a, less whatever the addition rounded away."""
    if node.op is not ops.SUB:
        return None
    left, right = node.inputs
    inner = context.producer(left)
    if inner is None or inner.op is not ops.ADD:
        return None
    first, second = inner.inputs
    if second == right:
        return first
    if first == right:
        return second
    return None


def _double_reciprocal(node: Node, context: Context) -> str | None:
    """One over one over x is x, to within two roundings."""
    if node.op is not ops.RECIPROCAL:
        return None
    inner = context.producer(node.inputs[0])
    if inner is not None and inner.op is ops.RECIPROCAL:
        return inner.inputs[0]
    return None


def _sqrt_squared(node: Node, context: Context) -> str | None:
    """The square root of x, squared, is x, to within two roundings."""
    if node.op is not ops.MUL:
        return None
    left, right = node.inputs
    if left != right:
        return None
    inner = context.producer(left)
    if inner is not None and inner.op is ops.SQRT:
        return inner.inputs[0]
    return None


EXACT_RULES: tuple[Rule, ...] = (
    Rule(name="mul_by_one", match=_mul_by_one, exact=True),
    Rule(name="div_by_one", match=_div_by_one, exact=True),
    Rule(
        name="add_zero",
        match=_add_zero,
        exact=True,
        note="exact except that negative zero plus zero is positive zero",
    ),
    Rule(name="sub_zero", match=_sub_zero, exact=True),
    Rule(name="double_negation", match=_double_negation, exact=True),
)

FAST_RULES: tuple[Rule, ...] = (
    Rule(
        name="mul_by_zero",
        match=_mul_by_zero,
        exact=False,
        note="infinity times zero is nan, not zero",
    ),
    Rule(
        name="add_then_subtract",
        match=_add_then_subtract,
        exact=False,
        note="the addition rounds away the low bits of the smaller operand",
    ),
    Rule(
        name="double_reciprocal",
        match=_double_reciprocal,
        exact=False,
        note="two roundings do not cancel",
    ),
    Rule(
        name="sqrt_squared",
        match=_sqrt_squared,
        exact=False,
        note="two roundings do not cancel",
    ),
)

ALL_RULES = EXACT_RULES + FAST_RULES

BY_NAME = {rule.name: rule for rule in ALL_RULES}


def get_rule(name: str) -> Rule:
    """Look up a rule by name."""
    if name not in BY_NAME:
        raise ConfigError(f"unknown rule {name!r}, expected one of {sorted(BY_NAME)}")
    return BY_NAME[name]


def build_context(graph: Graph) -> Context:
    """Gather what the rules need to look at."""
    constants = {
        node.name: float(node.attrs["value"]) for node in graph.nodes if node.op is ops.CONSTANT
    }
    return Context(constants=constants, producers={node.name: node for node in graph.nodes})


@dataclass
class SimplifyReport:
    """What algebraic simplification rewrote."""

    applied: dict[str, str] = field(default_factory=dict)
    inexact_applied: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Nodes replaced by one of their operands."""
        return len(self.applied)

    @property
    def changed_the_answer(self) -> bool:
        """Whether any rule that fires is one that alters the arithmetic."""
        return bool(self.inexact_applied)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "applied": self.count,
            "rules": sorted(set(self.applied.values())),
            "inexact": sorted(set(self.inexact_applied)),
        }


def simplify(graph: Graph, *, rules: tuple[Rule, ...] = EXACT_RULES) -> Graph:
    """Replace nodes that reduce to one of their operands.

    Defaults to the exact rules only. A compiler that silently swaps exact arithmetic for
    approximate arithmetic has changed the program without telling anybody, and the caller
    who wants that has to ask for it by name.
    """
    context = build_context(graph)
    replacement: dict[str, str] = {}
    kept: list[Node] = []

    for node in graph.nodes:
        rewritten = node.replace_inputs(replacement)
        context.producers[node.name] = rewritten
        target = None
        for rule in rules:
            target = rule.match(rewritten, context)
            if target is not None:
                break
        if target is None:
            kept.append(rewritten)
            continue
        replacement[node.name] = replacement.get(target, target)

    outputs = [replacement.get(name, name) for name in graph.outputs]
    return Graph(nodes=kept, inputs=list(graph.inputs), outputs=outputs)


def simplify_fast(graph: Graph) -> Graph:
    """Apply every rule, including the ones that change the answer."""
    return simplify(graph, rules=ALL_RULES)


def report_simplification(
    graph: Graph, *, rules: tuple[Rule, ...] = EXACT_RULES
) -> SimplifyReport:
    """Which rules would fire, without firing them."""
    context = build_context(graph)
    report = SimplifyReport()

    for node in graph.nodes:
        for rule in rules:
            if rule.match(node, context) is not None:
                report.applied[node.name] = rule.name
                if not rule.exact:
                    report.inexact_applied.append(rule.name)
                break
    return report


def inexact_rules_that_fire(graph: Graph) -> list[str]:
    """Which answer changing rules a graph is exposed to."""
    return sorted(set(report_simplification(graph, rules=ALL_RULES).inexact_applied))
