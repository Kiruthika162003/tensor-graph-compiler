from __future__ import annotations

from dataclasses import dataclass, field

from tgc.errors import ConfigError

# The operation set, and the properties every pass asks about.
#
# The properties are the point of the file. Fusion asks whether an op is elementwise, common
# subexpression elimination asks whether it is pure, dead code elimination asks whether
# removing it is observable, and buffer reuse asks whether it reads its input more than once.
# Answering those by name lookup inside each pass means every pass has its own idea of what
# a division is, and the ideas drift.
#
# ELEMENTWISE is the category that earns its keep: an elementwise op reads element i of each
# input and writes element i of its output, which is exactly the condition that lets a chain
# of them run as one loop with no buffer in between.

ELEMENTWISE = "elementwise"
REDUCTION = "reduction"
CONTRACTION = "contraction"
VIEW = "view"
LEAF = "leaf"
SIDE_EFFECT = "side_effect"

CATEGORIES = (ELEMENTWISE, REDUCTION, CONTRACTION, VIEW, LEAF, SIDE_EFFECT)


@dataclass(frozen=True)
class Op:
    """One operation, and what a pass is allowed to assume about it."""

    name: str
    category: str
    arity: int = 1
    commutative: bool = False
    pure: bool = True
    reads_input_once: bool = True
    identity_element: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("an op needs a name")
        if self.category not in CATEGORIES:
            raise ConfigError(f"unknown category {self.category!r} for {self.name}")
        if self.arity < 0:
            raise ConfigError(f"{self.name} cannot take {self.arity} inputs")
        if self.commutative and self.arity != 2:
            raise ConfigError(f"{self.name} is commutative but takes {self.arity} inputs")

    @property
    def is_elementwise(self) -> bool:
        """Whether it reads element i and writes element i.

        The condition for fusing without a buffer. Nothing else in the compiler is allowed to
        decide this for itself.
        """
        return self.category == ELEMENTWISE

    @property
    def is_leaf(self) -> bool:
        """Whether it produces a value without reading one."""
        return self.category == LEAF

    @property
    def can_be_removed_if_unused(self) -> bool:
        """Whether deleting it when nothing reads it is unobservable."""
        return self.pure and self.category != SIDE_EFFECT

    @property
    def can_write_over_input(self) -> bool:
        """Whether the output may share storage with an input.

        Only when the op reads each input element once. An op that reads its input twice,
        which for an elementwise op means the same buffer appears in two argument positions,
        would read a value it has already overwritten.
        """
        return self.is_elementwise and self.reads_input_once

    def __str__(self) -> str:
        return self.name


ADD = Op(name="add", category=ELEMENTWISE, arity=2, commutative=True, identity_element=0.0)
MUL = Op(name="mul", category=ELEMENTWISE, arity=2, commutative=True, identity_element=1.0)
SUB = Op(name="sub", category=ELEMENTWISE, arity=2)
DIV = Op(name="div", category=ELEMENTWISE, arity=2)
MAXIMUM = Op(name="maximum", category=ELEMENTWISE, arity=2, commutative=True)
MINIMUM = Op(name="minimum", category=ELEMENTWISE, arity=2, commutative=True)

NEG = Op(name="neg", category=ELEMENTWISE)
EXP = Op(name="exp", category=ELEMENTWISE)
LOG = Op(name="log", category=ELEMENTWISE)
SQRT = Op(name="sqrt", category=ELEMENTWISE)
TANH = Op(name="tanh", category=ELEMENTWISE)
RELU = Op(name="relu", category=ELEMENTWISE)
SIGMOID = Op(name="sigmoid", category=ELEMENTWISE)
RECIPROCAL = Op(name="reciprocal", category=ELEMENTWISE)
ABS = Op(name="abs", category=ELEMENTWISE)
CAST = Op(name="cast", category=ELEMENTWISE)

# One where the input is positive and zero everywhere else. Nothing a person writes uses this,
# and reverse mode needs it for every operation with a corner: relu, abs, maximum and the max
# reduction all differentiate into an indicator, and without an op for it the gradient of a
# relu has to be spelled as a division that is nan at exactly the point the corner sits.
STEP = Op(name="step", category=ELEMENTWISE)

SUM = Op(name="sum", category=REDUCTION)
MAX = Op(name="max", category=REDUCTION)
MEAN = Op(name="mean", category=REDUCTION)

MATMUL = Op(name="matmul", category=CONTRACTION, arity=2)

# Joining two tensors along an axis and taking a window out of one. Neither is a view in the
# strict sense, since both move data, and both live here because the shape rules are the same
# kind of rule and a pass that reasons about layout has to reason about them together.
CONCAT = Op(name="concat", category=VIEW, arity=2)
SLICE = Op(name="slice", category=VIEW)

RESHAPE = Op(name="reshape", category=VIEW)
TRANSPOSE = Op(name="transpose", category=VIEW)
BROADCAST_TO = Op(name="broadcast_to", category=VIEW)

INPUT = Op(name="input", category=LEAF, arity=0)
CONSTANT = Op(name="constant", category=LEAF, arity=0)

PRINT = Op(name="print", category=SIDE_EFFECT, pure=False)
ASSERT_FINITE = Op(name="assert_finite", category=SIDE_EFFECT, pure=False)

ALL_OPS = (
    ADD,
    MUL,
    SUB,
    DIV,
    MAXIMUM,
    MINIMUM,
    NEG,
    EXP,
    LOG,
    SQRT,
    TANH,
    RELU,
    SIGMOID,
    RECIPROCAL,
    ABS,
    CAST,
    STEP,
    SUM,
    MAX,
    MEAN,
    MATMUL,
    RESHAPE,
    TRANSPOSE,
    BROADCAST_TO,
    CONCAT,
    SLICE,
    INPUT,
    CONSTANT,
    PRINT,
    ASSERT_FINITE,
)

BY_NAME = {op.name: op for op in ALL_OPS}


def get_op(name: str) -> Op:
    """Look up an operation by name."""
    if name not in BY_NAME:
        raise ConfigError(f"unknown op {name!r}")
    return BY_NAME[name]


def elementwise_ops() -> tuple[Op, ...]:
    """Every op a fusion pass is allowed to merge."""
    return tuple(op for op in ALL_OPS if op.is_elementwise)


def reduction_ops() -> tuple[Op, ...]:
    """Every op that collapses a dimension."""
    return tuple(op for op in ALL_OPS if op.category == REDUCTION)


@dataclass
class OpCost:
    """What one element of an operation costs.

    Rough on purpose. A cost model that claims to predict nanoseconds is wrong in a way that
    is hard to notice; one that claims only to rank alternatives is wrong in a way that shows
    up the first time the autotuner disagrees with the measurement, which is where the
    disagreement belongs.
    """

    flops_per_element: float = 1.0
    transcendental: bool = False

    def __post_init__(self) -> None:
        if self.flops_per_element < 0:
            raise ConfigError("an operation cannot cost negative work")


COSTS: dict[str, OpCost] = {
    "add": OpCost(flops_per_element=1.0),
    "sub": OpCost(flops_per_element=1.0),
    "mul": OpCost(flops_per_element=1.0),
    "div": OpCost(flops_per_element=4.0),
    "neg": OpCost(flops_per_element=1.0),
    "abs": OpCost(flops_per_element=1.0),
    "maximum": OpCost(flops_per_element=1.0),
    "minimum": OpCost(flops_per_element=1.0),
    "relu": OpCost(flops_per_element=1.0),
    "reciprocal": OpCost(flops_per_element=4.0),
    "exp": OpCost(flops_per_element=8.0, transcendental=True),
    "log": OpCost(flops_per_element=8.0, transcendental=True),
    "sqrt": OpCost(flops_per_element=6.0, transcendental=True),
    "tanh": OpCost(flops_per_element=12.0, transcendental=True),
    "sigmoid": OpCost(flops_per_element=10.0, transcendental=True),
    "cast": OpCost(flops_per_element=0.5),
    "step": OpCost(flops_per_element=1.0),
    "sum": OpCost(flops_per_element=1.0),
    "mean": OpCost(flops_per_element=1.0),
    "max": OpCost(flops_per_element=1.0),
    "matmul": OpCost(flops_per_element=2.0),
    "concat": OpCost(flops_per_element=0.0),
    "slice": OpCost(flops_per_element=0.0),
}


def cost_of(op: Op) -> OpCost:
    """The per element cost of an operation."""
    return COSTS.get(op.name, OpCost(flops_per_element=1.0))


@dataclass
class OpStats:
    """A tally of what a graph is made of."""

    counts: dict[str, int] = field(default_factory=dict)

    def add(self, op: Op) -> None:
        """Record one occurrence."""
        self.counts[op.name] = self.counts.get(op.name, 0) + 1

    @property
    def total(self) -> int:
        """Operations counted."""
        return sum(self.counts.values())

    def count_in_category(self, category: str) -> int:
        """How many of a given category appear."""
        if category not in CATEGORIES:
            raise ConfigError(f"unknown category {category!r}")
        return sum(
            count for name, count in self.counts.items() if BY_NAME[name].category == category
        )

    def as_dict(self) -> dict[str, int]:
        """Flat mapping for logging."""
        return dict(sorted(self.counts.items()))
