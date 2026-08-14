from __future__ import annotations

from dataclasses import dataclass

from tgc.errors import ConfigError, TypeInferenceError

# Element types, and the promotion rules between them.
#
# Promotion is written out rather than deferred to the framework because a compiler has to
# answer the question before it runs anything. Constant folding in particular has to know
# what type the folded value would have had at runtime; folding in the widest type available
# and narrowing afterwards gives a different answer from the graph it replaced, and the
# difference shows up as a model that was fine until it was compiled.


@dataclass(frozen=True)
class DType:
    """One element type."""

    name: str
    bits: int
    kind: str

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ConfigError(f"{self.name} has no width")
        if self.kind not in ("float", "int", "uint", "bool"):
            raise ConfigError(f"unknown kind {self.kind!r} for {self.name}")

    @property
    def bytes(self) -> int:
        """Storage one element takes."""
        return max(1, self.bits // 8)

    @property
    def is_float(self) -> bool:
        """Whether arithmetic on it rounds."""
        return self.kind == "float"

    @property
    def is_integral(self) -> bool:
        """Whether arithmetic on it is exact until it overflows."""
        return self.kind in ("int", "uint", "bool")

    def __str__(self) -> str:
        return self.name


BOOL = DType(name="bool", bits=8, kind="bool")
INT8 = DType(name="int8", bits=8, kind="int")
INT32 = DType(name="int32", bits=32, kind="int")
INT64 = DType(name="int64", bits=64, kind="int")
FLOAT16 = DType(name="float16", bits=16, kind="float")
BFLOAT16 = DType(name="bfloat16", bits=16, kind="float")
FLOAT32 = DType(name="float32", bits=32, kind="float")
FLOAT64 = DType(name="float64", bits=64, kind="float")

ALL_DTYPES = (BOOL, INT8, INT32, INT64, FLOAT16, BFLOAT16, FLOAT32, FLOAT64)

BY_NAME = {dtype.name: dtype for dtype in ALL_DTYPES}

# Rank inside a kind. Promotion moves up this ladder and never sideways, which is what keeps
# float16 and bfloat16 from being ordered against each other: they have the same width and
# neither contains the other, so a graph that mixes them has to say which it wants.
_FLOAT_ORDER = (FLOAT16, BFLOAT16, FLOAT32, FLOAT64)
_INT_ORDER = (BOOL, INT8, INT32, INT64)


def from_name(name: str) -> DType:
    """Look up a type by the name a frontend used."""
    if name not in BY_NAME:
        raise ConfigError(f"unknown dtype {name!r}, expected one of {sorted(BY_NAME)}")
    return BY_NAME[name]


def promote(left: DType, right: DType) -> DType:
    """The type an operation between two others produces.

    Float beats integer regardless of width, which is the rule everybody expects, and int64
    combined with float16 gives float16 rather than something wider, which is the rule
    nobody expects and is what every framework does. Writing it down here means constant
    folding and shape inference agree with each other by construction.
    """
    if left == right:
        return left
    if left.is_float and right.is_float:
        return _wider_float(left, right)
    if left.is_float:
        return left
    if right.is_float:
        return right
    return _wider_int(left, right)


def _wider_float(left: DType, right: DType) -> DType:
    """The wider of two float types, refusing the pair that has no answer."""
    if {left, right} == {FLOAT16, BFLOAT16}:
        raise TypeInferenceError(
            "float16 and bfloat16 have the same width and neither contains the other, "
            "so the graph has to say which it wants"
        )
    return max(left, right, key=_FLOAT_ORDER.index)


def _wider_int(left: DType, right: DType) -> DType:
    """The wider of two integral types."""
    for dtype in (left, right):
        if dtype not in _INT_ORDER:
            raise TypeInferenceError(f"no promotion rule for {dtype}")
    return max(left, right, key=_INT_ORDER.index)


def promote_all(dtypes: list[DType]) -> DType:
    """Fold promotion across several types."""
    if not dtypes:
        raise TypeInferenceError("there is nothing to promote")
    result = dtypes[0]
    for dtype in dtypes[1:]:
        result = promote(result, dtype)
    return result


def can_represent_exactly(dtype: DType, value: int) -> bool:
    """Whether an integer survives a round trip through a type.

    The check constant folding needs before it replaces an integer expression with a literal
    in a float type. float16 stops counting at 2048, so a folded index arithmetic chain that
    was correct as int32 is silently wrong once it lands in a half precision constant.
    """
    if dtype.is_integral:
        if dtype is BOOL:
            return value in (0, 1)
        width = dtype.bits - 1
        return -(2**width) <= value < 2**width
    mantissa = {FLOAT16: 11, BFLOAT16: 8, FLOAT32: 24, FLOAT64: 53}[dtype]
    return abs(value) <= 2**mantissa


def accumulator_for(dtype: DType) -> DType:
    """The type a reduction over this one should accumulate in.

    Always at least float32 for the narrow floats. Summing ten thousand float16 values in
    float16 loses most of the tail, and the loss is not a rounding detail: the running total
    grows until each new addend is below its last bit and the sum stops moving entirely.
    """
    if dtype in (FLOAT16, BFLOAT16):
        return FLOAT32
    if dtype in (BOOL, INT8):
        return INT32
    return dtype
