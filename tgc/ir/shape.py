from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tgc.errors import ConfigError, TypeInferenceError

# Shapes, including the ones that are not known until the graph runs.
#
# A compiler that only handles static shapes is a much simpler compiler and is not useful for
# anything with a batch dimension. The middle ground taken here is a symbolic dimension: a
# named size that compares equal to itself and to nothing else, so broadcasting and matmul
# can still be checked, and only the questions that genuinely need a number are deferred.
#
# The distinction that matters downstream is between a dimension that is unknown and one that
# is known to be one. Broadcasting treats a one specially, so an unknown dimension cannot be
# broadcast against anything without either a guard or an assumption, and pretending
# otherwise is how a compiled graph produces the wrong shape for the second batch it sees.


@dataclass(frozen=True)
class Dim:
    """One dimension, either a number or a name."""

    value: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.value is None and not self.name:
            raise ConfigError("a dimension is either a size or a name")
        if self.value is not None and self.value < 0:
            raise ConfigError(f"a dimension cannot be negative, got {self.value}")
        if self.value is not None and self.name:
            raise ConfigError("a dimension cannot be both a size and a name")

    @property
    def is_static(self) -> bool:
        """Whether the size is known now."""
        return self.value is not None

    @property
    def is_one(self) -> bool:
        """Whether it is known to be the broadcastable size.

        Not the same as unknown. An unknown dimension might be one at runtime and might not,
        and a compiler that guesses gets the shape right on the first batch it sees.
        """
        return self.value == 1

    def __str__(self) -> str:
        return str(self.value) if self.is_static else self.name


def dim(size: int | str) -> Dim:
    """A dimension from a number or a name."""
    if isinstance(size, str):
        return Dim(name=size)
    return Dim(value=size)


def dims(sizes: Sequence[int | str]) -> tuple[Dim, ...]:
    """A tuple of dimensions from a mixed sequence."""
    return tuple(dim(size) for size in sizes)


@dataclass(frozen=True)
class Shape:
    """The dimensions of one tensor."""

    sizes: tuple[Dim, ...] = ()

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.sizes)

    @property
    def is_static(self) -> bool:
        """Whether every dimension is known."""
        return all(size.is_static for size in self.sizes)

    @property
    def is_scalar(self) -> bool:
        """Whether it holds exactly one element regardless of rank."""
        return all(size.is_static and size.value == 1 for size in self.sizes)

    @property
    def elements(self) -> int:
        """Elements the tensor holds, if that is knowable."""
        if not self.is_static:
            raise TypeInferenceError(f"the size of {self} is not known until it runs")
        total = 1
        for size in self.sizes:
            total *= size.value or 0
        return total

    def bytes_for(self, element_size: int) -> int:
        """Storage the tensor takes."""
        if element_size < 1:
            raise ConfigError("an element takes at least one byte")
        return self.elements * element_size

    def __str__(self) -> str:
        return "[" + ", ".join(str(size) for size in self.sizes) + "]"

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, index: int) -> Dim:
        return self.sizes[index]


def shape(*sizes: int | str) -> Shape:
    """A shape from a list of sizes or names."""
    return Shape(sizes=dims(sizes))


def dims_equal(left: Dim, right: Dim) -> bool:
    """Whether two dimensions are certainly the same size.

    Certainly, not probably. Two different symbolic names may hold the same number at runtime
    and the compiler has no way to know it, so treating them as equal turns a shape error
    into a wrong answer.
    """
    if left.is_static and right.is_static:
        return left.value == right.value
    if left.is_static or right.is_static:
        return False
    return left.name == right.name


def broadcast_dims(left: Dim, right: Dim) -> Dim:
    """The result of broadcasting one dimension against another."""
    if dims_equal(left, right):
        return left
    if left.is_one:
        return right
    if right.is_one:
        return left
    if not left.is_static or not right.is_static:
        raise TypeInferenceError(
            f"cannot broadcast {left} against {right} without knowing whether either is one"
        )
    raise TypeInferenceError(f"cannot broadcast a dimension of {left} against {right}")


def broadcast(left: Shape, right: Shape) -> Shape:
    """The shape an elementwise operation between two others produces.

    Aligned from the right, which is the convention every array library uses and the one
    reason a rank three tensor combines with a rank one tensor at all.
    """
    rank = max(left.rank, right.rank)
    result = []
    for offset in range(1, rank + 1):
        first = left.sizes[-offset] if offset <= left.rank else Dim(value=1)
        second = right.sizes[-offset] if offset <= right.rank else Dim(value=1)
        result.append(broadcast_dims(first, second))
    return Shape(sizes=tuple(reversed(result)))


def broadcast_all(shapes: Sequence[Shape]) -> Shape:
    """Fold broadcasting across several shapes."""
    if not shapes:
        raise TypeInferenceError("there is nothing to broadcast")
    result = shapes[0]
    for other in shapes[1:]:
        result = broadcast(result, other)
    return result


def is_broadcastable(left: Shape, right: Shape) -> bool:
    """Whether two shapes combine without raising."""
    try:
        broadcast(left, right)
    except TypeInferenceError:
        return False
    return True


def matmul_shape(left: Shape, right: Shape) -> Shape:
    """The shape of a matrix product, with leading dimensions broadcast."""
    if left.rank < 2 or right.rank < 2:
        raise TypeInferenceError(f"a matrix product needs two matrices, got {left} and {right}")
    if not dims_equal(left.sizes[-1], right.sizes[-2]):
        raise TypeInferenceError(
            f"{left} and {right} do not meet: {left.sizes[-1]} against {right.sizes[-2]}"
        )
    batch = broadcast(Shape(sizes=left.sizes[:-2]), Shape(sizes=right.sizes[:-2]))
    return Shape(sizes=(*batch.sizes, left.sizes[-2], right.sizes[-1]))


def reduce_shape(source: Shape, axes: Sequence[int], *, keepdims: bool = False) -> Shape:
    """The shape left after reducing over some axes."""
    normalised = normalise_axes(axes, source.rank)
    kept = []
    for index, size in enumerate(source.sizes):
        if index in normalised:
            if keepdims:
                kept.append(Dim(value=1))
            continue
        kept.append(size)
    return Shape(sizes=tuple(kept))


def normalise_axes(axes: Sequence[int], rank: int) -> tuple[int, ...]:
    """Turn possibly negative axis indices into a sorted set of positive ones."""
    if rank < 0:
        raise ConfigError("a rank cannot be negative")
    seen = set()
    for axis in axes:
        resolved = axis + rank if axis < 0 else axis
        if not 0 <= resolved < rank:
            raise ConfigError(f"axis {axis} is outside a tensor of rank {rank}")
        if resolved in seen:
            raise ConfigError(f"axis {axis} is given twice")
        seen.add(resolved)
    return tuple(sorted(seen))


def transpose_shape(source: Shape, permutation: Sequence[int]) -> Shape:
    """The shape after permuting dimensions."""
    if sorted(permutation) != list(range(source.rank)):
        raise ConfigError(
            f"a permutation of rank {source.rank} must use each axis once, "
            f"got {list(permutation)}"
        )
    return Shape(sizes=tuple(source.sizes[axis] for axis in permutation))


def reshape_shape(source: Shape, sizes: Sequence[int]) -> Shape:
    """The shape after a reshape, resolving at most one inferred dimension."""
    inferred = [index for index, size in enumerate(sizes) if size == -1]
    if len(inferred) > 1:
        raise ConfigError("at most one dimension can be inferred")
    if any(size < -1 or size == 0 for size in sizes):
        raise ConfigError(f"a reshape target must be positive or -1, got {list(sizes)}")
    if not inferred:
        target = Shape(sizes=dims(sizes))
        if source.is_static and target.elements != source.elements:
            raise TypeInferenceError(f"cannot reshape {source.elements} elements into {target}")
        return target
    known = 1
    for size in sizes:
        if size != -1:
            known *= size
    if source.elements % known != 0:
        raise TypeInferenceError(f"cannot reshape {source} into {list(sizes)}")
    resolved = list(sizes)
    resolved[inferred[0]] = source.elements // known
    return Shape(sizes=dims(resolved))


def contiguous_strides(source: Shape) -> tuple[int, ...]:
    """Strides for a densely packed tensor, in elements.

    Written down because several passes need to reason about whether a view is contiguous,
    and comparing a computed stride against the packed one is the only honest way to answer
    that without carrying a layout flag around and hoping it stayed true.
    """
    if not source.is_static:
        raise TypeInferenceError(f"strides for {source} are not known until it runs")
    strides = [1] * source.rank
    for index in range(source.rank - 2, -1, -1):
        next_size = source.sizes[index + 1].value or 0
        strides[index] = strides[index + 1] * next_size
    return tuple(strides)
