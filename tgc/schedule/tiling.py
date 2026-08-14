from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tgc.errors import ConfigError, ScheduleError

# Blocking a loop nest so the working set fits in a cache.
#
# A matrix product written as three nested loops touches the whole of one operand for every
# row of the other. Blocked into tiles it touches one tile of each, and if the tiles fit in
# cache the traffic to the level below falls by the tile size rather than by a constant.
#
# The arithmetic is worth stating precisely because the intuition is wrong in a specific way.
# Blocking does not reduce the number of loads issued; it reduces the number that miss. The
# quantity that changes is data brought in from the level below, and for a square tile of side
# t on a product of size n it falls from n cubed to n cubed over t. Doubling the tile halves
# the traffic, right up until the tile stops fitting, at which point the traffic goes back to
# where it started and the code is more complicated.


@dataclass
class Tile:
    """One blocking of a three dimensional loop nest."""

    rows: int
    columns: int
    depth: int

    def __post_init__(self) -> None:
        if min(self.rows, self.columns, self.depth) < 1:
            raise ConfigError("every tile dimension has to be positive")

    @property
    def working_set_elements(self) -> int:
        """Elements the three tiles hold at once.

        A tile of the left operand, a tile of the right, and the accumulator. Counting only
        two of them is the mistake that makes a tile look like it fits when it does not.
        """
        return self.rows * self.depth + self.depth * self.columns + self.rows * self.columns

    def working_set_bytes(self, element_bytes: int = 4) -> int:
        """Storage the working set occupies."""
        if element_bytes < 1:
            raise ConfigError("an element takes at least one byte")
        return self.working_set_elements * element_bytes

    def fits_in(self, cache_bytes: int, element_bytes: int = 4) -> bool:
        """Whether the working set stays in a cache of a given size."""
        if cache_bytes < 0:
            raise ConfigError("a cache cannot be negative")
        return self.working_set_bytes(element_bytes) <= cache_bytes

    def as_dict(self) -> dict[str, int]:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "columns": self.columns,
            "depth": self.depth,
            "working_set": self.working_set_elements,
        }


def square_tile(side: int) -> Tile:
    """A tile that is the same size in every dimension."""
    return Tile(rows=side, columns=side, depth=side)


@dataclass
class MatmulShape:
    """The three sizes of a matrix product."""

    rows: int = 512
    columns: int = 512
    depth: int = 512

    def __post_init__(self) -> None:
        if min(self.rows, self.columns, self.depth) < 1:
            raise ConfigError("every matmul dimension has to be positive")

    @property
    def flops(self) -> int:
        """Arithmetic the product performs."""
        return 2 * self.rows * self.columns * self.depth

    @property
    def operand_elements(self) -> int:
        """Elements the three matrices hold."""
        return self.rows * self.depth + self.depth * self.columns + self.rows * self.columns

    def as_dict(self) -> dict[str, int]:
        """Flat mapping for logging."""
        return {"rows": self.rows, "columns": self.columns, "depth": self.depth}


def untiled_traffic(shape: MatmulShape, element_bytes: int = 4) -> int:
    """Data a naive triple loop brings in from below.

    The inner loop walks a whole column of the right operand for every element of the output,
    so the right operand is re read once per output row and nothing stays resident.
    """
    reads = shape.rows * shape.columns * shape.depth * 2
    writes = shape.rows * shape.columns
    return (reads + writes) * element_bytes


def tiled_traffic(shape: MatmulShape, tile: Tile, element_bytes: int = 4) -> int:
    """Data a blocked loop nest brings in from below.

    Each operand tile is loaded once per tile of the third dimension, which is what turns the
    n cubed term into n cubed over the tile side. The accumulator tile is written once.
    """
    row_tiles = math.ceil(shape.rows / tile.rows)
    column_tiles = math.ceil(shape.columns / tile.columns)
    depth_tiles = math.ceil(shape.depth / tile.depth)

    left = row_tiles * depth_tiles * column_tiles * tile.rows * tile.depth
    right = column_tiles * depth_tiles * row_tiles * tile.depth * tile.columns
    output = row_tiles * column_tiles * tile.rows * tile.columns
    return (left + right + output) * element_bytes


def traffic_reduction(shape: MatmulShape, tile: Tile, element_bytes: int = 4) -> float:
    """How much less data the blocked version brings in."""
    tiled = tiled_traffic(shape, tile, element_bytes)
    if tiled == 0:
        raise ScheduleError("a tiling that moves nothing cannot be compared")
    return untiled_traffic(shape, element_bytes) / tiled


def effective_traffic(
    shape: MatmulShape, tile: Tile, cache_bytes: int, element_bytes: int = 4
) -> int:
    """Traffic once the tile is checked against the cache it was meant to fit in.

    A tile that does not fit gets no reuse at all, so the answer falls back to the naive
    number. That discontinuity is the whole reason tile selection is a search rather than a
    formula: the model is smooth until it is not.
    """
    if tile.fits_in(cache_bytes, element_bytes):
        return tiled_traffic(shape, tile, element_bytes)
    return untiled_traffic(shape, element_bytes)


def largest_fitting_tile(cache_bytes: int, element_bytes: int = 4, limit: int = 512) -> int:
    """The biggest square tile whose working set stays in cache."""
    if cache_bytes < 1:
        raise ConfigError("a cache has to hold something")
    if limit < 1:
        raise ConfigError("the limit has to be positive")
    best = 1
    for side in range(1, limit + 1):
        if square_tile(side).fits_in(cache_bytes, element_bytes):
            best = side
        else:
            break
    return best


def sweep_tiles(
    shape: MatmulShape | None = None,
    cache_bytes: int = 256 * 1024,
    sides: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
) -> list[dict]:
    """Traffic across a range of square tile sizes, with the cache limit applied.

    Two regimes with a cliff between them. Below the limit the traffic halves every time the
    tile doubles; above it the reuse disappears and the traffic jumps back to the naive
    number in one step.
    """
    target = shape or MatmulShape()
    if not sides:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for side in sides:
        tile = square_tile(side)
        rows.append(
            {
                "side": side,
                "working_set_bytes": tile.working_set_bytes(),
                "fits": tile.fits_in(cache_bytes),
                "traffic": effective_traffic(target, tile, cache_bytes),
                "reduction": round(
                    untiled_traffic(target) / effective_traffic(target, tile, cache_bytes), 3
                ),
            }
        )
    return rows


def best_tile(
    shape: MatmulShape | None = None,
    cache_bytes: int = 256 * 1024,
    sides: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
) -> int:
    """The tile side with the least traffic, chosen by search rather than by formula."""
    rows = sweep_tiles(shape, cache_bytes, sides)
    return min(rows, key=lambda row: (row["traffic"], -row["side"]))["side"]


def doubling_halves_traffic(
    shape: MatmulShape | None = None, cache_bytes: int = 1 << 30
) -> list[dict]:
    """Traffic as the tile doubles, with the cache made large enough not to interfere.

    The claim on its own, isolated from the cliff. Each doubling of the tile side takes the
    traffic to roughly half, which is what the n cubed over t term says and is worth seeing
    rather than deriving.
    """
    target = shape or MatmulShape()
    rows = []
    previous = None
    for side in (1, 2, 4, 8, 16, 32, 64):
        traffic = effective_traffic(target, square_tile(side), cache_bytes)
        rows.append(
            {
                "side": side,
                "traffic": traffic,
                "ratio_to_previous": round(previous / traffic, 3) if previous else None,
            }
        )
        previous = traffic
    return rows


def arithmetic_per_byte(shape: MatmulShape, tile: Tile, element_bytes: int = 4) -> float:
    """Work done per byte brought in, under a given blocking.

    The number blocking exists to raise. A naive product does two operations per two elements
    read; a blocked one does the same arithmetic against a fraction of the traffic, which is
    what moves a kernel across the ridge point of the roofline.
    """
    traffic = tiled_traffic(shape, tile, element_bytes)
    if traffic == 0:
        raise ScheduleError("a tiling that moves nothing has no intensity")
    return shape.flops / traffic
