from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tgc.errors import ConfigError

# Starting the next load before the current compute has finished, and what that is worth.
#
# A loop over tiles does two things per iteration: it fetches the tile and it computes on it.
# Done in order, the total is the number of tiles times the sum of the two. Done overlapped, one
# iteration's fetch happens during the previous iteration's compute, and the total is one fetch
# plus the number of tiles times whichever of the two is slower. The difference is the whole of
# software pipelining.
#
# Three numbers come out of writing it down that are not obvious from the description.
#
# The speedup is at most two, and it reaches two only when the fetch and the compute take
# exactly the same time. Away from that point it falls off quickly in both directions: a loop
# that is nine parts compute to one part fetch gains eleven percent, and so does a loop that is
# nine parts fetch to one part compute. Prefetching is a fix for a balanced loop and most loops
# are not balanced.
#
# Depth beyond one buys nothing at all for two stages. The steady state is already limited by
# the slower stage, and running three fetches ahead does not make the slower stage faster. It
# does cost a buffer per level of depth, so the usual instinct to prefetch further is paying
# memory for nothing. Depth pays when there are more than two stages or when the fetch time
# varies, and neither of those is what people usually mean when they reach for it.
#
# And the prologue matters at small tile counts. The first fetch has nothing to overlap with, so
# a loop of four tiles reaches four fifths of the asymptotic gain and needs nine to reach nine
# tenths. That is a number worth having before deciding a tile size, because a tile large enough
# to leave a handful of tiles has given away the overlap it was chosen for.
#
# The prologue also breaks the symmetry above over a finite loop. The stage that runs alone at
# the start is the fetch, so a fetch bound loop pays more for its prologue than a compute bound
# one and the two sides of the curve differ by a thousandth at a thousand tiles.


@dataclass
class Pipeline:
    """A loop over tiles with a fetch and a compute per tile."""

    tiles: int
    fetch: float
    compute: float
    depth: int = 1

    def __post_init__(self) -> None:
        if self.tiles < 1:
            raise ConfigError(f"a loop needs at least one tile, got {self.tiles}")
        if self.fetch < 0 or self.compute < 0:
            raise ConfigError("neither stage can take negative time")
        if self.depth < 0:
            raise ConfigError(f"the depth cannot be {self.depth}")

    @property
    def serial_time(self) -> float:
        """How long the loop takes with no overlap at all."""
        return self.tiles * (self.fetch + self.compute)

    @property
    def overlapped_time(self) -> float:
        """How long it takes with the fetches running ahead.

        One fetch that overlaps with nothing, then one slower stage per tile. Deeper prefetching
        does not appear in this because it cannot: the steady state is set by the slower stage
        and no amount of running ahead changes which stage that is.
        """
        if self.depth == 0:
            return self.serial_time
        return self.fetch + self.tiles * max(self.fetch, self.compute)

    @property
    def speedup(self) -> float:
        """How much shorter the overlapped loop is."""
        if self.overlapped_time <= 0:
            return 1.0
        return self.serial_time / self.overlapped_time

    @property
    def buffers(self) -> int:
        """How many tile buffers the depth needs."""
        return self.depth + 1

    @property
    def bound_by(self) -> str:
        """Which stage sets the steady state."""
        if self.fetch > self.compute:
            return "fetch"
        if self.compute > self.fetch:
            return "compute"
        return "balanced"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "tiles": self.tiles,
            "fetch": self.fetch,
            "compute": self.compute,
            "depth": self.depth,
            "serial": round(self.serial_time, 4),
            "overlapped": round(self.overlapped_time, 4),
            "speedup": round(self.speedup, 4),
            "buffers": self.buffers,
            "bound_by": self.bound_by,
        }


def asymptotic_speedup(fetch: float, compute: float) -> float:
    """The speedup a loop of infinitely many tiles would get.

    The prologue disappears and the ratio becomes the sum over the larger, which is between one
    and two and equals two exactly when the two stages are equal.
    """
    if fetch < 0 or compute < 0:
        raise ConfigError("neither stage can take negative time")
    larger = max(fetch, compute)
    if larger <= 0:
        return 1.0
    return (fetch + compute) / larger


def balance_sweep(
    ratios: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0), tiles: int = 1000
) -> list[dict]:
    """Speedup against how lopsided the two stages are.

    Symmetric about a ratio of one, which is the shape of the formula and worth seeing: a loop
    that is ten parts compute to one part fetch and a loop that is ten parts fetch to one part
    compute both gain about a tenth. Overlapping does not care which stage is the slow one.
    """
    if not ratios:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for ratio in ratios:
        pipeline = Pipeline(tiles=tiles, fetch=ratio, compute=1.0)
        rows.append(
            {
                "ratio": ratio,
                "speedup": round(pipeline.speedup, 4),
                "bound_by": pipeline.bound_by,
            }
        )
    return rows


def the_gain_peaks_when_the_stages_match(tiles: int = 1000) -> dict:
    """Where in that sweep the gain is largest, and how large."""
    rows = balance_sweep(tiles=tiles)
    best = max(rows, key=lambda row: row["speedup"])
    return {
        "best_ratio": best["ratio"],
        "best_speedup": best["speedup"],
        "worst_speedup": min(row["speedup"] for row in rows),
    }


def the_curve_is_symmetric(tolerance: float = 1e-9) -> dict:
    """Whether a fetch bound loop and a compute bound loop gain the same.

    In the limit they do, exactly, and it follows from the formula rather than from the numbers.
    Over a finite loop they do not, and the reason is the prologue: the one fetch that overlaps
    with nothing is a fetch, so a loop whose fetches are the expensive stage pays more for its
    prologue than one whose computes are. The gap is a thousandth at a thousand tiles and it is
    real, and a test asserting exact symmetry on a finite loop would have been wrong.
    """
    finite = []
    limits = []
    for ratio in (0.1, 0.25, 0.5):
        fast = Pipeline(tiles=1000, fetch=ratio, compute=1.0).speedup
        slow = Pipeline(tiles=1000, fetch=1.0, compute=ratio).speedup
        finite.append(abs(fast - slow))
        limits.append(abs(asymptotic_speedup(ratio, 1.0) - asymptotic_speedup(1.0, ratio)))
    return {
        "pairs": len(finite),
        "largest_finite_gap": max(finite),
        "largest_limit_gap": max(limits),
        "symmetric_in_the_limit": max(limits) <= tolerance,
        "symmetric_over_a_finite_loop": max(finite) <= tolerance,
    }


def never_more_than_double(samples: int = 200) -> dict:
    """Whether any pair of stage times beats a factor of two.

    None does. Two stages can hide at most one of themselves behind the other, so the bound is
    exact rather than a rule of thumb, and a measurement claiming more than it is measuring
    something else.
    """
    if samples < 1:
        raise ConfigError(f"the sample count must be positive, got {samples}")
    best = 0.0
    for index in range(1, samples + 1):
        ratio = index / samples * 4
        best = max(best, asymptotic_speedup(ratio, 1.0))
    return {"samples": samples, "largest_speedup": round(best, 6), "under_two": best <= 2.0}


def depth_sweep(
    depths: Sequence[int] = (0, 1, 2, 3, 4, 8), tiles: int = 100, ratio: float = 1.0
) -> list[dict]:
    """Speedup and buffers against how far ahead the fetches run.

    One step from nothing to something and then flat. Depth zero is the serial loop; depth one
    overlaps the stages; depth two and beyond change nothing and cost a buffer each. For two
    stages that is the whole story, and the instinct to prefetch further is paying memory for a
    number that does not move.
    """
    if not depths:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for depth in depths:
        pipeline = Pipeline(tiles=tiles, fetch=ratio, compute=1.0, depth=depth)
        rows.append(
            {
                "depth": depth,
                "speedup": round(pipeline.speedup, 4),
                "buffers": pipeline.buffers,
            }
        )
    return rows


def depth_beyond_one_buys_nothing() -> dict:
    """The flat part of that sweep, stated as a comparison."""
    rows = {row["depth"]: row for row in depth_sweep()}
    return {
        "at_depth_one": rows[1]["speedup"],
        "at_depth_eight": rows[8]["speedup"],
        "identical": rows[1]["speedup"] == rows[8]["speedup"],
        "extra_buffers": rows[8]["buffers"] - rows[1]["buffers"],
    }


def variable_fetch_time(
    fetches: Sequence[float], compute: float = 1.0, depth: int = 1
) -> float:
    """How long a loop takes when the fetches are not all the same length.

    The case where depth does pay, which is why it is written out separately rather than folded
    into the model above. A queue of depth d absorbs a fetch that runs long as long as the ones
    around it were short enough to build up slack, and the deeper the queue the more slack it
    can hold.
    """
    if not fetches:
        raise ConfigError("there is nothing to run")
    if compute < 0:
        raise ConfigError("a stage cannot take negative time")
    if depth < 0:
        raise ConfigError(f"the depth cannot be {depth}")

    slack = 0.0
    total = 0.0
    for index, fetch in enumerate(fetches):
        if index < depth:
            total += fetch
            continue
        ahead = min(depth, len(fetches) - index)
        available = compute * ahead + slack
        if fetch <= available:
            slack = min(available - fetch, compute * depth)
            total += compute
        else:
            total += fetch - available + compute
            slack = 0.0
    return total


def depth_pays_when_the_fetches_vary(
    depths: Sequence[int] = (1, 2, 4, 8), spike: float = 6.0, tiles: int = 40
) -> list[dict]:
    """A loop where most fetches are quick and a few are slow.

    The measurement that keeps the flat result above from being read as an argument against
    prefetching. Under a fixed fetch time, depth is worthless. Under a fetch time that spikes
    every so often, depth is exactly the buffer that absorbs the spike, and the time falls with
    it until the queue is deep enough to hold the whole excess.
    """
    if not depths:
        raise ConfigError("there is nothing to sweep")
    if tiles < 1:
        raise ConfigError(f"a loop needs at least one tile, got {tiles}")
    fetches = [spike if index % 8 == 0 else 0.5 for index in range(tiles)]
    return [
        {
            "depth": depth,
            "time": round(variable_fetch_time(fetches, compute=1.0, depth=depth), 3),
            "buffers": depth + 1,
        }
        for depth in depths
    ]


def a_deeper_queue_absorbs_a_spike() -> dict:
    """Whether that time really falls with the depth."""
    rows = {row["depth"]: row for row in depth_pays_when_the_fetches_vary()}
    return {
        "at_depth_one": rows[1]["time"],
        "at_depth_eight": rows[8]["time"],
        "improved": rows[8]["time"] < rows[1]["time"],
        "saving": round(rows[1]["time"] - rows[8]["time"], 3),
    }


def tile_count_sweep(
    counts: Sequence[int] = (1, 2, 4, 8, 16, 64, 256, 1024), ratio: float = 1.0
) -> list[dict]:
    """How close a loop gets to its asymptotic gain, by how many tiles it runs.

    The first fetch overlaps with nothing, so it is a fixed cost spread over the loop. At one
    tile there is no gain at all, at four tiles the loop has four fifths of the limit, and it
    takes nine to reach nine tenths and a thousand to reach the last percent.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    limit = asymptotic_speedup(ratio, 1.0)
    rows = []
    for count in counts:
        speedup = Pipeline(tiles=count, fetch=ratio, compute=1.0).speedup
        rows.append(
            {
                "tiles": count,
                "speedup": round(speedup, 4),
                "share_of_the_limit": round(speedup / limit, 4) if limit else 0.0,
            }
        )
    return rows


def tiles_needed_for(share: float = 0.9, ratio: float = 1.0, limit: int = 4096) -> int:
    """The fewest tiles that reach a given share of the asymptotic gain.

    Searched rather than solved, because the answer is wanted as an integer and the closed form
    would have to be rounded anyway. It is the number a tiling pass needs: a tile size that
    leaves fewer tiles than this gives up part of the overlap it was chosen for.
    """
    if not 0 < share <= 1:
        raise ConfigError(f"the share has to be in (0, 1], got {share}")
    target = asymptotic_speedup(ratio, 1.0) * share
    for count in range(1, limit + 1):
        if Pipeline(tiles=count, fetch=ratio, compute=1.0).speedup >= target:
            return count
    return limit


def a_single_tile_gains_nothing() -> dict:
    """The degenerate case, which is the reason tile size and prefetching interact.

    One tile means one fetch and one compute with nothing to overlap them with, so the speedup
    is one exactly. A tiling pass that picks a tile large enough to leave a handful of tiles has
    made the prefetch it was going to rely on worthless.
    """
    return {
        "one_tile": Pipeline(tiles=1, fetch=1.0, compute=1.0).speedup,
        "two_tiles": Pipeline(tiles=2, fetch=1.0, compute=1.0).speedup,
        "many_tiles": Pipeline(tiles=1024, fetch=1.0, compute=1.0).speedup,
    }


def buffer_cost(tile_bytes: int, depth: int) -> int:
    """The memory a prefetch depth costs, in bytes."""
    if tile_bytes < 0:
        raise ConfigError(f"a tile cannot be {tile_bytes} bytes")
    if depth < 0:
        raise ConfigError(f"the depth cannot be {depth}")
    return tile_bytes * (depth + 1)


def memory_against_gain(
    tile_bytes: int = 65536, depths: Sequence[int] = (0, 1, 2, 4, 8)
) -> list[dict]:
    """What each level of depth costs and what it returns.

    The table that makes the decision. Depth one costs one extra tile of memory and returns the
    whole of the available gain; every level after that costs another tile and returns nothing,
    on a loop whose fetch time does not vary.
    """
    if not depths:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for depth in depths:
        pipeline = Pipeline(tiles=100, fetch=1.0, compute=1.0, depth=depth)
        rows.append(
            {
                "depth": depth,
                "bytes": buffer_cost(tile_bytes, depth),
                "speedup": round(pipeline.speedup, 4),
            }
        )
    return rows


def best_depth(tile_bytes: int = 65536, budget: int = 262144) -> int:
    """The deepest prefetch that fits in a memory budget and is worth having.

    Which is one, on a loop with a fixed fetch time, whatever the budget allows. Written as a
    search over the table rather than as a constant so that changing the model changes the
    answer rather than leaving a stale number in the code.
    """
    if budget < tile_bytes:
        raise ConfigError(f"a budget of {budget} cannot hold one tile of {tile_bytes}")
    table = memory_against_gain(tile_bytes, (0, 1, 2, 4, 8))
    affordable = [row for row in table if row["bytes"] <= budget]
    best = max(row["speedup"] for row in affordable)
    return min(row["depth"] for row in affordable if row["speedup"] == best)
