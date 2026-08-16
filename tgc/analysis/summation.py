from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from tgc.errors import ConfigError

# What order a reduction is summed in, and why the compiler owns that question.
#
# passes/fusion.py and schedule/order.py are free to split a reduction across partitions and
# combine the partial sums, and nothing in the IR says they may not, because the IR says the
# node computes a sum. Floating point addition is not associative, so every one of those splits
# computes a different number. On sixty five thousand values with one large term the sequential
# order is wrong in the fifth significant figure and the pairwise tree is wrong in the ninth, a
# factor of nine thousand between two orders that do exactly the same number of additions.
#
# Four orders are measured against a correctly rounded reference: straight sequential, the
# pairwise tree a library uses, the compensated loop that carries the rounding error forward,
# and a sort first. The reference is math.fsum on the float64 values, which is exact, so the
# numbers below are errors rather than disagreements.
#
# Two results are worth carrying out of here. A split is an inexact rewrite in the same sense as
# the algebraic rules in passes/algebraic.py and belongs on the same side of that flag: it is
# usually an improvement, which is not the same as being safe, and a build that has to reproduce
# bit for bit has to pin the partition count rather than let the scheduler read it off the
# machine. And more partitions is not monotonically better. The error bottoms out at the square
# root of the length and then climbs straight back to where it started, because past that point
# the partial sums are too small to register against each other and the second stage has the
# same problem the first one had.

DEFAULT_BLOCK = 8


def _checked(values: np.ndarray) -> np.ndarray:
    """Reject anything that is not a one dimensional float32 array with values in it."""
    if values.ndim != 1:
        raise ConfigError(f"a reduction runs over a vector, got rank {values.ndim}")
    if values.size == 0:
        raise ConfigError("there is nothing to sum")
    if values.dtype != np.float32:
        raise ConfigError(f"these orders are about float32, got {values.dtype}")
    return values


def exact(values: np.ndarray) -> float:
    """The correctly rounded sum, in double precision.

    math.fsum keeps an exact running representation and rounds once at the end, so this is the
    answer every other order here is measured against rather than a fifth opinion.
    """
    return math.fsum(np.asarray(_checked(values), dtype=np.float64).tolist())


def sequential(values: np.ndarray) -> float:
    """One accumulator, left to right. What the reference interpreter does."""
    total = np.float32(0.0)
    for value in _checked(values):
        total = np.float32(total + value)
    return float(total)


def pairwise(values: np.ndarray, block: int = DEFAULT_BLOCK) -> float:
    """Halve the range until it fits a block, then sum the block sequentially.

    The order a library uses, and the order any parallel implementation falls into by accident.
    The accumulator only ever adds two values of comparable size, so the running total never
    grows far enough ahead of the next term to swallow it.
    """
    _checked(values)
    if block < 1:
        raise ConfigError(f"a block of {block} values is not a block")
    return float(_pairwise(values, block))


def _pairwise(values: np.ndarray, block: int) -> np.float32:
    """The recursion behind pairwise, kept separate so the checks run once."""
    if values.size <= block:
        total = np.float32(0.0)
        for value in values:
            total = np.float32(total + value)
        return total
    middle = values.size // 2
    return np.float32(_pairwise(values[:middle], block) + _pairwise(values[middle:], block))


def compensated(values: np.ndarray) -> float:
    """Carry the part of each addition that did not fit, and put it back next time.

    Four operations per element instead of one, and the error stops growing with the length
    altogether. Worth knowing before deciding that the pairwise tree is the end of the subject.
    """
    total = np.float32(0.0)
    carry = np.float32(0.0)
    for value in _checked(values):
        adjusted = np.float32(value - carry)
        raised = np.float32(total + adjusted)
        carry = np.float32(np.float32(raised - total) - adjusted)
        total = raised
    return float(total)


def sorted_ascending(values: np.ndarray) -> float:
    """Sort ascending, then sum. Small values go in first and the accumulator grows last.

    The fix people reach for. It sorts by value and not by magnitude, which is the same thing
    on positive data and the worst available order on signed data, because it puts every
    negative term first and then cancels the lot. It also costs a sort, which is more expensive
    than the reduction it protects and cannot be fused into anything.
    """
    return sequential(np.sort(_checked(values)))


def partitioned(values: np.ndarray, parts: int) -> float:
    """Split into equal partitions, sum each sequentially, then sum the partials.

    What a parallel reduction computes. One partition is the sequential order and one partition
    per element is a flat tree, and everything between them is a different number.
    """
    _checked(values)
    if parts < 1:
        raise ConfigError(f"{parts} partitions is not a split")
    if parts > values.size:
        raise ConfigError(f"{parts} partitions over {values.size} values leaves some empty")
    edges = [round(index * values.size / parts) for index in range(parts + 1)]
    partials = np.asarray(
        [sequential(values[edges[index] : edges[index + 1]]) for index in range(parts)],
        dtype=np.float32,
    )
    return sequential(partials)


def relative_error(approximate: float, reference: float) -> float:
    """How far one order landed from the correctly rounded answer."""
    if reference == 0.0:
        raise ConfigError("a relative error against zero is not a number")
    return abs(approximate - reference) / abs(reference)


def mixed_magnitudes(count: int = 4096, leader: float = 1e9, seed: int = 0) -> np.ndarray:
    """One large value followed by many small ones, all positive.

    The shape that breaks the sequential order. Once the accumulator reaches the leader, a term
    smaller than its rounding unit adds nothing at all, and the loop spends the rest of its time
    reading values and discarding them.
    """
    if count < 2:
        raise ConfigError(f"{count} values is not a reduction")
    if leader <= 0:
        raise ConfigError(f"a leader of {leader} is not a leader")
    generator = np.random.default_rng(seed)
    tail = generator.uniform(0.5, 1.5, size=count - 1)
    return np.asarray([leader, *tail], dtype=np.float32)


def uniform(count: int = 4096, seed: int = 0) -> np.ndarray:
    """Ordinary positive values of one magnitude. The case that looks like it should behave."""
    if count < 1:
        raise ConfigError(f"{count} values is not a reduction")
    generator = np.random.default_rng(seed)
    return np.asarray(generator.uniform(0.5, 1.5, size=count), dtype=np.float32)


def alternating(count: int = 4096, seed: int = 0) -> np.ndarray:
    """Values of both signs around zero, so the total is small and the terms are not.

    Cancellation rather than absorption, and the one input where the ranking changes. The
    relative error is measured against a total that nearly vanished, which magnifies everything,
    and sorting ascending is actively harmful here rather than merely expensive.
    """
    if count < 2:
        raise ConfigError(f"{count} values is not a reduction")
    generator = np.random.default_rng(seed)
    return np.asarray(generator.normal(0.0, 1.0, size=count), dtype=np.float32)


ORDERS = {
    "sequential": sequential,
    "pairwise": pairwise,
    "compensated": compensated,
    "sorted": sorted_ascending,
}


def error_of(order: str, values: np.ndarray) -> float:
    """One order's relative error on one input."""
    if order not in ORDERS:
        raise ConfigError(f"unknown order {order!r}, expected one of {sorted(ORDERS)}")
    return relative_error(ORDERS[order](values), exact(values))


def compare_orders(values: np.ndarray | None = None) -> list[dict]:
    """Every order on one input."""
    data = values if values is not None else mixed_magnitudes()
    return [
        {
            "order": name,
            "total": kernel(data),
            "error": error_of(name, data),
            "flops_per_element": flops_per_element(name),
        }
        for name, kernel in ORDERS.items()
    ]


def flops_per_element(order: str) -> int:
    """What each order costs per value, ignoring the sort.

    One for the straight loop and one for the tree, because the tree does the same number of
    additions in a different arrangement. Four for the compensated loop, which is the whole
    argument against it and the reason nothing enables it by default.
    """
    costs = {"sequential": 1, "pairwise": 1, "compensated": 4, "sorted": 1}
    if order not in costs:
        raise ConfigError(f"unknown order {order!r}, expected one of {sorted(costs)}")
    return costs[order]


def the_sequential_order_stops_adding(count: int = 4096) -> dict:
    """How much of the input the straight loop throws away.

    All of it but the first term. A float32 near a billion is spaced sixty four apart, so a term
    near one cannot move it, and every addition after the leader returns the accumulator
    unchanged. The loop reads four thousand values, does four thousand additions, and produces
    the number it had after the first one.
    """
    values = mixed_magnitudes(count=count)
    total = np.float32(0.0)
    absorbed = 0
    for value in values:
        raised = np.float32(total + value)
        if raised == total:
            absorbed += 1
        total = raised
    return {
        "values": count,
        "absorbed": absorbed,
        "share_absorbed": round(absorbed / count, 3),
        "error": relative_error(float(total), exact(values)),
    }


def error_grows_with_length(
    counts: Sequence[int] = (1024, 4096, 16384, 65536),
) -> list[dict]:
    """How each order degrades as the reduction gets longer.

    The straight loop is the only one that moves, and it moves in proportion: four times the
    length is four times the error, because every term after the leader is lost and the number
    lost is the count. The other three sit between one and eight parts in a hundred million at
    every length in the sweep.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in counts:
        values = mixed_magnitudes(count=count)
        rows.append(
            {
                "count": count,
                **{name: error_of(name, values) for name in ORDERS},
            }
        )
    return rows


def the_tree_beats_the_loop() -> dict:
    """By how much, on the length the sweep ends at."""
    values = mixed_magnitudes(count=65536)
    loop = error_of("sequential", values)
    tree = error_of("pairwise", values)
    return {
        "sequential": loop,
        "pairwise": tree,
        "ratio": round(loop / tree, 1) if tree else 0.0,
        "same_flops": flops_per_element("sequential") == flops_per_element("pairwise"),
    }


def compensation_is_flat(
    counts: Sequence[int] = (1024, 4096, 16384, 65536),
) -> dict:
    """Whether the compensated loop's error depends on the length at all.

    It does not. The carry puts back what the last addition dropped, so the error stays at the
    rounding unit of the final result no matter how many terms went into it, and the longest
    reduction in the sweep is the most accurate rather than the least.
    """
    rows = {row["count"]: row for row in error_grows_with_length(counts)}
    errors = [row["compensated"] for row in rows.values()]
    return {
        "smallest": min(errors),
        "largest": max(errors),
        "spread": max(errors) - min(errors),
        "grew": errors[-1] > errors[0] * 2,
    }


def sorting_helps_and_costs() -> dict:
    """What the sort buys, and what it does not.

    It lands on exactly the number the tree lands on, three orders below the straight loop, and
    it pays for a sort of the whole input to get there. The sort is more expensive than the
    addition it protects and cannot be fused into the producer, so it buys nothing the tree did
    not already give away for free.
    """
    values = mixed_magnitudes(count=16384)
    return {
        "sequential": error_of("sequential", values),
        "sorted": error_of("sorted", values),
        "pairwise": error_of("pairwise", values),
        "beats_the_loop": error_of("sorted", values) < error_of("sequential", values),
        "matches_the_tree": error_of("sorted", values) == error_of("pairwise", values),
    }


def a_split_changes_the_answer(counts: Sequence[int] = (1, 2, 8, 64)) -> dict:
    """Whether partitioning a reduction changes the number it produces.

    It does, at every partition count, and no two of them agree. All four are legal lowerings of
    the same node and they spread four thousand apart on a total of a billion, which is the
    entire problem with letting the schedule decide how many cores to use.
    """
    if len(counts) < 2:
        raise ConfigError("a comparison needs at least two partition counts")
    values = mixed_magnitudes(count=4096)
    totals = {count: partitioned(values, count) for count in counts}
    return {
        "totals": {str(count): total for count, total in totals.items()},
        "all_different": len(set(totals.values())) == len(totals),
        "spread": max(totals.values()) - min(totals.values()),
        "reference": exact(values),
    }


def partition_sweep(
    counts: Sequence[int] = (1, 2, 4, 16, 64, 256, 1024, 4096),
) -> list[dict]:
    """How the error moves as a reduction is cut into more pieces.

    Down by two orders and then all the way back up. Splitting shortens each accumulator, which
    is the improvement, but it also lengthens the sequential pass over the partials, which is
    the same problem again. The two cross at the square root of the length, and past there the
    partials are small enough that the leader absorbs them exactly as it absorbed the terms.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    values = mixed_magnitudes(count=4096)
    reference = exact(values)
    rows = []
    for count in counts:
        rows.append(
            {
                "partitions": count,
                "length": 4096 // count,
                "error": relative_error(partitioned(values, count), reference),
            }
        )
    return rows


def the_best_split_is_the_square_root() -> dict:
    """Where the bottom of that sweep is.

    At sixty four partitions of sixty four, on four thousand values. One partition is the
    sequential order and one value per partition is the sequential order again in the second
    stage, and the two ends of the sweep produce the same error to the last digit. Eighty times
    better sits exactly between them, which is a bound on how much a scheduler can win here by
    accident and how much it can throw away by going further.
    """
    rows = {row["partitions"]: row for row in partition_sweep()}
    return {
        "at_one": rows[1]["error"],
        "at_sixty_four": rows[64]["error"],
        "at_four_thousand": rows[4096]["error"],
        "the_ends_agree": rows[1]["error"] == rows[4096]["error"],
        "ratio": round(rows[1]["error"] / rows[64]["error"], 1) if rows[64]["error"] else 0.0,
    }


def reversing_the_input_changes_the_total() -> dict:
    """The shortest demonstration that the rewrite is not exact.

    Sum the same values in the other direction and the answer differs. A pass that reorders a
    reduction is doing this, and no amount of care about the rest of the graph makes it an
    identity.
    """
    values = mixed_magnitudes(count=4096)
    forward = sequential(values)
    backward = sequential(np.ascontiguousarray(values[::-1]))
    return {
        "forward": forward,
        "backward": backward,
        "identical": forward == backward,
        "gap": abs(forward - backward),
        "reference": exact(values),
    }


def reordering_usually_improves_it() -> dict:
    """And that the inexact rewrite is nearly always the better one.

    Backwards is closer to the truth here, because the large value arrives last and the small
    terms have already been accumulated together. That is a good reason to enable the rewrite
    and not a reason to call it exact.
    """
    values = mixed_magnitudes(count=4096)
    reference = exact(values)
    forward = relative_error(sequential(values), reference)
    backward = relative_error(sequential(np.ascontiguousarray(values[::-1])), reference)
    return {
        "forward": forward,
        "backward": backward,
        "backward_is_better": backward < forward,
        "ratio": round(forward / backward, 1) if backward else 0.0,
    }


def double_precision_removes_the_problem() -> dict:
    """What the whole subject costs to avoid.

    Twice the traffic and, on most accelerators, a good deal more than twice the time. The
    error drops to the point where the comparison stops being meaningful, which is the trade
    somebody makes when they widen an accumulator rather than reorder it.
    """
    values = mixed_magnitudes(count=16384)
    reference = exact(values)
    narrow = relative_error(sequential(values), reference)
    total = np.float64(0.0)
    for value in np.asarray(values, dtype=np.float64):
        total = np.float64(total + value)
    wide = relative_error(float(total), reference)
    return {
        "float32": narrow,
        "float64": wide,
        "ratio": round(narrow / wide, 1) if wide else 0.0,
        "bytes_per_element": 8,
        "beats_every_narrow_order": all(wide < error_of(name, values) for name in ORDERS),
    }


def compare_inputs() -> list[dict]:
    """Every order on all three input shapes.

    The ranking does not hold. Sorting is the second best order on positive data and the worst
    of the four by three orders of magnitude on data with signs in it, because ascending order
    puts every negative term first and the accumulator swings out to a large magnitude before
    coming back. The tree and the compensated loop are the only two that are never bad.
    """
    rows = []
    for label, values in (
        ("mixed magnitudes", mixed_magnitudes(count=16384)),
        ("uniform", uniform(count=16384)),
        ("alternating", alternating(count=16384)),
    ):
        rows.append(
            {
                "input": label,
                **{name: error_of(name, values) for name in ORDERS},
            }
        )
    return rows


def uniform_data_cares_too() -> dict:
    """Whether any of this matters when every value is the same size.

    Yes, which is the part that is easy to miss. There is no leader to swallow anything and the
    straight loop is still eighty times worse than the tree, because after ten thousand terms
    the accumulator is ten thousand times the size of the next one and each addition drops a
    little. The dramatic input makes the effect visible. It does not cause it.
    """
    values = uniform(count=16384)
    return {
        "sequential": error_of("sequential", values),
        "pairwise": error_of("pairwise", values),
        "ratio": round(error_of("sequential", values) / error_of("pairwise", values), 2)
        if error_of("pairwise", values)
        else 0.0,
    }


def an_unknown_order_is_refused() -> bool:
    """Whether asking for an order that does not exist names the ones that do."""
    try:
        error_of("magic", uniform(count=16))
    except ConfigError:
        return True
    return False


def an_empty_reduction_is_refused() -> bool:
    """Whether summing nothing is refused rather than answering zero.

    Zero is the right answer arithmetically and the wrong answer here, because a reduction over
    an empty axis in a graph is almost always a shape that went wrong upstream.
    """
    try:
        sequential(np.asarray([], dtype=np.float32))
    except ConfigError:
        return True
    return False


def a_split_wider_than_the_input_is_refused() -> bool:
    """Whether asking for more partitions than values is caught."""
    try:
        partitioned(uniform(count=16), 32)
    except ConfigError:
        return True
    return False


def a_double_input_is_refused() -> bool:
    """Whether handing these float32 orders a float64 array is refused.

    Every kernel here rounds to float32 after each operation on purpose. Given doubles it would
    silently measure something else and report it as a float32 result.
    """
    try:
        sequential(np.asarray([1.0, 2.0], dtype=np.float64))
    except ConfigError:
        return True
    return False
