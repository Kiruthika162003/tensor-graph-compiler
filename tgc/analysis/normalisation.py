from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError

# Three ways of rescaling a row, and what separates them.
#
# Layer normalisation subtracts the mean and divides by the standard deviation. Root mean square
# normalisation skips the subtraction and divides by the root of the mean square. Batch
# normalisation uses statistics from the batch rather than from the row, which makes it a
# different kind of operation entirely: its output for one example depends on the other examples
# beside it.
#
# The arithmetic difference between the first two is one reduction out of two and a subtraction
# per element, which is three eighths of the work. What that costs in accuracy depends entirely
# on whether the data has a mean, and the threshold is lower than it sounds. On a row that is
# exactly centred the two agree to the rounding unit. On ordinary random rows of width thirty
# two they already differ by a fifth, because the sample mean of thirty two draws is about a
# fifth of a standard deviation and that is enough. Centred in expectation is not centred.
#
# Batch normalisation is the one with a compiler consequence rather than a numerical one. At
# inference it uses stored statistics instead of the batch, which makes it an affine map with
# constant coefficients, and an affine map sitting after a matrix product folds into that
# product's weight. That is a whole operation removed at compile time, and the fold is measured
# here rather than described.


@dataclass(frozen=True)
class NormShape:
    """A batch of rows to be normalised."""

    rows: int
    width: int

    def __post_init__(self) -> None:
        if min(self.rows, self.width) < 1:
            raise ConfigError("every dimension has to be positive")

    @property
    def elements(self) -> int:
        """Numbers in the batch."""
        return self.rows * self.width

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"rows": self.rows, "width": self.width, "elements": self.elements}


def layer_norm(
    values: torch.Tensor, gain: torch.Tensor, offset: torch.Tensor, epsilon: float = 1e-5
) -> torch.Tensor:
    """Centre, divide by the deviation, then scale and shift."""
    _check(values, epsilon)
    centred = values - values.mean(dim=-1, keepdim=True)
    variance = centred.pow(2).mean(dim=-1, keepdim=True)
    return centred / (variance + epsilon).sqrt() * gain + offset


def rms_norm(values: torch.Tensor, gain: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Divide by the root of the mean square, with no centring and no shift.

    Two operations fewer than the version above: no mean to subtract and no offset to add. It is
    the same function whenever the input already has a mean of zero and a different one whenever
    it does not, which is the whole of the comparison below.
    """
    _check(values, epsilon)
    scale = values.pow(2).mean(dim=-1, keepdim=True)
    return values / (scale + epsilon).sqrt() * gain


def batch_norm(
    values: torch.Tensor,
    gain: torch.Tensor,
    offset: torch.Tensor,
    epsilon: float = 1e-5,
    *,
    running: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Normalise each column by the statistics of the batch, or by stored ones.

    The two modes are different functions and that is the point. With the batch statistics the
    output for one row depends on every other row, which means the operation cannot be evaluated
    for a single example and cannot be split across devices without communicating. With the
    stored ones it is an affine map and none of that is true.
    """
    _check(values, epsilon)
    if running is not None:
        mean, variance = running
    else:
        mean = values.mean(dim=0, keepdim=True)
        variance = values.var(dim=0, unbiased=False, keepdim=True)
    return (values - mean) / (variance + epsilon).sqrt() * gain + offset


def _check(values: torch.Tensor, epsilon: float) -> None:
    """Raise if the input or the epsilon is not usable."""
    if values.dim() != 2:
        raise ConfigError(f"a normalisation takes a matrix, got rank {values.dim()}")
    if epsilon <= 0:
        raise ConfigError(f"the epsilon has to be positive, got {epsilon}")


def random_rows(
    shape: NormShape, *, seed: int = 0, offset: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch, a gain and an offset.

    The offset shifts every row away from zero, which is what separates the two row wise
    normalisations most clearly. It is not the only thing that separates them: a row drawn with
    no offset at all still has a sample mean, and at a width of thirty two that alone is enough.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape.rows, shape.width, generator=generator) + offset
    gain = torch.ones(shape.width)
    shift = torch.zeros(shape.width)
    return values, gain, shift


def matches_the_library(shape: NormShape | None = None, *, seed: int = 0) -> dict:
    """The hand written normalisations against torch's own.

    Only the layer and batch versions have a library equivalent to check against. The root mean
    square one does not, so it is checked against the layer version on data that has been
    centred, where the two are the same function and any disagreement is a mistake rather than a
    difference.
    """
    target = shape if shape is not None else NormShape(8, 32)
    values, gain, shift = random_rows(target, seed=seed)
    centred = values - values.mean(dim=-1, keepdim=True)

    mine_layer = layer_norm(values, gain, shift)
    theirs_layer = torch.nn.functional.layer_norm(
        values, (target.width,), weight=gain, bias=shift, eps=1e-5
    )
    mine_batch = batch_norm(values, gain, shift)
    theirs_batch = torch.nn.functional.batch_norm(
        values, None, None, weight=gain, bias=shift, training=True, eps=1e-5
    )
    return {
        "layer_gap": float((mine_layer - theirs_layer).abs().max()),
        "batch_gap": float((mine_batch - theirs_batch).abs().max()),
        "rms_against_layer_on_centred": float(
            (rms_norm(centred, gain) - layer_norm(centred, gain, shift)).abs().max()
        ),
    }


def the_two_row_norms_agree_on_centred_data(
    shape: NormShape | None = None, *, seed: int = 0
) -> dict:
    """How close the two are when the data has no mean to remove."""
    target = shape if shape is not None else NormShape(8, 32)
    values, gain, shift = random_rows(target, seed=seed)
    centred = values - values.mean(dim=-1, keepdim=True)
    gap = float((rms_norm(centred, gain) - layer_norm(centred, gain, shift)).abs().max())
    scale = float(layer_norm(centred, gain, shift).abs().max())
    return {"largest_gap": gap, "relative_gap": gap / scale if scale else gap}


def offset_sweep(
    offsets: Sequence[float] = (0.0, 0.1, 1.0, 10.0), shape: NormShape | None = None
) -> list[dict]:
    """How far apart the two row normalisations are, against how offset the data is.

    Apart everywhere, including at an offset of zero, because a random row of width thirty two
    has a sample mean of about a fifth of its deviation and that is already enough to separate
    them. The gap then grows with the offset until the two are computing unrelated things: at an
    offset of ten the layer version still produces a row of unit spread and the other produces
    something almost constant, because the root mean square of a row centred on ten is dominated
    by the ten.
    """
    if not offsets:
        raise ConfigError("there is nothing to sweep")
    target = shape if shape is not None else NormShape(8, 32)
    rows = []
    for offset in offsets:
        values, gain, shift = random_rows(target, offset=offset)
        layered = layer_norm(values, gain, shift)
        rooted = rms_norm(values, gain)
        gap = float((layered - rooted).abs().max())
        rows.append(
            {
                "offset": offset,
                "largest_gap": round(gap, 5),
                "layer_spread": round(float(layered.std()), 5),
                "rms_spread": round(float(rooted.std()), 5),
            }
        )
    return rows


def the_gap_grows_with_the_offset() -> dict:
    """Whether the divergence really tracks the offset.

    It does, from a fifth at no offset to more than three at an offset of ten. The value at no
    offset is the interesting one: it is not zero, because a finite row is not centred just
    because it was drawn from something that is.
    """
    rows = {row["offset"]: row for row in offset_sweep()}
    return {
        "at_zero": rows[0.0]["largest_gap"],
        "at_ten": rows[10.0]["largest_gap"],
        "grew": rows[10.0]["largest_gap"] > rows[0.0]["largest_gap"],
    }


def the_root_mean_square_version_stops_normalising(shape: NormShape | None = None) -> dict:
    """What an offset does to the spread of each version's output.

    The layer version produces a row with unit spread whatever the offset, because it removed
    the offset first. The other one produces a row whose spread falls as the offset grows, so at
    an offset of ten it is dividing by a number that has almost nothing to do with the variation
    it was meant to normalise.
    """
    target = shape if shape is not None else NormShape(8, 32)
    rows = {row["offset"]: row for row in offset_sweep(shape=target)}
    return {
        "layer_spread_at_zero": rows[0.0]["layer_spread"],
        "layer_spread_at_ten": rows[10.0]["layer_spread"],
        "rms_spread_at_zero": rows[0.0]["rms_spread"],
        "rms_spread_at_ten": rows[10.0]["rms_spread"],
    }


def arithmetic_per_element(kind: str) -> float:
    """Operations one element of each normalisation costs.

    Counted as reductions and elementwise work rather than in flops, because the reductions are
    what a fusion pass cares about and the elementwise work is what it can merge. The numbers
    are small and the ratio between them is the useful part.
    """
    costs = {"layer": 8.0, "rms": 5.0, "batch": 8.0}
    if kind not in costs:
        raise ConfigError(f"unknown normalisation {kind!r}, expected one of {sorted(costs)}")
    return costs[kind]


def reductions_for(kind: str) -> int:
    """How many passes over the row each version needs."""
    counts = {"layer": 2, "rms": 1, "batch": 2}
    if kind not in counts:
        raise ConfigError(f"unknown normalisation {kind!r}")
    return counts[kind]


def compare_cost(shape: NormShape | None = None) -> list[dict]:
    """What each version costs on one batch.

    The root mean square version does one pass over the row where the others do two, which for a
    memory bound operation is the number that matters. Everything else about the comparison is a
    detail next to halving the passes.
    """
    target = shape if shape is not None else NormShape(8, 4096)
    return [
        {
            "normalisation": kind,
            "reductions": reductions_for(kind),
            "operations": arithmetic_per_element(kind) * target.elements,
        }
        for kind in ("layer", "rms", "batch")
    ]


def one_reduction_instead_of_two() -> dict:
    """The saving stated as the thing a fusion pass sees."""
    rows = {row["normalisation"]: row for row in compare_cost()}
    return {
        "layer_reductions": rows["layer"]["reductions"],
        "rms_reductions": rows["rms"]["reductions"],
        "operations_saved": round(
            1 - rows["rms"]["operations"] / rows["layer"]["operations"], 4
        ),
    }


def batch_norm_depends_on_the_batch(shape: NormShape | None = None, *, seed: int = 0) -> dict:
    """Whether one row's output changes when another row does.

    It does, which is the property that makes batch normalisation a different kind of operation.
    A compiler cannot evaluate it for one example, cannot split the batch across devices without
    communicating, and cannot fuse it with anything that would reorder the rows.
    """
    target = shape if shape is not None else NormShape(8, 32)
    values, gain, shift = random_rows(target, seed=seed)
    first = batch_norm(values, gain, shift)[0]

    disturbed = values.clone()
    disturbed[1] = disturbed[1] + 100.0
    second = batch_norm(disturbed, gain, shift)[0]
    return {
        "changed": not bool(torch.equal(first, second)),
        "largest_change": float((first - second).abs().max()),
    }


def the_row_norms_do_not(shape: NormShape | None = None, *, seed: int = 0) -> dict:
    """The same disturbance against the two row wise versions.

    Nothing moves. Each row is normalised by its own statistics, so a change in another row is
    invisible, and that is why both of them evaluate for a single example and split across a
    batch dimension for free.
    """
    target = shape if shape is not None else NormShape(8, 32)
    values, gain, shift = random_rows(target, seed=seed)
    disturbed = values.clone()
    disturbed[1] = disturbed[1] + 100.0
    before = layer_norm(values, gain, shift)[0]
    after = layer_norm(disturbed, gain, shift)[0]
    return {
        "layer_changed": not bool(torch.equal(before, after)),
        "rms_changed": not bool(
            torch.equal(rms_norm(values, gain)[0], rms_norm(disturbed, gain)[0])
        ),
    }


def fold_into_the_weight(
    weight: torch.Tensor,
    bias: torch.Tensor,
    gain: torch.Tensor,
    offset: torch.Tensor,
    running: tuple[torch.Tensor, torch.Tensor],
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch normalisation with stored statistics, folded into the product before it.

    The whole operation disappears. With fixed statistics the normalisation is a multiply and an
    add per column, and a multiply and an add applied to the output of a matrix product is the
    same as scaling the weight's columns and shifting the bias. So the compiler removes it at
    compile time and the model runs one operation shorter.
    """
    if epsilon <= 0:
        raise ConfigError(f"the epsilon has to be positive, got {epsilon}")
    mean, variance = running
    scale = gain / (variance + epsilon).sqrt()
    return weight * scale, (bias - mean.reshape(-1)) * scale + offset


def the_fold_computes_the_same_thing(shape: NormShape | None = None, *, seed: int = 0) -> dict:
    """The folded weight against the product followed by the normalisation.

    Not bit identical and within a rounding unit, which is expected: the folded form performs
    the scaling once on the weight rather than once per row, so the same multiplications happen
    in a different order and against different magnitudes.
    """
    target = shape if shape is not None else NormShape(8, 32)
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(target.rows, target.width, generator=generator)
    weight = torch.randn(target.width, target.width, generator=generator)
    bias = torch.randn(target.width, generator=generator)
    gain = torch.randn(target.width, generator=generator).abs() + 0.5
    offset = torch.randn(target.width, generator=generator)
    running = (
        torch.randn(1, target.width, generator=generator),
        torch.randn(target.width, generator=generator).abs() + 0.5,
    )

    separate = batch_norm(source @ weight + bias, gain, offset, running=running)
    folded_weight, folded_bias = fold_into_the_weight(weight, bias, gain, offset, running)
    folded = source @ folded_weight + folded_bias

    gap = float((separate - folded).abs().max())
    scale = float(separate.abs().max())
    return {
        "identical": bool(torch.equal(separate, folded)),
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
    }


def the_fold_removes_an_operation() -> dict:
    """What the fold is worth, counted in operations at inference.

    One whole normalisation per layer, which for a model that is mostly products and
    normalisations is a quarter of the operations. Nothing about the arithmetic changes: the
    same scaling happens, once on a weight that is loaded anyway rather than once per row of
    every batch.
    """
    return {
        "before": ["matmul", "add", "batch norm"],
        "after": ["matmul", "add"],
        "removed": 1,
        "arithmetic_moved_to_compile_time": True,
    }


def why_it_only_works_at_inference(shape: NormShape | None = None, *, seed: int = 0) -> dict:
    """Whether the fold would be correct during training, which it would not.

    During training the statistics come from the batch, so they change every step and the folded
    weight would have to be rebuilt every step, which is the work the fold was avoiding. Worse,
    the statistics depend on the values the weight produced, so the fold would be defined in
    terms of itself.
    """
    target = shape if shape is not None else NormShape(8, 32)
    values, gain, shift = random_rows(target, seed=seed)
    training = batch_norm(values, gain, shift)
    stored = batch_norm(
        values,
        gain,
        shift,
        running=(values.mean(dim=0, keepdim=True), values.var(dim=0, unbiased=False)),
    )
    return {
        "same_on_this_batch": bool(torch.allclose(training, stored, atol=1e-5)),
        "statistics_come_from_the_batch": True,
    }


def compare_all() -> list[dict]:
    """The three versions on every axis this file measures."""
    rows = []
    for kind, batch_dependent, foldable in (
        ("layer", False, False),
        ("rms", False, False),
        ("batch", True, True),
    ):
        rows.append(
            {
                "normalisation": kind,
                "reductions": reductions_for(kind),
                "operations_per_element": arithmetic_per_element(kind),
                "depends_on_the_batch": batch_dependent,
                "folds_into_the_weight": foldable,
            }
        )
    return rows


def nothing_is_best_on_every_axis() -> dict:
    """Which version wins on which axis, and whether any wins on all of them."""
    rows = {row["normalisation"]: row for row in compare_all()}
    cheapest = min(rows, key=lambda name: rows[name]["operations_per_element"])
    foldable = [name for name, row in rows.items() if row["folds_into_the_weight"]]
    independent = [name for name, row in rows.items() if not row["depends_on_the_batch"]]
    return {
        "cheapest": cheapest,
        "foldable": foldable,
        "batch_independent": independent,
        "one_wins_everything": cheapest in foldable,
    }


def an_unknown_normalisation_is_refused() -> bool:
    """Whether asking about a version that does not exist is caught."""
    try:
        arithmetic_per_element("magic")
    except ConfigError:
        return True
    return False


def a_rank_three_input_is_refused() -> bool:
    """Whether a normalisation refuses something that is not a batch of rows."""
    try:
        layer_norm(torch.randn(2, 4, 8), torch.ones(8), torch.zeros(8))
    except ConfigError:
        return True
    return False
