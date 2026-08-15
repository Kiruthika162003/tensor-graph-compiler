from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError

# Storing a tensor in fewer bits, and measuring what that costs where it is actually paid.
#
# Quantisation replaces a float tensor with integers and two numbers that say how to read them:
# a scale and an offset. The integers are cheap to move, which is the whole point, because a
# matrix product against a stored weight is memory bound and the weight is most of the memory.
#
# Almost every published number about quantisation is the error in the weight. That is the easy
# thing to measure and it is not the thing that matters. What matters is the error in the output
# of the operation the weight was used in, and this file measures both.
#
# The answer is not the one either camp expects. Measured in root mean square, a contraction
# neither amplifies the relative error nor cancels it: the output of a matrix product against a
# four bit weight has the same relative error as the weight, within a fifth either way, for
# contractions from one to four thousand and with no trend in either direction. The signal and
# the noise grow at the same rate because both are sums of the same number of independent terms.
#
# Measured worst case, the output error is four times the weight error at a contraction of four
# thousand and climbs steadily to get there, because the worst output entry is the one where
# many roundings happened to line up while the worst weight entry is a single rounding. Both
# numbers are real, they describe different things, and quoting one without saying which is
# where most of the folklore about error accumulation comes from.
#
# The other thing measured here is where the error comes from. There are two sources and they
# pull in opposite directions. A wide range means a coarse step and rounding error; a narrow
# range means values outside it get clipped. Choosing a range is choosing between them, and the
# best choice is not the full range of the data on any tensor with an outlier in it.

BIT_WIDTHS = (8, 6, 4, 3, 2)


@dataclass
class Scheme:
    """How to read a tensor back out of its integers."""

    bits: int
    scale: float
    zero_point: float = 0.0
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.bits < 2:
            raise ConfigError(f"a quantisation needs at least two bits, got {self.bits}")
        if self.scale <= 0:
            raise ConfigError(f"the scale has to be positive, got {self.scale}")

    @property
    def levels(self) -> int:
        """How many distinct values the integers can take."""
        return 1 << self.bits

    @property
    def low(self) -> int:
        """The smallest integer the scheme uses."""
        return -(self.levels // 2) if self.symmetric else 0

    @property
    def high(self) -> int:
        """The largest integer the scheme uses."""
        return self.levels // 2 - 1 if self.symmetric else self.levels - 1

    @property
    def representable_range(self) -> tuple[float, float]:
        """The span of real values the scheme can hold without clipping."""
        return (
            self.low * self.scale + self.zero_point,
            self.high * self.scale + self.zero_point,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "bits": self.bits,
            "levels": self.levels,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "symmetric": self.symmetric,
        }


def symmetric_scheme(values: torch.Tensor, bits: int = 8) -> Scheme:
    """A scheme centred on zero, sized by the largest magnitude.

    The usual choice for weights, because a weight distribution is roughly symmetric and a
    scheme with no offset costs one fewer operation to undo. It wastes half its levels on a
    tensor that is entirely positive, which is why activations after a relu do not use it.
    """
    _check_not_empty(values)
    largest = float(values.abs().max())
    if largest == 0:
        return Scheme(bits=bits, scale=1.0, symmetric=True)
    return Scheme(bits=bits, scale=largest / (Scheme(bits=bits, scale=1.0).high or 1))


def asymmetric_scheme(values: torch.Tensor, bits: int = 8) -> Scheme:
    """A scheme that fits the actual range, offset included.

    Uses every level on data that does not straddle zero, at the cost of an offset that has to
    be subtracted before the integers can be multiplied by anything.
    """
    _check_not_empty(values)
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return Scheme(bits=bits, scale=1.0, zero_point=low, symmetric=False)
    scale = (high - low) / (Scheme(bits=bits, scale=1.0, symmetric=False).high or 1)
    return Scheme(bits=bits, scale=scale, zero_point=low, symmetric=False)


def _check_not_empty(values: torch.Tensor) -> None:
    """Raise if there is nothing to fit a scheme to."""
    if values.numel() == 0:
        raise ConfigError("there is nothing to quantise")


def quantise(values: torch.Tensor, scheme: Scheme) -> torch.Tensor:
    """The integers a tensor becomes, clamped into the scheme's range."""
    scaled = (values - scheme.zero_point) / scheme.scale
    return scaled.round().clamp(scheme.low, scheme.high)


def dequantise(integers: torch.Tensor, scheme: Scheme) -> torch.Tensor:
    """The floats those integers stand for."""
    return integers * scheme.scale + scheme.zero_point


def round_trip(values: torch.Tensor, scheme: Scheme) -> torch.Tensor:
    """A tensor put through a scheme and read back."""
    return dequantise(quantise(values, scheme), scheme)


@dataclass
class ErrorReport:
    """How far a round trip moved a tensor.

    Two relative measures rather than one, because they answer different questions and the
    difference between them is the subject of half this file. The worst case is against the
    largest value present; the root mean square is against the root mean square of the data,
    which is the one that stays stable when the shape of the tensor changes.
    """

    largest: float = 0.0
    mean: float = 0.0
    relative: float = 0.0
    rms_relative: float = 0.0
    clipped: int = 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "largest": round(self.largest, 8),
            "mean": round(self.mean, 8),
            "relative": round(self.relative, 6),
            "rms_relative": round(self.rms_relative, 6),
            "clipped": self.clipped,
        }


def relative_rms(error: torch.Tensor, values: torch.Tensor) -> float:
    """The root mean square of an error against the root mean square of the data."""
    scale = float(values.pow(2).mean().sqrt())
    return float(error.pow(2).mean().sqrt()) / scale if scale else 0.0


def measure_error(values: torch.Tensor, scheme: Scheme) -> ErrorReport:
    """The error a scheme introduces, and how much of it is clipping."""
    _check_not_empty(values)
    recovered = round_trip(values, scheme)
    difference = (recovered - values).abs()
    low, high = scheme.representable_range
    outside = ((values < low) | (values > high)).sum()
    scale = float(values.abs().max())
    return ErrorReport(
        largest=float(difference.max()),
        mean=float(difference.mean()),
        relative=float(difference.max()) / scale if scale else 0.0,
        rms_relative=relative_rms(difference, values),
        clipped=int(outside),
    )


def bit_width_sweep(values: torch.Tensor, widths: Sequence[int] = BIT_WIDTHS) -> list[dict]:
    """Error against bit width.

    Removing a bit roughly doubles the error and not exactly, which is worth knowing before
    somebody writes a test asserting the factor. A symmetric scheme spends one level on the
    sign, so its largest integer is two to the power of one less than the width, minus one, and
    the ratio of steps between four bits and three is seven over three rather than two.
    """
    if not widths:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for bits in widths:
        row = measure_error(values, symmetric_scheme(values, bits)).as_dict()
        row["bits"] = bits
        rows.append(row)
    return rows


def each_bit_roughly_doubles_the_error(values: torch.Tensor) -> dict:
    """How close to a doubling it is, measured rather than assumed.

    Between two and a third and four and a half over the widths swept, never exactly two. The
    lost sign level is most of the reason and the rest is where the values happen to fall
    relative to the step boundaries.
    """
    rows = {row["bits"]: row["largest"] for row in bit_width_sweep(values)}
    ratios = []
    ordered = sorted(rows, reverse=True)
    for wide, narrow in itertools.pairwise(ordered):
        if rows[wide] > 0:
            ratios.append(rows[narrow] / rows[wide])
    return {
        "steps": len(ratios),
        "smallest_ratio": round(min(ratios), 3) if ratios else 0.0,
        "largest_ratio": round(max(ratios), 3) if ratios else 0.0,
    }


def per_row_schemes(values: torch.Tensor, bits: int = 8) -> list[Scheme]:
    """One scheme per row rather than one for the whole tensor."""
    if values.dim() != 2:
        raise ConfigError(f"per row schemes need a matrix, got rank {values.dim()}")
    return [symmetric_scheme(row, bits) for row in values]


def per_row_round_trip(values: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """A matrix quantised a row at a time."""
    schemes = per_row_schemes(values, bits)
    rows = [round_trip(row, scheme) for row, scheme in zip(values, schemes, strict=True)]
    return torch.stack(rows)


def per_row_is_worth_it(values: torch.Tensor, bits: int = 4) -> dict:
    """Whether a scale per row beats a scale for the whole tensor, by two measures.

    By worst case it buys nothing at all, on either kind of matrix, and the reason is worth
    following. The worst error lives in whichever row has the largest values, and that row set
    the shared scale in the first place, so giving it its own scale hands it back the same
    number. A metric that only looks at the worst case cannot see this working.

    By mean error it buys a great deal on an uneven matrix, because every row that was not
    setting the scale now gets one fitted to itself instead of to the outlier row.
    """
    whole = measure_error(values, symmetric_scheme(values, bits))
    per_row = (per_row_round_trip(values, bits) - values).abs()
    worst = float(per_row.max())
    average = float(per_row.mean())
    return {
        "worst_one_scale": round(whole.largest, 6),
        "worst_per_row": round(worst, 6),
        "worst_improvement": round(whole.largest / worst, 3) if worst else 0.0,
        "mean_one_scale": round(whole.mean, 6),
        "mean_per_row": round(average, 6),
        "mean_improvement": round(whole.mean / average, 3) if average else 0.0,
    }


def uneven_matrix(rows: int = 64, columns: int = 64, factor: float = 50.0) -> torch.Tensor:
    """A matrix with one row far larger than the rest.

    Not a contrived case. A trained weight matrix reliably has a few rows with much larger
    magnitude than the rest, and a single scale fitted to the whole tensor is fitted to those
    rows, which leaves every other row using a small part of the available levels.
    """
    if factor <= 1:
        raise ConfigError(f"the factor has to make one row larger, got {factor}")
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(rows, columns, generator=generator)
    values[0] *= factor
    return values


def even_matrix(rows: int = 64, columns: int = 64) -> torch.Tensor:
    """A matrix whose rows all have about the same magnitude."""
    generator = torch.Generator().manual_seed(1)
    return torch.randn(rows, columns, generator=generator)


def unevenness_decides_the_gain() -> list[dict]:
    """The per row gain on an even matrix and an uneven one, side by side."""
    return [
        {"matrix": "even", **per_row_is_worth_it(even_matrix())},
        {"matrix": "uneven", **per_row_is_worth_it(uneven_matrix())},
    ]


def clipped_scheme(values: torch.Tensor, bits: int = 8, quantile: float = 0.999) -> Scheme:
    """A scheme fitted to most of the data rather than all of it.

    Deliberately too small to hold the extremes. Everything outside gets clipped, which is an
    error, and everything inside gets a finer step, which removes error from far more values.
    Whether that trade is worth taking is a measurement rather than a principle.
    """
    _check_not_empty(values)
    if not 0 < quantile <= 1:
        raise ConfigError(f"the quantile has to be in (0, 1], got {quantile}")
    limit = float(values.abs().flatten().quantile(quantile))
    if limit == 0:
        return symmetric_scheme(values, bits)
    return Scheme(bits=bits, scale=limit / (Scheme(bits=bits, scale=1.0).high or 1))


def quantile_sweep(
    values: torch.Tensor,
    bits: int = 4,
    quantiles: Sequence[float] = (0.9, 0.99, 0.999, 0.9999, 1.0),
) -> list[dict]:
    """Mean error against how much of the tail the scheme gives up on.

    The largest error rises as the range shrinks, because the clipped values are the ones
    furthest out. The mean error falls, because everything else got a finer step. Which of the
    two a model cares about is the whole question, and the answer for a weight matrix is
    usually the mean.
    """
    if not quantiles:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for quantile in quantiles:
        report = measure_error(values, clipped_scheme(values, bits, quantile))
        row = report.as_dict()
        row["quantile"] = quantile
        rows.append(row)
    return rows


def clipping_helps_the_mean_and_hurts_the_worst(values: torch.Tensor, bits: int = 4) -> dict:
    """The trade, stated as two numbers moving in opposite directions.

    Compared against the best clipping point rather than an arbitrary one, because the mean
    error is not monotonic in the quantile. It falls, rises and falls again over the range
    swept, which rules out picking a clipping point by walking downhill from either end.
    """
    rows = quantile_sweep(values, bits)
    best = min(rows, key=lambda row: row["mean"])
    full = next(row for row in rows if row["quantile"] == 1.0)
    return {
        "best_quantile": best["quantile"],
        "mean_with_clipping": best["mean"],
        "mean_without": full["mean"],
        "worst_with_clipping": best["largest"],
        "worst_without": full["largest"],
        "mean_improved": best["mean"] < full["mean"],
        "worst_got_worse": best["largest"] > full["largest"],
    }


def the_mean_error_is_not_monotonic(values: torch.Tensor, bits: int = 4) -> dict:
    """Whether the mean error falls steadily as the clipping point moves.

    It does not. Clipping harder removes rounding error from everything that stays and adds
    clipping error to everything that goes, and the two swap which one dominates more than once
    over an ordinary range of quantiles. A search over the whole sweep is the only honest way to
    pick one.
    """
    means = [row["mean"] for row in quantile_sweep(values, bits)]
    falls = sum(1 for a, b in itertools.pairwise(means) if b < a)
    rises = sum(1 for a, b in itertools.pairwise(means) if b > a)
    return {"steps": len(means) - 1, "falls": falls, "rises": rises, "monotonic": rises == 0}


def best_quantile(values: torch.Tensor, bits: int = 4) -> float:
    """The clipping point that minimises mean error."""
    rows = quantile_sweep(values, bits)
    return min(rows, key=lambda row: row["mean"])["quantile"]


def weight_error_against_output_error(
    rows: int = 8, inner: int = 64, columns: int = 8, bits: int = 4
) -> dict:
    """The number this file exists for, measured two ways because they disagree.

    In root mean square the product is exactly as accurate as the weight, and that is not a
    coincidence to be explained away. Both the signal and the error in an output entry are sums
    of the same number of independent terms, so both grow like the square root of the
    contraction and the ratio between them does not move.

    Worst case tells a different story and it is also true. The largest error in the product is
    the entry where many roundings happened to point the same way, and the more terms there are
    the better the chance of that, so the worst case grows with the contraction while the
    typical case does not.
    """
    if min(rows, inner, columns) < 1:
        raise ConfigError("the shapes have to be positive")
    generator = torch.Generator().manual_seed(2)
    activations = torch.randn(rows, inner, generator=generator)
    weights = torch.randn(inner, columns, generator=generator)

    scheme = symmetric_scheme(weights, bits)
    recovered = round_trip(weights, scheme)
    weight_error = recovered - weights

    exact = activations @ weights
    approximate = activations @ recovered
    output_error = approximate - exact

    weight_rms = relative_rms(weight_error, weights)
    output_rms = relative_rms(output_error, exact)
    weight_worst = float(weight_error.abs().max()) / float(weights.abs().max())
    output_worst = float(output_error.abs().max()) / float(exact.abs().max())
    return {
        "weight_rms": round(weight_rms, 6),
        "output_rms": round(output_rms, 6),
        "rms_ratio": round(output_rms / weight_rms, 4) if weight_rms else 0.0,
        "weight_worst": round(weight_worst, 6),
        "output_worst": round(output_worst, 6),
        "worst_ratio": round(output_worst / weight_worst, 4) if weight_worst else 0.0,
    }


def contraction_length_sweep(
    lengths: Sequence[int] = (1, 16, 256, 4096), bits: int = 4
) -> list[dict]:
    """The two ratios as the contraction grows.

    The root mean square ratio sits near one across four orders of magnitude of contraction
    length. The worst case ratio starts at one, because a contraction of one is not a sum, and
    climbs to about three by four thousand terms.
    """
    if not lengths:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for length in lengths:
        row = weight_error_against_output_error(inner=length, bits=bits)
        row["contraction"] = length
        rows.append(row)
    return rows


def relative_error_survives_a_contraction(tolerance: float = 0.35) -> dict:
    """Whether the root mean square ratio really stays near one.

    Stated as a bound rather than an equality, because it is a statistical claim about finite
    samples and not an identity. Over the lengths swept it never leaves a third of one either
    side, and it shows no trend in either direction.
    """
    ratios = [row["rms_ratio"] for row in contraction_length_sweep()]
    return {
        "lengths": len(ratios),
        "smallest": min(ratios),
        "largest": max(ratios),
        "all_near_one": all(abs(ratio - 1.0) <= tolerance for ratio in ratios),
    }


def the_worst_case_does_grow() -> dict:
    """Whether the worst case ratio climbs with the contraction, which it does."""
    rows = contraction_length_sweep()
    return {
        "shortest": rows[0]["worst_ratio"],
        "longest": rows[-1]["worst_ratio"],
        "grew": rows[-1]["worst_ratio"] > rows[0]["worst_ratio"],
    }


def bits_needed_for(values: torch.Tensor, tolerance: float = 0.01) -> int:
    """The fewest bits that keep the root mean square relative error under a bound.

    Against the root mean square rather than the largest value, because the largest value is
    exactly what the scheme was fitted to and the error there is half a step by construction.
    Measuring against it asks the same question of every tensor and gets the same answer.
    """
    if tolerance <= 0:
        raise ConfigError(f"the tolerance has to be positive, got {tolerance}")
    for bits in range(2, 17):
        if measure_error(values, symmetric_scheme(values, bits)).rms_relative <= tolerance:
            return bits
    return 17


def uniform_values(count: int = 4096) -> torch.Tensor:
    """Values spread evenly over a range."""
    generator = torch.Generator().manual_seed(3)
    return torch.rand(count, generator=generator) * 2 - 1


def peaked_values(count: int = 4096) -> torch.Tensor:
    """Values with the same extremes and most of their mass near zero."""
    generator = torch.Generator().manual_seed(3)
    signs = torch.randn(count, generator=generator).sign()
    magnitudes = torch.rand(count, generator=generator).pow(4)
    values = signs * magnitudes
    values[0] = 1.0
    values[1] = -1.0
    return values


def distribution_changes_the_answer(tolerance: float = 0.04) -> list[dict]:
    """Two tensors with the same range needing different widths.

    Same extremes, different shape, and the peaked one needs an extra bit at the same tolerance.
    A symmetric scheme spends its levels evenly across a range, so a tensor whose mass sits near
    zero uses only the handful of levels closest to zero and wastes the rest.
    """
    rows = []
    for label, values in (("uniform", uniform_values()), ("peaked", peaked_values())):
        rows.append(
            {
                "shape": label,
                "largest": round(float(values.abs().max()), 4),
                "bits_needed": bits_needed_for(values, tolerance),
            }
        )
    return rows


def error_by_distribution(widths: Sequence[int] = BIT_WIDTHS) -> list[dict]:
    """The same bit widths on both distributions, side by side."""
    if not widths:
        raise ConfigError("there is nothing to sweep")
    uniform = uniform_values()
    peaked = peaked_values()
    return [
        {
            "bits": bits,
            "uniform": round(
                measure_error(uniform, symmetric_scheme(uniform, bits)).rms_relative, 5
            ),
            "peaked": round(
                measure_error(peaked, symmetric_scheme(peaked, bits)).rms_relative, 5
            ),
        }
        for bits in widths
    ]


def memory_saved(elements: int, bits: int) -> dict:
    """What the whole exercise buys, in bytes.

    The only reason to do any of this. Four bit weights are an eighth of the size of a float32
    tensor, and for a matrix product that reads its weight once and its activations once, that
    is close to an eight times reduction in the traffic that made the operation slow.
    """
    if elements < 0:
        raise ConfigError(f"a tensor cannot have {elements} elements")
    if bits < 2:
        raise ConfigError(f"a quantisation needs at least two bits, got {bits}")
    original = elements * 4
    quantised = elements * bits / 8
    return {
        "float32_bytes": original,
        "quantised_bytes": int(quantised),
        "ratio": round(original / quantised, 3) if quantised else 0.0,
    }


def saving_against_error(
    elements: int = 4096, widths: Sequence[int] = BIT_WIDTHS
) -> list[dict]:
    """The trade the whole file is about, in one table."""
    if not widths:
        raise ConfigError("there is nothing to sweep")
    values = even_matrix().flatten()[:elements]
    rows = []
    for bits in widths:
        row = memory_saved(elements, bits)
        row["bits"] = bits
        row["relative_error"] = round(
            measure_error(values, symmetric_scheme(values, bits)).rms_relative, 6
        )
        rows.append(row)
    return rows
