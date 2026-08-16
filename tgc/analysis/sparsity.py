from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.analysis.quantize import relative_rms
from tgc.errors import ConfigError

# Throwing away most of a weight, and the three different things that can mean.
#
# Removing the smallest weights and keeping the rest is the cheapest form of compression there
# is, and it comes in shapes. Unstructured pruning removes any element and keeps the best
# possible set at a given count. Block pruning removes whole tiles, which is worse for the same
# count and much better for the hardware, because a tile that is entirely absent is a tile the
# kernel never loads. Two of every four is the compromise the hardware actually implements.
#
# The measurements say three things that the usual framing does not.
#
# Structure is not free and it is not ruinous either. At half density, two of four carries
# thirty seven percent more error than the best possible set of the same size, and four by four
# blocks carry a hundred and twenty five percent more. The tile pattern hurts because a four by
# four tile of a random matrix has no reason to be uniformly small.
#
# The penalty for structure shrinks as the weight thins rather than growing. At ninety percent
# density the block pattern is nearly ten times worse than unstructured; at ten percent it is a
# fifth worse. Both are throwing away most of the norm by then and there is not much left to be
# clever about, so the argument for unstructured pruning is strongest exactly where the
# compression is least useful.
#
# And the storage argument runs opposite to the accuracy argument. Unstructured sparsity needs a
# position per surviving element, and a position is the same size as the value it points at, so
# at half density the format is exactly as large as the dense one it replaced. Structure is not
# a concession to the hardware, it is what makes the compression real.


@dataclass
class SparsityReport:
    """What a pruning left behind."""

    kept: int
    total: int
    error: float = 0.0

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ConfigError(f"a tensor has to have elements, got {self.total}")
        if not 0 <= self.kept <= self.total:
            raise ConfigError(f"cannot keep {self.kept} of {self.total}")

    @property
    def density(self) -> float:
        """Share of the elements that survived."""
        return self.kept / self.total

    @property
    def sparsity(self) -> float:
        """Share that did not."""
        return 1.0 - self.density

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kept": self.kept,
            "total": self.total,
            "density": round(self.density, 4),
            "error": round(self.error, 6),
        }


def prune_unstructured(values: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the largest elements by magnitude and zero the rest.

    The best possible pruning at a given count, by construction: any other set of the same size
    has a larger sum of squared removals. That makes it the baseline everything structured is
    measured against rather than a method anybody deploys.
    """
    _check_density(density)
    if values.numel() == 0:
        raise ConfigError("there is nothing to prune")
    keep = max(round(values.numel() * density), 1)
    flat = values.flatten()
    threshold = flat.abs().sort(descending=True).values[keep - 1]
    mask = flat.abs() >= threshold
    return (flat * mask).reshape(values.shape)


def prune_blocks(values: torch.Tensor, density: float, block: int = 4) -> torch.Tensor:
    """Keep the largest square tiles and zero the rest.

    A tile is scored by the sum of the squares of its elements, which is the right score because
    removing it costs exactly that. Worse than unstructured at the same density and the only
    kind a kernel can skip entirely, since a tile of zeros is a tile that never has to be read.
    """
    _check_density(density)
    if values.dim() != 2:
        raise ConfigError(f"block pruning needs a matrix, got rank {values.dim()}")
    rows, columns = values.shape
    if rows % block or columns % block:
        raise ConfigError(f"a {rows} by {columns} matrix does not divide into {block} blocks")

    tiles = values.reshape(rows // block, block, columns // block, block)
    scores = tiles.pow(2).sum(dim=(1, 3))
    keep = max(round(scores.numel() * density), 1)
    threshold = scores.flatten().sort(descending=True).values[keep - 1]
    mask = (scores >= threshold).to(values.dtype)
    return (tiles * mask.unsqueeze(1).unsqueeze(3)).reshape(rows, columns)


def prune_n_of_m(values: torch.Tensor, keep: int = 2, group: int = 4) -> torch.Tensor:
    """Keep the largest few of every consecutive run.

    The pattern the hardware implements. Every group of four contributes exactly two survivors,
    so the positions can be stored as two bits per group and the kernel knows the layout without
    reading an index array, which is the whole reason it exists.
    """
    if keep < 1 or group < keep:
        raise ConfigError(f"cannot keep {keep} of every {group}")
    if values.numel() % group:
        raise ConfigError(f"{values.numel()} elements do not divide into groups of {group}")

    grouped = values.flatten().reshape(-1, group)
    order = grouped.abs().argsort(dim=1, descending=True)
    mask = torch.zeros_like(grouped)
    mask.scatter_(1, order[:, :keep], 1.0)
    return (grouped * mask).reshape(values.shape)


def _check_density(density: float) -> None:
    """Raise if a density is not a share."""
    if not 0 < density <= 1:
        raise ConfigError(f"the density has to be in (0, 1], got {density}")


def measure(original: torch.Tensor, pruned: torch.Tensor) -> SparsityReport:
    """How much survived a pruning and how far it moved the tensor."""
    return SparsityReport(
        kept=int((pruned != 0).sum()),
        total=original.numel(),
        error=relative_rms(pruned - original, original),
    )


def weight_matrix(rows: int = 128, columns: int = 128, *, seed: int = 0) -> torch.Tensor:
    """A matrix to prune."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator)


def compare_patterns(density: float = 0.5, block: int = 4) -> list[dict]:
    """The three patterns at one density, side by side.

    Unstructured is the floor by construction. Two of four carries a bit over a third more
    error than that floor, which is the price of a pattern a machine can decode without an
    index array. Block pruning more than doubles it, because a four by four tile of a random
    matrix has no reason to be uniformly small.
    """
    _check_density(density)
    values = weight_matrix()
    keep = max(round(4 * density), 1)
    return [
        {
            "pattern": "unstructured",
            **measure(values, prune_unstructured(values, density)).as_dict(),
        },
        {"pattern": f"{keep} of 4", **measure(values, prune_n_of_m(values, keep)).as_dict()},
        {
            "pattern": f"{block} by {block} blocks",
            **measure(values, prune_blocks(values, density, block)).as_dict(),
        },
    ]


def structure_costs_little_accuracy(density: float = 0.5) -> dict:
    """How much the hardware pattern gives up against the best possible one."""
    rows = {row["pattern"]: row for row in compare_patterns(density)}
    best = rows["unstructured"]["error"]
    structured = rows["2 of 4"]["error"]
    blocked = rows["4 by 4 blocks"]["error"]
    return {
        "unstructured": best,
        "two_of_four": structured,
        "blocks": blocked,
        "structure_penalty": round(structured / best, 4) if best else 0.0,
        "block_penalty": round(blocked / best, 4) if best else 0.0,
    }


def density_sweep(
    densities: Sequence[float] = (0.9, 0.75, 0.5, 0.25, 0.1), block: int = 4
) -> list[dict]:
    """Error against density for each pattern.

    Both climb as the density falls and the gap between them narrows, which is the opposite of
    what it looks like it should do. At ninety percent density unstructured pruning removes only
    the very smallest elements and the block method has to take whole tiles including large
    ones, so the ratio is nearly ten. At ten percent both have thrown away most of the norm and
    the ratio is down to a fifth. Being clever about which elements to drop matters most when
    few of them are being dropped.
    """
    if not densities:
        raise ConfigError("there is nothing to sweep")
    values = weight_matrix()
    rows = []
    for density in densities:
        rows.append(
            {
                "density": density,
                "unstructured": round(
                    measure(values, prune_unstructured(values, density)).error, 5
                ),
                "blocks": round(measure(values, prune_blocks(values, density, block)).error, 5),
            }
        )
    return rows


def the_gap_narrows_as_the_weight_thins() -> dict:
    """Whether the block penalty really shrinks rather than growing.

    It shrinks, from a factor of nearly ten at ninety percent density to a fifth at ten percent.
    Written as a check because the intuition points the other way and the intuition is what a
    reader will bring to the sweep above.
    """
    rows = density_sweep()
    first = rows[0]["blocks"] / rows[0]["unstructured"]
    last = rows[-1]["blocks"] / rows[-1]["unstructured"]
    return {
        "ratio_at_ninety_percent": round(first, 4),
        "ratio_at_ten_percent": round(last, 4),
        "narrowed": last < first,
    }


def block_size_sweep(
    sizes: Sequence[int] = (1, 2, 4, 8, 16), density: float = 0.5
) -> list[dict]:
    """Error against how large the pruned tiles are.

    A tile of one is unstructured pruning, so the sweep starts at the floor and climbs from
    there. Every step up is a hardware convenience bought with accuracy, and having the whole
    curve is what makes that a decision rather than a preference.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    values = weight_matrix()
    rows = []
    for size in sizes:
        pruned = prune_blocks(values, density, size)
        rows.append({"block": size, "error": round(measure(values, pruned).error, 5)})
    return rows


def a_block_of_one_is_unstructured(density: float = 0.5) -> dict:
    """Whether the two agree where they should.

    They have to, and it is worth checking rather than assuming, because the block method scores
    tiles by the sum of their squares and the unstructured method scores elements by magnitude,
    and those are the same order only when a tile is a single element.
    """
    values = weight_matrix()
    return {
        "unstructured": round(measure(values, prune_unstructured(values, density)).error, 6),
        "blocks_of_one": round(measure(values, prune_blocks(values, density, 1)).error, 6),
    }


def storage_bytes(elements: int, density: float, *, pattern: str = "unstructured") -> int:
    """Bytes a sparse format needs, indices included.

    An unstructured format stores a value and a position, and the position is four bytes because
    a large weight has more than sixty five thousand elements in it. A two of four format stores
    two bits per group of four and no positions at all. A dense format stores everything and
    nothing else.
    """
    if elements < 1:
        raise ConfigError(f"a tensor has to have elements, got {elements}")
    _check_density(density)
    kept = round(elements * density)
    if pattern == "dense":
        return elements * 4
    if pattern == "unstructured":
        return kept * 4 + kept * 4
    if pattern == "n of m":
        return kept * 4 + (elements + 3) // 4
    raise ConfigError(f"unknown pattern {pattern!r}")


def storage_sweep(
    densities: Sequence[float] = (0.9, 0.5, 0.25, 0.1, 0.05), elements: int = 1 << 20
) -> list[dict]:
    """What each format costs to store, against the dense one.

    The unstructured format is larger than dense until the density falls below a half, which is
    the point of the whole table. Half the elements removed and the file got bigger, because
    every survivor now carries a four byte position it did not need before.
    """
    if not densities:
        raise ConfigError("there is nothing to sweep")
    dense = storage_bytes(elements, 1.0, pattern="dense")
    rows = []
    for density in densities:
        unstructured = storage_bytes(elements, density, pattern="unstructured")
        structured = storage_bytes(elements, density, pattern="n of m")
        rows.append(
            {
                "density": density,
                "dense": dense,
                "unstructured": unstructured,
                "n_of_m": structured,
                "unstructured_saves": round(dense / unstructured, 4),
                "n_of_m_saves": round(dense / structured, 4),
            }
        )
    return rows


def indices_undo_the_saving() -> dict:
    """The density at which an unstructured format stops costing more than dense.

    A half, exactly, because a position is the same size as the value it points at. At that
    density the sparse format and the dense one are the same size to the byte, and everything
    above it is a compression scheme that makes the tensor larger. Worth stating plainly,
    because the accuracy tables never mention it.
    """
    for row in storage_sweep(densities=(0.9, 0.75, 0.6, 0.5, 0.4, 0.25)):
        if row["unstructured_saves"] >= 1.0:
            return {
                "density": row["density"],
                "unstructured_saves": row["unstructured_saves"],
                "n_of_m_saves": row["n_of_m_saves"],
            }
    return {"density": 0.0, "unstructured_saves": 0.0, "n_of_m_saves": 0.0}


def output_error(
    density: float = 0.5, rows: int = 8, inner: int = 128, columns: int = 128, block: int = 4
) -> dict:
    """What each pattern does to the product rather than to the weight.

    Measured for the same reason the quantisation error is: the weight is not what anybody uses.
    The relative error carries through a contraction almost unchanged, which is the same result
    analysis/quantize.py found for rounding error and for the same reason, so the ranking of the
    three patterns on the weight is the ranking on the output.
    """
    _check_density(density)
    generator = torch.Generator().manual_seed(7)
    activations = torch.randn(rows, inner, generator=generator)
    weights = weight_matrix(inner, columns)
    exact = activations @ weights

    keep = max(round(4 * density), 1)
    result = {}
    for label, pruned in (
        ("unstructured", prune_unstructured(weights, density)),
        ("n of m", prune_n_of_m(weights, keep)),
        ("blocks", prune_blocks(weights, density, block)),
    ):
        result[label] = {
            "weight_error": round(relative_rms(pruned - weights, weights), 5),
            "output_error": round(relative_rms(activations @ pruned - exact, exact), 5),
        }
    return result


def the_ranking_survives_the_contraction() -> dict:
    """Whether the order of the three patterns is the same on the output as on the weight."""
    rows = output_error()
    by_weight = sorted(rows, key=lambda label: rows[label]["weight_error"])
    by_output = sorted(rows, key=lambda label: rows[label]["output_error"])
    return {
        "by_weight": by_weight,
        "by_output": by_output,
        "same_order": by_weight == by_output,
    }


def speedup_from(density: float, *, overhead: float = 0.0) -> float:
    """How much faster a kernel gets from skipping the zeros.

    The overhead is the share of the dense time a sparse kernel spends on things a dense one
    does not: decoding positions, gathering rows, dealing with an irregular loop. It is a
    fraction of the dense time rather than of the sparse time, because it does not fall when the
    weight does.
    """
    _check_density(density)
    if overhead < 0:
        raise ConfigError(f"the overhead cannot be {overhead}")
    return 1.0 / (density + overhead)


def overhead_sweep(
    overheads: Sequence[float] = (0.0, 0.05, 0.2, 0.5), density: float = 0.5
) -> list[dict]:
    """Speedup against how much the sparse kernel costs to run at all.

    At half density and no overhead the kernel is twice as fast. At half density and an overhead
    of a fifth it is one and a half times, and at an overhead of a half it is exactly even. That
    last number is the reason unstructured sparsity is rarely faster in practice: gathering
    scattered rows costs about that much.
    """
    if not overheads:
        raise ConfigError("there is nothing to sweep")
    return [
        {"overhead": overhead, "speedup": round(speedup_from(density, overhead=overhead), 4)}
        for overhead in overheads
    ]


def break_even_density(overhead: float = 0.2) -> float:
    """The density below which a sparse kernel is worth running.

    One minus the overhead, which is a two line derivation and worth having as a function
    because the overhead is a property of the kernel and changing it should change the answer
    rather than leaving a remembered number behind.
    """
    if not 0 <= overhead < 1:
        raise ConfigError(f"the overhead has to be in [0, 1), got {overhead}")
    return 1.0 - overhead


def accuracy_against_speed(
    densities: Sequence[float] = (0.9, 0.5, 0.25, 0.1), overhead: float = 0.2
) -> list[dict]:
    """The trade the whole file is about, in one table.

    Error against speedup for the pattern a machine can actually run. The top row is not an
    option at all: at ninety percent density the sparse kernel is slower than the dense one,
    because the overhead it carries is larger than the tenth of the work it skipped. Below that
    the speedup climbs steadily and so does the error, and there is no knee in either curve to
    point at as the right answer.
    """
    if not densities:
        raise ConfigError("there is nothing to sweep")
    values = weight_matrix()
    rows = []
    for density in densities:
        pruned = prune_blocks(values, density, 4)
        rows.append(
            {
                "density": density,
                "error": round(measure(values, pruned).error, 5),
                "speedup": round(speedup_from(density, overhead=overhead), 4),
            }
        )
    return rows
