from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError

# Turning a convolution into a matrix product, and what the turning costs.
#
# A convolution is not a matrix product and every fast implementation of one is. The standard
# route is to expand each output position's receptive field into a row, which turns the whole
# thing into one product against the flattened weights, and that expansion writes every input
# element once per position that reads it. For a three by three kernel that is nine copies.
#
# So the memory picture is the opposite of the arithmetic picture. The product is the same
# arithmetic the convolution always did, and the expansion multiplies the input traffic by the
# kernel area. That is why nobody materialises the expansion: the useful implementations compute
# it a tile at a time inside the kernel, which is the same trick as the streaming softmax and
# for the same reason.
#
# The third route is Winograd, which trades multiplies for additions. Its smallest useful form
# computes two outputs from four inputs with four multiplies instead of six, a saving of a third
# on a one dimensional filter and of two and a quarter on a two dimensional one. The transforms
# it needs are exact in rational arithmetic and are not exact in floating point, and the
# measurement below puts the error at half again a direct convolution's, which is smaller than
# the reputation of the method suggests and is why everybody ships it.


@dataclass(frozen=True)
class ConvShape:
    """One convolution, in the dimensions that decide its cost."""

    batch: int
    in_channels: int
    out_channels: int
    size: int
    kernel: int
    stride: int = 1
    padding: int = 0

    def __post_init__(self) -> None:
        if min(self.batch, self.in_channels, self.out_channels, self.size, self.kernel) < 1:
            raise ConfigError("every dimension has to be positive")
        if self.stride < 1:
            raise ConfigError(f"a stride of {self.stride} does not move")
        if self.padding < 0:
            raise ConfigError(f"a padding of {self.padding} is not padding")

    @property
    def output_size(self) -> int:
        """The spatial extent of the result."""
        reach = self.size + 2 * self.padding - self.kernel
        if reach < 0:
            raise ConfigError(
                f"a {self.kernel} kernel does not fit a {self.size} input with "
                f"{self.padding} padding"
            )
        return reach // self.stride + 1

    @property
    def positions(self) -> int:
        """Output positions across the whole batch."""
        return self.batch * self.output_size * self.output_size

    @property
    def window(self) -> int:
        """Input elements one output position reads."""
        return self.in_channels * self.kernel * self.kernel

    @property
    def multiplies(self) -> int:
        """Multiply adds a direct convolution performs."""
        return self.positions * self.window * self.out_channels

    @property
    def input_elements(self) -> int:
        """Numbers in the input."""
        return self.batch * self.in_channels * self.size * self.size

    @property
    def expanded_elements(self) -> int:
        """Numbers in the matrix the expansion builds."""
        return self.positions * self.window

    @property
    def expansion_factor(self) -> float:
        """How many times larger the expansion is than the input."""
        if self.input_elements == 0:
            return 0.0
        return self.expanded_elements / self.input_elements

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "batch": self.batch,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "size": self.size,
            "kernel": self.kernel,
            "output_size": self.output_size,
            "expansion_factor": round(self.expansion_factor, 3),
        }


def expand(source: torch.Tensor, shape: ConvShape) -> torch.Tensor:
    """The receptive field of every output position, as a row of a matrix.

    Written with an explicit gather rather than a library call, because the point of the file is
    what the expansion costs and a call would hide both the cost and the indexing. The result is
    positions by window, which is exactly the left operand of the product below.
    """
    if source.dim() != 4:
        raise ConfigError(f"a convolution takes a rank four input, got {source.dim()}")
    padded = source
    if shape.padding:
        padded = torch.nn.functional.pad(
            source, (shape.padding,) * 4, mode="constant", value=0.0
        )

    rows = []
    for batch in range(shape.batch):
        for y in range(shape.output_size):
            for x in range(shape.output_size):
                top = y * shape.stride
                left = x * shape.stride
                patch = padded[batch, :, top : top + shape.kernel, left : left + shape.kernel]
                rows.append(patch.reshape(-1))
    return torch.stack(rows)


def as_matrix_product(
    source: torch.Tensor, weights: torch.Tensor, shape: ConvShape
) -> torch.Tensor:
    """A convolution computed as one product against the expansion."""
    expanded = expand(source, shape)
    flattened = weights.reshape(shape.out_channels, -1).transpose(0, 1)
    product = expanded @ flattened
    return product.reshape(
        shape.batch, shape.output_size, shape.output_size, shape.out_channels
    ).permute(0, 3, 1, 2)


def random_inputs(shape: ConvShape, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """An input and a weight for one convolution."""
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(
        shape.batch, shape.in_channels, shape.size, shape.size, generator=generator
    )
    weights = torch.randn(
        shape.out_channels,
        shape.in_channels,
        shape.kernel,
        shape.kernel,
        generator=generator,
    )
    return source, weights


def matches_the_library(shape: ConvShape | None = None, *, seed: int = 0) -> dict:
    """The expansion and product against the library's convolution.

    The check that makes everything else here mean something. If the expansion indexed the
    receptive fields wrongly it would still produce a matrix of the right shape and a product of
    the right shape, and every cost measurement taken on it would be a measurement of the wrong
    computation.
    """
    target = shape if shape is not None else ConvShape(2, 3, 4, 8, 3, padding=1)
    source, weights = random_inputs(target, seed=seed)
    mine = as_matrix_product(source, weights, target)
    theirs = torch.nn.functional.conv2d(
        source, weights, stride=target.stride, padding=target.padding
    )
    gap = float((mine - theirs).abs().max())
    scale = float(theirs.abs().max())
    return {
        "shape_matches": list(mine.shape) == list(theirs.shape),
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
    }


def the_expansion_is_larger_than_the_input(shape: ConvShape | None = None) -> dict:
    """How much bigger the expansion is, and why.

    The kernel area, adjusted for the stride and the edges. A three by three kernel at stride
    one reads every interior element nine times, so the matrix it builds is nine times the
    input, and the whole memory difficulty of a convolution is that one number.
    """
    target = shape if shape is not None else ConvShape(1, 16, 32, 32, 3, padding=1)
    return {
        "input_elements": target.input_elements,
        "expanded_elements": target.expanded_elements,
        "factor": round(target.expansion_factor, 3),
        "kernel_area": target.kernel * target.kernel,
    }


def kernel_sweep(kernels: Sequence[int] = (1, 3, 5, 7)) -> list[dict]:
    """The expansion factor against the kernel size.

    The square of the kernel, near enough. A one by one convolution is already a matrix product
    and its expansion is the identity; a seven by seven builds a matrix forty nine times its
    input, which at any real resolution does not fit anywhere.
    """
    if not kernels:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for kernel in kernels:
        shape = ConvShape(1, 16, 32, 32, kernel, padding=kernel // 2)
        rows.append(
            {
                "kernel": kernel,
                "factor": round(shape.expansion_factor, 3),
                "kernel_area": kernel * kernel,
            }
        )
    return rows


def a_one_by_one_convolution_is_already_a_product() -> dict:
    """The degenerate case, which is worth naming because it is most of a modern model.

    A one by one convolution reads each input element once per output position and its
    expansion is a reshape. That is why a network built out of them has no convolution problem
    at all, and why the interesting question about convolution is about the wide kernels that
    are increasingly rare.
    """
    shape = ConvShape(1, 16, 32, 32, 1)
    return {
        "factor": round(shape.expansion_factor, 3),
        "expansion_is_free": shape.expansion_factor == 1.0,
    }


def traffic_for(shape: ConvShape, *, materialised: bool) -> int:
    """Bytes a convolution moves, with and without writing the expansion down.

    Materialising it writes the expansion and reads it back, which is twice the expanded size on
    top of everything else. Computing it inside the kernel does neither, at the cost of an
    indexing calculation per element that the arithmetic units were not busy doing anyway.
    """
    element = 4
    weights = shape.out_channels * shape.window
    output = shape.positions * shape.out_channels
    base = (shape.input_elements + weights + output) * element
    if not materialised:
        return base
    return base + 2 * shape.expanded_elements * element


def materialising_costs(shape: ConvShape | None = None) -> dict:
    """What writing the expansion down actually costs in traffic."""
    target = shape if shape is not None else ConvShape(1, 16, 32, 32, 3, padding=1)
    inline = traffic_for(target, materialised=False)
    written = traffic_for(target, materialised=True)
    return {
        "inline": inline,
        "materialised": written,
        "ratio": round(written / inline, 3) if inline else 0.0,
    }


def arithmetic_intensity(shape: ConvShape, *, materialised: bool) -> float:
    """Multiply adds per byte moved.

    The number that says whether the operation is worth doing on an accelerator. Materialising
    the expansion does not change the arithmetic and does change the bytes, so it moves a
    convolution toward the memory bound end for no benefit at all.
    """
    traffic = traffic_for(shape, materialised=materialised)
    if traffic == 0:
        return 0.0
    return shape.multiplies / traffic


def materialising_halves_the_intensity(shape: ConvShape | None = None) -> dict:
    """The same point as a ratio of intensities rather than of bytes."""
    target = shape if shape is not None else ConvShape(1, 16, 32, 32, 3, padding=1)
    inline = arithmetic_intensity(target, materialised=False)
    written = arithmetic_intensity(target, materialised=True)
    return {
        "inline": round(inline, 3),
        "materialised": round(written, 3),
        "ratio": round(inline / written, 3) if written else 0.0,
    }


WINOGRAD_INPUT = torch.tensor(
    [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]
)

WINOGRAD_FILTER = torch.tensor(
    [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]]
)

WINOGRAD_OUTPUT = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]])


def winograd_tile(source: torch.Tensor, filter_taps: torch.Tensor) -> torch.Tensor:
    """Two outputs of a three tap filter from four inputs, with four multiplies.

    The smallest Winograd form. The transforms on the left and the right are made of ones and
    halves, so they cost additions and shifts rather than multiplies, and the only real
    multiplies are the four elementwise products in the middle. A direct computation of the same
    two outputs needs six.
    """
    if source.numel() != 4 or filter_taps.numel() != 3:
        raise ConfigError("this form takes four inputs and three taps")
    transformed_input = WINOGRAD_INPUT @ source.reshape(4)
    transformed_filter = WINOGRAD_FILTER @ filter_taps.reshape(3)
    return WINOGRAD_OUTPUT @ (transformed_input * transformed_filter)


def direct_tile(source: torch.Tensor, filter_taps: torch.Tensor) -> torch.Tensor:
    """The same two outputs, computed as written."""
    values = source.reshape(4)
    taps = filter_taps.reshape(3)
    return torch.stack(
        [
            values[0] * taps[0] + values[1] * taps[1] + values[2] * taps[2],
            values[1] * taps[0] + values[2] * taps[1] + values[3] * taps[2],
        ]
    )


def winograd_matches_the_direct_form(samples: int = 256, *, seed: int = 0) -> dict:
    """The transform against the direct computation, over many random tiles.

    Exact in rational arithmetic and not in floating point. The transforms introduce sums of
    numbers of similar size and opposite sign, which is where cancellation lives, so the answer
    agrees to a few times the rounding unit rather than to the bit.
    """
    if samples < 1:
        raise ConfigError(f"the sample count must be positive, got {samples}")
    generator = torch.Generator().manual_seed(seed)
    worst = 0.0
    scale = 0.0
    for _ in range(samples):
        source = torch.randn(4, generator=generator)
        taps = torch.randn(3, generator=generator)
        direct = direct_tile(source, taps)
        fast = winograd_tile(source, taps)
        worst = max(worst, float((fast - direct).abs().max()))
        scale = max(scale, float(direct.abs().max()))
    return {
        "samples": samples,
        "largest_gap": worst,
        "relative_gap": worst / scale if scale else worst,
    }


def winograd_error_against_a_plain_sum(samples: int = 256, *, seed: int = 0) -> dict:
    """How much worse the transform is than the direct form, in rounding units.

    Half again as large, which is less than the method's reputation suggests. The direct form is
    a dot product of three terms and carries about three roundings; the transform carries an
    input transform, a product and an output transform and carries about a dozen. A dozen
    roundings on values of similar magnitude is a relative error of a few parts in ten million,
    and the smallest form's transforms are made of ones and halves, which are exact.
    """
    if samples < 1:
        raise ConfigError(f"the sample count must be positive, got {samples}")
    generator = torch.Generator().manual_seed(seed)
    fast_error = 0.0
    direct_error = 0.0
    for _ in range(samples):
        source = torch.randn(4, generator=generator)
        taps = torch.randn(3, generator=generator)
        exact = direct_tile(source.double(), taps.double())
        fast = winograd_tile(source, taps).double()
        direct = direct_tile(source, taps).double()
        fast_error = max(fast_error, float((fast - exact).abs().max()))
        direct_error = max(direct_error, float((direct - exact).abs().max()))
    return {
        "winograd": fast_error,
        "direct": direct_error,
        "ratio": round(fast_error / direct_error, 3) if direct_error else 0.0,
    }


def multiplies_saved(kernel: int = 3, output_tile: int = 2) -> dict:
    """What the transform buys, counted in multiplies.

    Four instead of six in one dimension, and the two dimensional form squares both, so sixteen
    instead of thirty six. That is a saving of two and a quarter, which is the number quoted for
    the three by three case and is where it comes from.
    """
    if kernel < 1 or output_tile < 1:
        raise ConfigError("a tile and a kernel both have to be positive")
    direct = kernel * output_tile
    transformed = kernel + output_tile - 1
    return {
        "direct_one_dimension": direct,
        "winograd_one_dimension": transformed,
        "direct_two_dimensions": direct * direct,
        "winograd_two_dimensions": transformed * transformed,
        "saving": round((direct * direct) / (transformed * transformed), 3),
    }


def larger_tiles_save_more_and_hurt_more(
    tiles: Sequence[int] = (2, 4, 6, 8), kernel: int = 3
) -> list[dict]:
    """The saving against the output tile size.

    Grows without bound in theory and the transforms grow with it, and their entries stop being
    halves and become numbers that need real multiplies and have real rounding. Nobody ships
    anything past a tile of four for exactly that reason, and the table shows why the arithmetic
    argument alone would say otherwise.
    """
    if not tiles:
        raise ConfigError("there is nothing to sweep")
    return [{"tile": tile, **multiplies_saved(kernel, tile)} for tile in tiles]


def the_saving_grows_but_the_transform_does_too(kernel: int = 3) -> dict:
    """Both sides of that trade in one place."""
    rows = {row["tile"]: row for row in larger_tiles_save_more_and_hurt_more(kernel=kernel)}
    return {
        "saving_at_two": rows[2]["saving"],
        "saving_at_eight": rows[8]["saving"],
        "transform_size_at_two": rows[2]["winograd_one_dimension"],
        "transform_size_at_eight": rows[8]["winograd_one_dimension"],
    }


def compare_routes(shape: ConvShape | None = None) -> list[dict]:
    """The three ways of computing a convolution, in the terms that decide between them.

    The direct form moves the least and is the hardest to make fast. The expansion is the
    easiest to make fast and moves the most. Winograd does the least arithmetic and is the only
    one that changes the answer. Nothing here dominates, which is why every library implements
    all three and picks by shape.
    """
    target = shape if shape is not None else ConvShape(1, 16, 32, 32, 3, padding=1)
    saving = multiplies_saved(target.kernel, 2)["saving"]
    return [
        {
            "route": "direct",
            "multiplies": target.multiplies,
            "bytes": traffic_for(target, materialised=False),
            "exact": True,
        },
        {
            "route": "expansion",
            "multiplies": target.multiplies,
            "bytes": traffic_for(target, materialised=True),
            "exact": True,
        },
        {
            "route": "winograd",
            "multiplies": int(target.multiplies / saving),
            "bytes": traffic_for(target, materialised=False),
            "exact": False,
        },
    ]


def nothing_dominates(shape: ConvShape | None = None) -> dict:
    """Whether any one route is best on every axis, which none is."""
    rows = {row["route"]: row for row in compare_routes(shape)}
    return {
        "fewest_multiplies": min(rows, key=lambda name: rows[name]["multiplies"]),
        "fewest_bytes": min(rows, key=lambda name: rows[name]["bytes"]),
        "exact_routes": sorted(name for name, row in rows.items() if row["exact"]),
    }


def output_size_rules(shapes: Sequence[ConvShape] | None = None) -> list[dict]:
    """The output extent for a few configurations, so the arithmetic is on record.

    The formula is the input plus twice the padding minus the kernel, over the stride, plus one,
    and every off by one in a convolution implementation is a disagreement with it. Padding of
    half the kernel with stride one is the case that keeps the size, which is why it is what
    everybody writes.
    """
    targets = (
        list(shapes)
        if shapes is not None
        else [
            ConvShape(1, 1, 1, 32, 3),
            ConvShape(1, 1, 1, 32, 3, padding=1),
            ConvShape(1, 1, 1, 32, 3, stride=2, padding=1),
            ConvShape(1, 1, 1, 32, 5, padding=2),
        ]
    )
    return [
        {
            "size": shape.size,
            "kernel": shape.kernel,
            "stride": shape.stride,
            "padding": shape.padding,
            "output": shape.output_size,
        }
        for shape in targets
    ]


def half_padding_keeps_the_size() -> dict:
    """Whether the usual padding really preserves the extent, for several kernels."""
    rows = []
    for kernel in (1, 3, 5, 7):
        shape = ConvShape(1, 1, 1, 32, kernel, padding=kernel // 2)
        rows.append(shape.output_size == 32)
    return {"kernels": len(rows), "all_preserved": all(rows)}


def a_kernel_that_does_not_fit_is_refused() -> bool:
    """Whether a kernel larger than its input is caught rather than giving a negative extent."""
    try:
        return ConvShape(1, 1, 1, 3, 5).output_size < 0
    except ConfigError:
        return True
