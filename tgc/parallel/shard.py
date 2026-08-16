from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError, PassError

# Splitting one tensor across several devices, and paying for the pieces that have to meet.
#
# A matrix product has three axes and each one can be split. Splitting the rows of the left
# operand gives every device a slice of the output rows and needs no communication. Splitting
# the columns of the right operand gives every device a slice of the output columns and needs
# none either. Splitting the contracted axis gives every device a partial sum of the whole
# output and needs an all reduce, which is the expensive one.
#
# The reason all three exist is that they compose differently. Two products in a row, which is
# what an mlp is, can be sharded so that the intermediate never has to be gathered: split the
# first on its output columns and the second on its contracted axis, and the piece each device
# holds after the first product is exactly the piece it needs for the second. One all reduce at
# the end instead of a gather in the middle and another at the end. That composition is the
# single most valuable fact about tensor parallelism and it is four lines to check.
#
# Everything here is checked against the unsharded product rather than argued about. The
# splitting is done with real tensors and the pieces are really combined, so a scheme that
# looks right and computes something else fails here rather than later.
#
# Doing that turned up something worth knowing. Splitting the output rows performs exactly the
# same arithmetic in exactly the same groups, and the answer is still not bit identical, at
# float32 or at float64. The library chooses its blocking from the shape of the operands, so a
# product of eight rows accumulates differently from a product of thirty two. A compiler cannot
# see that and should not promise bit equality across a sharding it did not do the maths for.

REPLICATED = -1


@dataclass(frozen=True)
class ShardSpec:
    """How one tensor is spread over a set of devices."""

    axis: int
    devices: int

    def __post_init__(self) -> None:
        if self.devices < 1:
            raise ConfigError(f"there has to be at least one device, got {self.devices}")
        if self.axis < REPLICATED:
            raise ConfigError(f"{self.axis} is not an axis")

    @property
    def is_replicated(self) -> bool:
        """Whether every device holds the whole tensor."""
        return self.axis == REPLICATED

    def shard_shape(self, shape: Sequence[int]) -> list[int]:
        """The shape of the piece one device holds."""
        sizes = list(shape)
        if self.is_replicated:
            return sizes
        if self.axis >= len(sizes):
            raise PassError(f"cannot shard axis {self.axis} of a rank {len(sizes)} tensor")
        if sizes[self.axis] % self.devices:
            raise PassError(
                f"{sizes[self.axis]} does not divide evenly over {self.devices} devices"
            )
        sizes[self.axis] //= self.devices
        return sizes

    def elements_per_device(self, shape: Sequence[int]) -> int:
        """How many numbers one device stores."""
        total = 1
        for size in self.shard_shape(shape):
            total *= size
        return total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "axis": "replicated" if self.is_replicated else self.axis,
            "devices": self.devices,
        }


def replicated(devices: int) -> ShardSpec:
    """The specification that splits nothing."""
    return ShardSpec(axis=REPLICATED, devices=devices)


@dataclass
class MatmulPlan:
    """One way of splitting a matrix product across devices."""

    name: str
    left: ShardSpec
    right: ShardSpec
    needs_all_reduce: bool
    description: str

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "plan": self.name,
            "left": self.left.as_dict()["axis"],
            "right": self.right.as_dict()["axis"],
            "all_reduce": self.needs_all_reduce,
        }


def row_parallel(devices: int) -> MatmulPlan:
    """Split the rows of the left operand.

    Each device computes a horizontal slice of the output and nothing has to be summed. The
    weight is replicated, so this saves no weight memory at all, which is why it is the plan
    for splitting a batch and not the plan for splitting a model.
    """
    return MatmulPlan(
        name="row parallel",
        left=ShardSpec(axis=0, devices=devices),
        right=replicated(devices),
        needs_all_reduce=False,
        description="slice the output rows, replicate the weight",
    )


def column_parallel(devices: int) -> MatmulPlan:
    """Split the columns of the right operand.

    Each device computes a vertical slice of the output. The weight is split, so the weight
    memory falls with the device count, and the activation has to be replicated, which is
    usually much smaller than the weight.
    """
    return MatmulPlan(
        name="column parallel",
        left=replicated(devices),
        right=ShardSpec(axis=1, devices=devices),
        needs_all_reduce=False,
        description="slice the output columns, split the weight",
    )


def contraction_parallel(devices: int) -> MatmulPlan:
    """Split the axis being summed over.

    Each device computes a partial sum of the whole output and the partials have to be added,
    which is the all reduce. The only plan of the three whose cost grows with the size of the
    output rather than staying flat, and the only one that composes with column parallel to
    avoid a gather.
    """
    return MatmulPlan(
        name="contraction parallel",
        left=ShardSpec(axis=1, devices=devices),
        right=ShardSpec(axis=0, devices=devices),
        needs_all_reduce=True,
        description="slice the contracted axis, sum the partials",
    )


PLANS = (row_parallel, column_parallel, contraction_parallel)


def shard(tensor: torch.Tensor, spec: ShardSpec) -> list[torch.Tensor]:
    """The pieces one specification cuts a tensor into."""
    if spec.is_replicated:
        return [tensor for _ in range(spec.devices)]
    spec.shard_shape(list(tensor.shape))
    return list(tensor.chunk(spec.devices, dim=spec.axis))


def run_plan(left: torch.Tensor, right: torch.Tensor, plan: MatmulPlan) -> torch.Tensor:
    """Run a matrix product under a plan and put the pieces back together.

    Really split and really recombined, because a plan that is described correctly and
    implemented wrongly produces a tensor of the right shape full of the wrong numbers, and
    nothing about the description would say so.
    """
    lefts = shard(left, plan.left)
    rights = shard(right, plan.right)
    partials = [one @ other for one, other in zip(lefts, rights, strict=True)]

    if plan.needs_all_reduce:
        total = partials[0]
        for piece in partials[1:]:
            total = total + piece
        return total
    if plan.left.axis == 0:
        return torch.cat(partials, dim=0)
    return torch.cat(partials, dim=1)


def check_plan(
    rows: int = 32, inner: int = 64, columns: int = 48, devices: int = 4, *, seed: int = 0
) -> list[dict]:
    """Every plan against the unsharded product, on the same numbers.

    Only column parallel comes out bit identical. Contraction parallel is not expected to be:
    summing four partial sums adds the terms in a different order than one contraction does, and
    floating point addition is not associative.

    Row parallel is the surprise. Slicing the output rows performs exactly the same arithmetic
    on exactly the same numbers, so it should be bit identical and it is not. The reason is
    below the compiler entirely: the library picks its blocking from the shape of the operands,
    so a product of eight rows accumulates in a different order than a product of thirty two.
    """
    if min(rows, inner, columns, devices) < 1:
        raise ConfigError("every dimension has to be positive")
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(rows, inner, generator=generator)
    right = torch.randn(inner, columns, generator=generator)
    exact = left @ right

    results = []
    for build in PLANS:
        plan = build(devices)
        result = run_plan(left, right, plan)
        gap = float((result - exact).abs().max())
        results.append(
            {
                "plan": plan.name,
                "shape_matches": list(result.shape) == list(exact.shape),
                "bit_identical": bool(torch.equal(result, exact)),
                "largest_gap": gap,
            }
        )
    return results


def every_plan_computes_the_same_product(tolerance: float = 1e-5) -> bool:
    """Whether all three plans agree with the unsharded product."""
    return all(row["shape_matches"] and row["largest_gap"] <= tolerance for row in check_plan())


def only_the_reduction_loses_a_bit() -> dict:
    """Which plans are exact and which are not, and by how much."""
    rows = {row["plan"]: row for row in check_plan()}
    return {
        "row_parallel_exact": rows["row parallel"]["bit_identical"],
        "column_parallel_exact": rows["column parallel"]["bit_identical"],
        "contraction_parallel_exact": rows["contraction parallel"]["bit_identical"],
        "contraction_gap": rows["contraction parallel"]["largest_gap"],
    }


@dataclass
class Traffic:
    """What one plan costs, per device."""

    plan: str
    weight_elements: int
    activation_elements: int
    communicated_elements: int

    @property
    def stored(self) -> int:
        """Numbers one device has to hold."""
        return self.weight_elements + self.activation_elements

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "plan": self.plan,
            "weight_per_device": self.weight_elements,
            "activation_per_device": self.activation_elements,
            "communicated": self.communicated_elements,
        }


def measure_traffic(
    plan: MatmulPlan, rows: int = 32, inner: int = 64, columns: int = 48
) -> Traffic:
    """How much one device stores and how much crosses the link.

    The communication figure is the volume an all reduce moves in a ring, which is twice the
    output size scaled by one less than the device count over the device count. Two of the
    three plans move nothing, and that is the whole reason to prefer them where they fit.
    """
    left_shape = [rows, inner]
    right_shape = [inner, columns]
    activation = plan.left.elements_per_device(left_shape)
    weight = plan.right.elements_per_device(right_shape)

    communicated = 0
    if plan.needs_all_reduce:
        devices = plan.left.devices
        communicated = int(2 * rows * columns * (devices - 1) / devices)
    return Traffic(
        plan=plan.name,
        weight_elements=weight,
        activation_elements=activation,
        communicated_elements=communicated,
    )


def even_an_exact_split_is_not_bit_identical(seed: int = 0) -> dict:
    """Row parallel at two precisions, to show where its gap comes from.

    It is not the sharding. The split performs the same multiplications and the same additions
    in the same groups, and the answer still differs, at both precisions and by about the
    rounding unit of each. What changed is the blocking the library chose for a shorter matrix,
    which is a thing a compiler can neither see nor control and has to expect.
    """
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(32, 64, generator=generator)
    right = torch.randn(64, 48, generator=generator)

    rows = {}
    for label, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        wide_left = left.to(dtype)
        wide_right = right.to(dtype)
        exact = wide_left @ wide_right
        split = torch.cat([piece @ wide_right for piece in wide_left.chunk(4)], dim=0)
        rows[label] = {
            "bit_identical": bool(torch.equal(split, exact)),
            "largest_gap": float((split - exact).abs().max()),
        }
    return rows


def compare_plans(devices: int = 4) -> list[dict]:
    """What each plan stores and moves, side by side."""
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    return [measure_traffic(build(devices)).as_dict() for build in PLANS]


def only_column_parallel_shrinks_the_weight(devices: int = 4) -> dict:
    """Which plans reduce the memory a device needs for the weight.

    Column parallel and contraction parallel do and row parallel does not, which decides the
    question for a model too large to fit. Splitting the batch does not make a model smaller,
    however many devices it is split over.
    """
    rows = {row["plan"]: row for row in compare_plans(devices)}
    single = {row["plan"]: row for row in compare_plans(1)}
    return {
        plan: rows[plan]["weight_per_device"] < single[plan]["weight_per_device"]
        for plan in rows
    }


def mlp_needs_one_all_reduce(devices: int = 4) -> dict:
    """The composition worth knowing, checked rather than asserted.

    Column parallel then contraction parallel. The first product leaves each device holding the
    columns of the intermediate that correspond to the rows of the second weight it holds, so
    the intermediate never has to be gathered and the only communication in the pair is the one
    all reduce at the end.

    Compared against the naive alternative of gathering after the first product, which moves the
    intermediate as well and is larger than the output by the expansion factor.
    """
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    rows, hidden, expansion = 32, 64, 4
    wide = hidden * expansion

    fused = measure_traffic(contraction_parallel(devices), rows, wide, hidden)
    gather = int(2 * rows * wide * (devices - 1) / devices)
    return {
        "composed": fused.communicated_elements,
        "with_a_gather_in_the_middle": fused.communicated_elements + gather,
        "saving": gather,
        "ratio": round((fused.communicated_elements + gather) / fused.communicated_elements, 3)
        if fused.communicated_elements
        else 0.0,
    }


def run_sharded_mlp(
    rows: int = 32, hidden: int = 64, expansion: int = 4, devices: int = 4, *, seed: int = 0
) -> dict:
    """The composition run on real tensors and compared with the unsharded version.

    The check that the description above is a description of what the code does. Each device
    gets a slice of the first weight's columns and the matching slice of the second weight's
    rows, computes its own piece end to end, and only the final partial sums are added.
    """
    if min(rows, hidden, expansion, devices) < 1:
        raise ConfigError("every dimension has to be positive")
    wide = hidden * expansion
    if wide % devices:
        raise ConfigError(f"{wide} does not divide over {devices} devices")

    generator = torch.Generator().manual_seed(seed)
    activation = torch.randn(rows, hidden, generator=generator)
    up = torch.randn(hidden, wide, generator=generator)
    down = torch.randn(wide, hidden, generator=generator)
    exact = torch.relu(activation @ up) @ down

    up_pieces = up.chunk(devices, dim=1)
    down_pieces = down.chunk(devices, dim=0)
    total = None
    for up_piece, down_piece in zip(up_pieces, down_pieces, strict=True):
        partial = torch.relu(activation @ up_piece) @ down_piece
        total = partial if total is None else total + partial

    gap = float((total - exact).abs().max())
    scale = float(exact.abs().max())
    return {
        "devices": devices,
        "largest_gap": gap,
        "relative_gap": gap / scale if scale else gap,
        "shape_matches": list(total.shape) == list(exact.shape),
    }


def the_composition_holds_at_every_device_count(
    counts: Sequence[int] = (1, 2, 4, 8, 16), tolerance: float = 1e-4
) -> list[dict]:
    """The sharded mlp against the unsharded one, over a range of device counts."""
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in counts:
        row = run_sharded_mlp(devices=count)
        row["agrees"] = row["relative_gap"] <= tolerance
        rows.append(row)
    return rows


def the_relu_is_why_it_works() -> dict:
    """Why the composition needs the activation to be elementwise.

    A relu applied to a slice of the columns is the same as a slice of the relu applied to all
    of them, so the nonlinearity can happen before anything is gathered. A softmax cannot, and
    that is the reason attention is sharded by head and an mlp is sharded by column: the
    reduction inside the softmax spans the axis the sharding would split.
    """
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(8, 32, generator=generator)
    pieces = values.chunk(4, dim=1)

    relu_after = torch.relu(values)
    relu_before = torch.cat([torch.relu(piece) for piece in pieces], dim=1)

    softmax_after = torch.softmax(values, dim=1)
    softmax_before = torch.cat([torch.softmax(piece, dim=1) for piece in pieces], dim=1)
    return {
        "relu_commutes": bool(torch.equal(relu_after, relu_before)),
        "softmax_does_not": not bool(torch.allclose(softmax_after, softmax_before)),
        "softmax_gap": float((softmax_after - softmax_before).abs().max()),
    }


def arithmetic_per_device(rows: int, inner: int, columns: int, devices: int) -> float:
    """The multiply adds one device performs under any of these plans."""
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    return 2.0 * rows * inner * columns / devices


def scaling_sweep(
    counts: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    rows: int = 32,
    inner: int = 256,
    columns: int = 256,
    link_ratio: float = 100.0,
) -> list[dict]:
    """Arithmetic and communication per device as the device count grows.

    The arithmetic falls like one over the device count. The communication rises toward twice
    the output size and then flattens, because the ring factor approaches one. So the ratio of
    the two gets worse without limit, and the only question is where it crosses.

    The link ratio is how many arithmetic operations the machine does in the time it moves one
    number. A hundred is roughly a fast interconnect next to a fast accelerator.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    if link_ratio <= 0:
        raise ConfigError(f"the link ratio has to be positive, got {link_ratio}")
    result = []
    for count in counts:
        plan = contraction_parallel(count)
        traffic = measure_traffic(plan, rows, inner, columns)
        compute = arithmetic_per_device(rows, inner, columns, count)
        communication = traffic.communicated_elements * link_ratio
        result.append(
            {
                "devices": count,
                "arithmetic": compute,
                "communication_cost": communication,
                "communication_share": round(communication / (compute + communication), 4)
                if compute + communication
                else 0.0,
            }
        )
    return result


def where_communication_takes_over(threshold: float = 0.5, **kwargs) -> int:
    """The device count past which more than half the time is the link.

    The number that decides how far a model can usefully be split. It moves with the shape of
    the product and with the machine, which is why it is computed rather than remembered.
    """
    for row in scaling_sweep(**kwargs):
        if row["communication_share"] >= threshold:
            return int(row["devices"])
    return 0


def a_bigger_product_scales_further() -> list[dict]:
    """How the crossover moves with the size of the thing being split.

    Further out for a larger product, which is the one piece of good news in the sweep. The
    arithmetic grows with three dimensions and the communication with two, so a model twice as
    wide in every direction pushes the crossover out rather than hitting it sooner.
    """
    rows = []
    for size in (64, 128, 256, 512):
        rows.append(
            {
                "size": size,
                "crossover": where_communication_takes_over(inner=size, columns=size),
            }
        )
    return rows


def resharding_cost(source: ShardSpec, target: ShardSpec, shape: Sequence[int]) -> int:
    """How many numbers move to convert one layout into another.

    Nothing if the two specifications agree. Everything if one is replicated and the other is
    not, or if they split different axes, because every device ends up needing a piece it does
    not have. There is no cheap resharding, which is why a compiler should be choosing layouts
    to avoid one rather than costing it accurately.
    """
    if source.devices != target.devices:
        raise PassError(f"cannot reshard between {source.devices} and {target.devices} devices")
    if source == target:
        return 0
    total = 1
    for size in shape:
        total *= size
    if target.is_replicated:
        return total * (source.devices - 1) // source.devices * source.devices
    return total


def resharding_table(shape: Sequence[int] = (64, 64), devices: int = 4) -> list[dict]:
    """The cost of every conversion between the layouts in use."""
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    specs = [
        replicated(devices),
        ShardSpec(axis=0, devices=devices),
        ShardSpec(axis=1, devices=devices),
    ]
    rows = []
    for source in specs:
        for target in specs:
            rows.append(
                {
                    "from": source.as_dict()["axis"],
                    "to": target.as_dict()["axis"],
                    "elements": resharding_cost(source, target, shape),
                }
            )
    return rows


def staying_put_is_free(shape: Sequence[int] = (64, 64), devices: int = 4) -> dict:
    """How many of the conversions cost nothing, which is only the ones that change nothing."""
    rows = resharding_table(shape, devices)
    free = [row for row in rows if row["elements"] == 0]
    return {
        "conversions": len(rows),
        "free": len(free),
        "all_free_ones_are_no_ops": all(row["from"] == row["to"] for row in free),
    }
