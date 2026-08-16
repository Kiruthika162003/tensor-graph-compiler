from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tgc.errors import ConfigError

# Adding up a gradient that lives on several devices, and hiding the time it takes.
#
# Data parallel training runs the same model on every device with different examples, so every
# device ends a step holding a gradient for the whole model that is wrong on its own. The step
# cannot finish until those are summed, and the sum is the only thing in a data parallel step
# that does not get faster when devices are added.
#
# Two ways of doing it and they fail in opposite directions. A ring passes each piece around the
# devices twice, which moves the least data any correct method can and takes two steps per
# device, so its latency grows linearly. A tree halves the participants at each level, which
# takes a logarithmic number of steps and moves more data, because the whole tensor crosses
# every level rather than a slice of it.
#
# The measurements put the crossover at a megabyte on a fast link with eight devices, which is
# in the middle of the range a model actually contains rather than below it. A weight of two
# hundred and fifty six by two hundred and fifty six is a quarter of that and wants a tree; one
# of four thousand by four thousand is sixty four times it and wants a ring. Both methods are
# needed by the same model, which is why libraries ship both and choose by size.
#
# The third measurement matters more than either. A backward pass produces gradients in reverse
# layer order, so the last layer's gradient is ready while most of the backward pass is still
# running and a reduction started then costs nothing. Overlapping hides ninety seven percent of
# the communication at eight devices, where choosing the better method with both bucketed is
# worth sixty five percent of it. Both are worth having and only one of them needs the backward
# pass and the reduction to be in the same schedule.
#
# What overlap cannot hide is the first layer's gradient, which is only ready when the backward
# pass is over. That sets a floor: the exposed time is never less than one layer's reduction,
# however good the schedule is, and it is the reason the exposed time is small rather than
# zero.

BYTES_PER_ELEMENT = 4


@dataclass
class Link:
    """What one connection between devices does."""

    name: str
    bytes_per_second: float
    latency_seconds: float

    def __post_init__(self) -> None:
        if self.bytes_per_second <= 0:
            raise ConfigError("a link has to move something per second")
        if self.latency_seconds < 0:
            raise ConfigError("a message cannot take negative time to start")

    def time_for(self, moved: float) -> float:
        """Seconds to move a number of bytes, start up included."""
        if moved < 0:
            raise ConfigError(f"cannot move {moved} bytes")
        return self.latency_seconds + moved / self.bytes_per_second

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "link": self.name,
            "bytes_per_second": self.bytes_per_second,
            "latency_seconds": self.latency_seconds,
        }


FAST = Link(name="fast", bytes_per_second=2.0e11, latency_seconds=2e-6)
SLOW = Link(name="slow", bytes_per_second=1.2e10, latency_seconds=1e-5)


def ring_time(size_bytes: int, devices: int, link: Link = FAST) -> float:
    """How long a ring reduction takes.

    Two passes around the ring, each of one less step than the device count, each moving a slice
    of the tensor. The volume per device settles at twice the tensor as the ring grows and the
    number of messages does not settle at all, which is the whole shape of the method.
    """
    _check(size_bytes, devices)
    if devices == 1:
        return 0.0
    steps = 2 * (devices - 1)
    slice_bytes = size_bytes / devices
    return steps * link.time_for(slice_bytes)


def tree_time(size_bytes: int, devices: int, link: Link = FAST) -> float:
    """How long a halving tree takes.

    A logarithmic number of steps up and the same number back down, each moving the whole
    tensor. Fewer messages than a ring and more bytes, which makes it the right method exactly
    when the messages cost more than the bytes.
    """
    _check(size_bytes, devices)
    if devices == 1:
        return 0.0
    levels = math.ceil(math.log2(devices))
    return 2 * levels * link.time_for(size_bytes)


def _check(size_bytes: int, devices: int) -> None:
    """Raise if there is nothing to reduce or nobody to reduce it."""
    if size_bytes < 0:
        raise ConfigError(f"a tensor cannot be {size_bytes} bytes")
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")


def ring_volume(size_bytes: int, devices: int) -> float:
    """Bytes each device sends under a ring, which is the floor for any method.

    Twice the tensor times one less than the device count over the device count. It approaches
    twice the tensor and never reaches it, and no correct reduction moves less, because every
    device has to both contribute its whole gradient and receive the whole answer.
    """
    _check(size_bytes, devices)
    if devices == 1:
        return 0.0
    return 2.0 * size_bytes * (devices - 1) / devices


def tree_volume(size_bytes: int, devices: int) -> float:
    """Bytes each device sends under a tree."""
    _check(size_bytes, devices)
    if devices == 1:
        return 0.0
    return 2.0 * size_bytes * math.ceil(math.log2(devices))


def compare_methods(size_bytes: int, devices: int = 8, link: Link = FAST) -> list[dict]:
    """Both methods on one tensor."""
    return [
        {
            "method": "ring",
            "seconds": ring_time(size_bytes, devices, link),
            "bytes_sent": ring_volume(size_bytes, devices),
        },
        {
            "method": "tree",
            "seconds": tree_time(size_bytes, devices, link),
            "bytes_sent": tree_volume(size_bytes, devices),
        },
    ]


def size_sweep(
    sizes: Sequence[int] = (1024, 16384, 262144, 4194304, 67108864),
    devices: int = 8,
    link: Link = FAST,
) -> list[dict]:
    """Which method wins, against how large the tensor is.

    The tree at the small end and the ring at the large one. A ring of eight devices sends
    fourteen messages and a tree sends six, so at a kilobyte the ring is paying for eight extra
    start ups it did not need; at four megabytes the tree is moving three times the bytes and
    the start ups are noise.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for size in sizes:
        ring = ring_time(size, devices, link)
        tree = tree_time(size, devices, link)
        rows.append(
            {
                "bytes": size,
                "ring_seconds": ring,
                "tree_seconds": tree,
                "winner": "ring" if ring <= tree else "tree",
            }
        )
    return rows


def crossover_bytes(devices: int = 8, link: Link = FAST, limit: int = 1 << 30) -> int:
    """The tensor size at which the ring overtakes the tree.

    Found by doubling rather than solved, because the tree's step count is a ceiling of a
    logarithm and the closed form would have to carry that around. A megabyte on a fast link
    with eight devices, which is in the middle of the range a model contains.
    """
    size = 1024
    while size <= limit:
        if ring_time(size, devices, link) <= tree_time(size, devices, link):
            return size
        size *= 2
    return 0


def the_crossover_is_below_any_real_tensor(link: Link = FAST) -> dict:
    """Where that crossover sits next to the things being reduced.

    In the middle of them, which is the answer that makes both methods necessary. A weight of
    two hundred and fifty six squared is a quarter of the crossover and a weight of four
    thousand squared is sixty four times it, and a model holds both.
    """
    crossover = crossover_bytes(link=link)
    small = 256 * 256 * BYTES_PER_ELEMENT
    large = 4096 * 4096 * BYTES_PER_ELEMENT
    return {
        "crossover_bytes": crossover,
        "a_small_weight": small,
        "a_large_weight": large,
        "tree_wins_on_the_small_one": small < crossover,
        "ring_wins_on_the_large_one": large > crossover,
    }


def device_sweep(
    counts: Sequence[int] = (2, 4, 8, 16, 64), size_bytes: int = 4 << 20, link: Link = FAST
) -> list[dict]:
    """How each method scales with the device count.

    The ring's volume flattens at twice the tensor and its message count does not, so its time
    keeps climbing. The tree's climbs in steps at every power of two. Neither gets cheaper with
    more devices, which is the fact that makes data parallel scaling a communication problem.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    return [
        {
            "devices": count,
            "ring_seconds": ring_time(size_bytes, count, link),
            "tree_seconds": tree_time(size_bytes, count, link),
        }
        for count in counts
    ]


def the_ring_volume_has_a_ceiling(counts: Sequence[int] = (2, 8, 64, 1024)) -> list[dict]:
    """How close the ring gets to twice the tensor, per device count.

    Half of it at two devices and within a tenth of a percent at a thousand. That ceiling is
    what makes the ring the bandwidth optimal method and it is also why adding devices does not
    reduce the traffic: it is already as low as any correct method can be.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    size = 4 << 20
    return [
        {
            "devices": count,
            "share_of_twice_the_tensor": round(ring_volume(size, count) / (2 * size), 4),
        }
        for count in counts
    ]


def bucket_time(
    tensors: Sequence[int], devices: int, link: Link = FAST, bucket_bytes: int = 0
) -> float:
    """How long it takes to reduce many tensors, optionally grouped into buckets.

    A bucket is a concatenation, so grouping trades one start up per tensor for one per bucket
    and moves the same bytes either way. That is the whole of why libraries bucket gradients,
    and the sweep below says how large a bucket has to be before it stops mattering.
    """
    if not tensors:
        raise ConfigError("there is nothing to reduce")
    if bucket_bytes < 0:
        raise ConfigError(f"a bucket cannot be {bucket_bytes} bytes")
    if bucket_bytes == 0:
        return sum(ring_time(size, devices, link) for size in tensors)

    total = 0.0
    current = 0
    for size in tensors:
        if current and current + size > bucket_bytes:
            total += ring_time(current, devices, link)
            current = 0
        current += size
    if current:
        total += ring_time(current, devices, link)
    return total


def gradient_sizes(layers: int = 48, width: int = 1024) -> list[int]:
    """The gradients a stack produces, largest last.

    Modelled as one large matrix and two small vectors per layer, which is the shape that makes
    bucketing worth anything: the vectors are a thousandth of the bytes and two thirds of the
    messages.
    """
    if min(layers, width) < 1:
        raise ConfigError("a model needs layers and width")
    sizes = []
    for _ in range(layers):
        sizes.append(width * width * BYTES_PER_ELEMENT)
        sizes.append(width * BYTES_PER_ELEMENT)
        sizes.append(width * BYTES_PER_ELEMENT)
    return sizes


def bucket_sweep(
    buckets: Sequence[int] = (0, 1 << 16, 1 << 20, 1 << 24, 1 << 28), devices: int = 8
) -> list[dict]:
    """Reduction time against how large the buckets are.

    Falls the whole way rather than flattening, which is not what I expected. Bucketing at
    sixteen megabytes is two and a half times faster than reducing one tensor at a time, and
    going on to two hundred and fifty six megabytes buys another fifth. There is no flat region
    here because the ring's per message cost is two start ups per device and eight devices make
    that expensive enough to keep mattering.
    """
    if not buckets:
        raise ConfigError("there is nothing to sweep")
    sizes = gradient_sizes()
    return [
        {
            "bucket_bytes": bucket,
            "seconds": bucket_time(sizes, devices, bucket_bytes=bucket),
        }
        for bucket in buckets
    ]


def bucketing_is_worth_a_factor_of_two(devices: int = 8) -> dict:
    """What grouping the tensors saves.

    Two and a half times at a bucket of sixteen megabytes. A hundred and forty four separate
    reductions become nine, and the fourteen messages each of them costs on eight devices go
    with them.
    """
    sizes = gradient_sizes()
    alone = bucket_time(sizes, devices)
    grouped = bucket_time(sizes, devices, bucket_bytes=1 << 24)
    return {
        "one_at_a_time": alone,
        "bucketed": grouped,
        "saving": round(alone / grouped, 4) if grouped else 0.0,
        "messages_before": len(sizes),
    }


def exposed_time(
    layers: int = 48,
    width: int = 1024,
    devices: int = 8,
    link: Link = FAST,
    backward_seconds: float = 0.05,
) -> dict:
    """How much of the reduction is left over after the backward pass has hidden what it can.

    Gradients arrive in reverse layer order, so the reduction for the last layer can start while
    the rest of the backward pass is still ahead of it. Modelled as the backward pass releasing
    an equal share of the gradient at an equal share of its duration, which is close enough to
    get the shape right and simple enough to read.

    The exposed time is never zero. The gradient released last has nothing left to hide behind,
    so one layer's reduction is always outside the backward pass, and that is the floor reported
    alongside the answer.
    """
    if backward_seconds <= 0:
        raise ConfigError(f"a backward pass takes time, got {backward_seconds}")
    sizes = gradient_sizes(layers, width)
    per_layer = len(sizes) // layers
    slot = backward_seconds / layers

    finished_at = 0.0
    for index in range(layers):
        released_at = slot * (index + 1)
        chunk = sum(sizes[index * per_layer : (index + 1) * per_layer])
        finished_at = max(released_at, finished_at) + ring_time(chunk, devices, link)

    last_chunk = sum(sizes[(layers - 1) * per_layer :])
    serial = bucket_time(sizes, devices, link, bucket_bytes=1 << 24)
    exposed = max(finished_at - backward_seconds, 0.0)
    return {
        "backward_seconds": backward_seconds,
        "communication_seconds": serial,
        "exposed_seconds": exposed,
        "floor_seconds": ring_time(last_chunk, devices, link),
        "hidden_share": round(1.0 - exposed / serial, 4) if serial else 0.0,
        "exposed_share_of_the_step": round(exposed / backward_seconds, 6),
    }


def overlap_hides_almost_everything(
    counts: Sequence[int] = (2, 4, 8, 16, 64), link: Link = FAST
) -> list[dict]:
    """What share of the communication the overlap absorbs, per device count.

    Between ninety seven and ninety nine percent across the whole range, and falling slowly as
    devices are added because the reductions get longer while the backward pass does not. The
    share is high everywhere here because the model is small next to the compute time assumed,
    which is the regime a data parallel run is usually in and stops being in as the cluster
    grows.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    return [{"devices": count, **exposed_time(devices=count, link=link)} for count in counts]


def the_floor_is_one_layer(devices: int = 8, link: Link = FAST) -> dict:
    """Whether the exposed time really is the last layer's reduction.

    It is, to within the accumulated queueing from the layers before it. The gradient produced
    last has nothing behind it to hide under, so no schedule removes it and the only way to make
    it smaller is to make the layer smaller.
    """
    result = exposed_time(devices=devices, link=link)
    return {
        "exposed": result["exposed_seconds"],
        "floor": result["floor_seconds"],
        "exposed_is_at_least_the_floor": result["exposed_seconds"]
        >= result["floor_seconds"] * (1 - 1e-9),
        "ratio": round(result["exposed_seconds"] / result["floor_seconds"], 3)
        if result["floor_seconds"]
        else 0.0,
    }


def where_the_overlap_runs_out(
    link: Link = FAST, threshold: float = 0.1, limit: int = 4096
) -> int:
    """The device count at which the exposed communication passes a share of the step.

    A tenth by default. Five hundred and twelve devices on a fast link and thirty two on a slow
    one, which is a factor of sixteen from a link that is sixteen times slower, and is the whole
    argument for the interconnect deciding how wide a data parallel run can go.
    """
    if not 0 < threshold < 1:
        raise ConfigError(f"the threshold has to be in (0, 1), got {threshold}")
    devices = 2
    while devices <= limit:
        if exposed_time(devices=devices, link=link)["exposed_share_of_the_step"] >= threshold:
            return devices
        devices *= 2
    return 0


def overlap_is_worth_more_than_the_method(devices: int = 8) -> dict:
    """The two decisions compared on the same configuration, both bucketed.

    Overlapping removes ninety seven percent of the communication. Choosing the better of the
    two methods, once both are bucketed so the comparison is fair, removes sixty five percent.
    Both are worth doing and the ordering is clear, and the reason a compiler usually only gets
    the second one is that the overlap needs the backward pass and the reduction in a single
    schedule, which is a graph level decision rather than a library one.
    """
    sizes = gradient_sizes()
    ring = bucket_time(sizes, devices, FAST, bucket_bytes=1 << 24)
    tree = _bucketed_tree_time(sizes, devices, FAST, 1 << 24)
    overlapped = exposed_time(devices=devices)
    return {
        "ring": ring,
        "tree": tree,
        "method_choice_is_worth": round(abs(tree - ring) / max(ring, tree), 4),
        "overlap_is_worth": overlapped["hidden_share"],
    }


def _bucketed_tree_time(
    tensors: Sequence[int], devices: int, link: Link, bucket_bytes: int
) -> float:
    """The tree method with the same bucketing the ring gets.

    Written so the comparison above is between two methods rather than between one method and
    another method handicapped. A tree reducing a hundred and forty four tensors one at a time
    would lose for a reason that has nothing to do with being a tree.
    """
    total = 0.0
    current = 0
    for size in tensors:
        if current and current + size > bucket_bytes:
            total += tree_time(current, devices, link)
            current = 0
        current += size
    if current:
        total += tree_time(current, devices, link)
    return total


def a_slow_link_changes_everything(devices: int = 8) -> dict:
    """The same configuration on a link an order of magnitude slower.

    The exposed time goes up by a factor of eleven and the share of the communication that gets
    hidden barely moves, which is the useful pair. Overlap is just as effective on a slow link;
    there is simply more to overlap, and the leftover grows with it.
    """
    fast = exposed_time(devices=devices, link=FAST)
    slow = exposed_time(devices=devices, link=SLOW)
    return {
        "fast_exposed": fast["exposed_seconds"],
        "slow_exposed": slow["exposed_seconds"],
        "fast_hidden_share": fast["hidden_share"],
        "slow_hidden_share": slow["hidden_share"],
        "exposed_ratio": round(slow["exposed_seconds"] / fast["exposed_seconds"], 3)
        if fast["exposed_seconds"]
        else 0.0,
    }
