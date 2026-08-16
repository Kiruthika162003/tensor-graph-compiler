from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tgc.errors import ConfigError

# What training costs in memory, which is mostly not the model.
#
# A parameter needs four bytes. Training it needs a gradient, another four, and whatever the
# optimiser keeps between steps, which for the one everybody uses is two more copies. Then
# mixed precision adds a half precision copy of the parameter to compute with and keeps the
# original as the thing that gets updated, because a half precision parameter plus a small
# update is the parameter again.
#
# Adding that up gives sixteen bytes per parameter before a single activation is stored, which
# is the number this file exists to derive rather than to quote. Four for the master parameter,
# two for the narrow copy, two for the narrow gradient, and eight for the two optimiser moments.
#
# Three things fall out of writing it down.
#
# The optimiser state is half of it. Sharding just that across devices, which costs nothing in
# arithmetic because each device only updates the parameters it holds, takes sixteen bytes per
# parameter to five and a half at eight devices and to four and a fifth at sixty four. It stops
# there: four of the sixteen bytes are the narrow copy and the narrow gradient, and every device
# needs both of those whole.
#
# The update is memory bound by thirty five to one. Adam does ten arithmetic operations per
# parameter and moves twenty six bytes, and halving the arithmetic changes the time by nothing
# at all. The only lever on the optimiser step is the traffic.
#
# And activations overtake parameters immediately, not eventually. A model of a hundred million
# parameters costs one and a half gigabytes to train; a single layer of activations at a batch
# of thirty two and a sequence of two thousand costs three, and at forty eight layers the
# weights are one percent of the total. Everything about training memory is about activations.

BYTES_PER_FLOAT32 = 4
BYTES_PER_FLOAT16 = 2


@dataclass(frozen=True)
class OptimizerKind:
    """One optimiser, and how many copies of the parameters it keeps."""

    name: str
    states: int
    flops_per_parameter: float

    def __post_init__(self) -> None:
        if self.states < 0:
            raise ConfigError(f"{self.name} cannot keep {self.states} states")
        if self.flops_per_parameter < 0:
            raise ConfigError(f"{self.name} cannot do negative work")

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "optimizer": self.name,
            "states": self.states,
            "flops_per_parameter": self.flops_per_parameter,
        }


SGD = OptimizerKind(name="sgd", states=0, flops_per_parameter=2.0)
MOMENTUM = OptimizerKind(name="momentum", states=1, flops_per_parameter=4.0)
ADAM = OptimizerKind(name="adam", states=2, flops_per_parameter=10.0)

OPTIMIZERS = (SGD, MOMENTUM, ADAM)


@dataclass
class MemoryBreakdown:
    """Bytes per parameter, split by what holds them."""

    master: int = 0
    narrow_copy: int = 0
    gradient: int = 0
    state: int = 0

    @property
    def total(self) -> int:
        """Bytes a single parameter costs to train."""
        return self.master + self.narrow_copy + self.gradient + self.state

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "master": self.master,
            "narrow_copy": self.narrow_copy,
            "gradient": self.gradient,
            "state": self.state,
            "total": self.total,
        }


def bytes_per_parameter(
    optimizer: OptimizerKind = ADAM, *, mixed_precision: bool = True
) -> MemoryBreakdown:
    """The breakdown, derived rather than remembered.

    In mixed precision the master parameter stays wide and a narrow copy is made for the forward
    pass, so the parameter is paid for twice. The gradient arrives narrow because the backward
    pass produced it there. The optimiser state stays wide, because a moment accumulated in half
    precision stops moving for the same reason a long sum does.
    """
    if mixed_precision:
        return MemoryBreakdown(
            master=BYTES_PER_FLOAT32,
            narrow_copy=BYTES_PER_FLOAT16,
            gradient=BYTES_PER_FLOAT16,
            state=optimizer.states * BYTES_PER_FLOAT32,
        )
    return MemoryBreakdown(
        master=BYTES_PER_FLOAT32,
        narrow_copy=0,
        gradient=BYTES_PER_FLOAT32,
        state=optimizer.states * BYTES_PER_FLOAT32,
    )


def the_sixteen_bytes(optimizer: OptimizerKind = ADAM) -> dict:
    """Where the number everybody quotes comes from."""
    breakdown = bytes_per_parameter(optimizer)
    return {
        "master": breakdown.master,
        "narrow_copy": breakdown.narrow_copy,
        "gradient": breakdown.gradient,
        "moments": breakdown.state,
        "total": breakdown.total,
    }


def compare_optimizers(*, mixed_precision: bool = True) -> list[dict]:
    """Bytes per parameter for each optimiser.

    Plain descent costs eight and Adam costs sixteen, so switching optimisers doubles the
    memory of a training run before anything else is decided. That is a larger effect than most
    of what a compiler can do, and it belongs in the same table as everything else the compiler
    is trading against.
    """
    return [
        {
            "optimizer": kind.name,
            **bytes_per_parameter(kind, mixed_precision=mixed_precision).as_dict(),
        }
        for kind in OPTIMIZERS
    ]


def mixed_precision_saves_nothing_on_the_parameters() -> dict:
    """Whether narrowing the model reduces what training holds, which it does not.

    Exactly nothing, to the byte. A wide run holds four bytes of parameter and four of gradient;
    a mixed run holds four of master parameter, two of narrow copy and two of narrow gradient.
    The two bytes saved on the gradient are spent on the second copy of the parameter.

    So mixed precision pays for itself entirely in the activations and in the arithmetic rate.
    The parameter side is a wash, and any account of it that claims a saving there has left the
    master copy out.
    """
    wide = bytes_per_parameter(ADAM, mixed_precision=False)
    narrow = bytes_per_parameter(ADAM, mixed_precision=True)
    return {
        "wide": wide.total,
        "mixed": narrow.total,
        "identical": wide.total == narrow.total,
        "difference": narrow.total - wide.total,
    }


def training_memory(parameters: int, optimizer: OptimizerKind = ADAM) -> int:
    """Bytes a model needs before any activation is stored."""
    if parameters < 1:
        raise ConfigError(f"a model needs parameters, got {parameters}")
    return parameters * bytes_per_parameter(optimizer).total


def sharded_bytes_per_parameter(devices: int, optimizer: OptimizerKind = ADAM) -> float:
    """What each device holds when the optimiser state is split across them.

    The state and the master parameter divide, because each device only updates the slice it
    owns. The narrow copy and the gradient do not, because every device needs the whole model to
    run the forward pass and produces a gradient for all of it.
    """
    if devices < 1:
        raise ConfigError(f"there has to be at least one device, got {devices}")
    breakdown = bytes_per_parameter(optimizer)
    divided = (breakdown.master + breakdown.state) / devices
    return divided + breakdown.narrow_copy + breakdown.gradient


def sharding_sweep(
    counts: Sequence[int] = (1, 2, 4, 8, 16, 64), optimizer: OptimizerKind = ADAM
) -> list[dict]:
    """Bytes per parameter per device, against how many devices share the state.

    Falls fast and then flattens at four, which is the part that is not obvious. Twelve of the
    sixteen bytes divide and four do not, so the floor is four however many devices are added
    and most of the benefit has arrived by eight.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    return [
        {
            "devices": count,
            "bytes_per_parameter": round(sharded_bytes_per_parameter(count, optimizer), 3),
        }
        for count in counts
    ]


def sharding_has_a_floor(optimizer: OptimizerKind = ADAM) -> dict:
    """What is left after the state has been divided as far as it goes."""
    breakdown = bytes_per_parameter(optimizer)
    return {
        "at_one_device": breakdown.total,
        "at_sixty_four": round(sharded_bytes_per_parameter(64, optimizer), 3),
        "floor": breakdown.narrow_copy + breakdown.gradient,
    }


def update_is_memory_bound(
    parameters: int = 1_000_000,
    optimizer: OptimizerKind = ADAM,
    flops_per_second: float = 1.2e13,
    bytes_per_second: float = 9.0e11,
) -> dict:
    """Whether the optimiser step is limited by arithmetic or by traffic.

    By traffic, thirty five to one. The step reads the gradient, both moments and the master
    parameter and writes three of them back, which is twenty six bytes for ten operations, and
    no machine built this decade has that ratio.
    """
    if parameters < 1:
        raise ConfigError(f"a model needs parameters, got {parameters}")
    if min(flops_per_second, bytes_per_second) <= 0:
        raise ConfigError("a machine has to do some work per second")

    breakdown = bytes_per_parameter(optimizer)
    moved = parameters * (breakdown.master * 2 + breakdown.state * 2 + breakdown.gradient)
    work = parameters * optimizer.flops_per_parameter
    compute_seconds = work / flops_per_second
    traffic_seconds = moved / bytes_per_second
    return {
        "bytes_moved": moved,
        "arithmetic": work,
        "compute_seconds": compute_seconds,
        "traffic_seconds": traffic_seconds,
        "memory_bound": traffic_seconds > compute_seconds,
        "ratio": round(traffic_seconds / compute_seconds, 2) if compute_seconds else 0.0,
    }


def cheaper_arithmetic_would_not_help(parameters: int = 1_000_000) -> dict:
    """What halving the arithmetic in the update would buy.

    Nothing measurable. The step is bound by the bytes and the arithmetic is not close to the
    limit, so an optimiser with half the operations runs at the same speed. Anything that
    reduces the traffic, which means keeping the moments in fewer bits, is the only lever.
    """
    full = update_is_memory_bound(parameters)
    cheaper = OptimizerKind(name="cheaper", states=2, flops_per_parameter=5.0)
    halved = update_is_memory_bound(parameters, cheaper)
    return {
        "time_now": max(full["compute_seconds"], full["traffic_seconds"]),
        "time_with_half_the_arithmetic": max(
            halved["compute_seconds"], halved["traffic_seconds"]
        ),
        "unchanged": max(full["compute_seconds"], full["traffic_seconds"])
        == max(halved["compute_seconds"], halved["traffic_seconds"]),
    }


def activation_bytes(
    batch: int, sequence: int, width: int, layers: int, *, per_layer_tensors: int = 6
) -> int:
    """What the forward pass has to keep for the backward pass.

    Counted as a fixed number of tensors per layer, which is the honest level of detail: the
    exact count depends on which intermediates a backward pass needs and that depends on the
    rematerialisation policy, so a single number here with the assumption stated is better than
    a precise number that is precise about the wrong configuration.
    """
    if min(batch, sequence, width, layers, per_layer_tensors) < 1:
        raise ConfigError("every dimension has to be positive")
    return batch * sequence * width * layers * per_layer_tensors * BYTES_PER_FLOAT16


def where_activations_overtake(
    parameters: int = 100_000_000,
    batch: int = 32,
    sequence: int = 2048,
    width: int = 4096,
    limit: int = 512,
) -> int:
    """The depth at which the activations exceed everything else the training holds.

    One layer, at these sizes. A single layer of activations for a batch of thirty two at a
    sequence of two thousand and a width of four thousand is three gigabytes and the entire
    training state of a hundred million parameters is one and a half. Searched rather than
    solved because both sides move with the configuration, and the answer being one is the
    reason the memory conversation about large models is about batch size and sequence length
    rather than about weights.
    """
    weights = training_memory(parameters)
    for layers in range(1, limit + 1):
        if activation_bytes(batch, sequence, width, layers) > weights:
            return layers
    return 0


def memory_split(
    parameters: int = 100_000_000,
    batch: int = 32,
    sequence: int = 2048,
    width: int = 4096,
    layers: int = 48,
) -> dict:
    """The whole picture for one configuration, in bytes."""
    weights = training_memory(parameters)
    activations = activation_bytes(batch, sequence, width, layers)
    total = weights + activations
    return {
        "weights_and_state": weights,
        "activations": activations,
        "total": total,
        "activation_share": round(activations / total, 4) if total else 0.0,
    }


def batch_sweep(sizes: Sequence[int] = (1, 4, 16, 64, 256)) -> list[dict]:
    """How the split moves with the batch.

    Linearly on one side and not at all on the other, so the share held by activations goes from
    a fifth at a batch of one to almost everything at two hundred and fifty six. Every decision
    about training memory is downstream of that one line.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    return [{"batch": size, **memory_split(batch=size)} for size in sizes]


def adam_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    step: int,
    *,
    rate: float = 1e-3,
    beta_one: float = 0.9,
    beta_two: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One update, written out so the memory count above can be checked against real work.

    The bias correction is the part most write ups leave implicit. Both moments start at zero
    and are pulled toward the gradient by a factor less than one, so early steps are biased
    toward zero by a known amount and dividing it out is what makes the first step the size it
    should be rather than a thousandth of it.
    """
    if step < 1:
        raise ConfigError(f"the step count starts at one, got {step}")
    if not 0 <= beta_one < 1 or not 0 <= beta_two < 1:
        raise ConfigError("the decay rates have to be in [0, 1)")

    first = beta_one * first + (1 - beta_one) * gradient
    second = beta_two * second + (1 - beta_two) * gradient * gradient
    corrected_first = first / (1 - beta_one**step)
    corrected_second = second / (1 - beta_two**step)
    updated = parameter - rate * corrected_first / (corrected_second.sqrt() + epsilon)
    return updated, first, second


def matches_the_library(steps: int = 5, size: int = 64, *, seed: int = 0) -> dict:
    """The update above against the one in torch, over several steps.

    Several rather than one, because the bias correction only differs from the uncorrected
    version early and a single step would agree for the wrong reason on some of the terms.
    """
    if steps < 1:
        raise ConfigError(f"the step count must be positive, got {steps}")
    generator = torch.Generator().manual_seed(seed)
    start = torch.randn(size, generator=generator)
    gradients = [torch.randn(size, generator=generator) for _ in range(steps)]

    mine = start.clone()
    first = torch.zeros(size)
    second = torch.zeros(size)
    for index, gradient in enumerate(gradients, start=1):
        mine, first, second = adam_step(mine, gradient, first, second, index)

    theirs = start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theirs], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    for gradient in gradients:
        optimizer.zero_grad()
        theirs.grad = gradient.clone()
        optimizer.step()

    gap = float((mine - theirs.detach()).abs().max())
    scale = float(theirs.detach().abs().max())
    return {"largest_gap": gap, "relative_gap": gap / scale if scale else gap}


def bias_correction_matters_for_thousands_of_steps(
    steps: Sequence[int] = (1, 5, 20, 100, 500, 1000, 3000, 7000), size: int = 32
) -> list[dict]:
    """How far the corrected and uncorrected updates are apart, at several step counts.

    Not a short warmup, which is what the name bias correction suggests. The ratio starts at
    about a third, falls to a sixth by the twentieth step, and only comes back to one after a
    few thousand. The reason is that the two moments have different decay rates, so their
    corrections shrink at different speeds, and the second moment's decay of a thousandth sets
    the timescale for the whole thing.
    """
    if not steps:
        raise ConfigError("there is nothing to sweep")
    generator = torch.Generator().manual_seed(1)
    gradient = torch.randn(size, generator=generator).abs() + 0.5

    rows = []
    for step in steps:
        if step < 1:
            raise ConfigError(f"the step count starts at one, got {step}")
        first = (1 - 0.9**step) * gradient
        second = (1 - 0.999**step) * gradient * gradient
        corrected = (first / (1 - 0.9**step)) / ((second / (1 - 0.999**step)).sqrt() + 1e-8)
        uncorrected = first / (second.sqrt() + 1e-8)
        rows.append({"step": step, "ratio": round(float((corrected / uncorrected).mean()), 4)})
    return rows


def the_correction_is_not_monotonic() -> dict:
    """Where in that sweep the two updates are furthest apart.

    Not at the first step, which is where the name suggests it would be. The gap widens for the
    first twenty or so steps before it starts closing, because the first moment's correction
    fades ten times faster than the second's and their ratio gets worse before it gets better.
    """
    rows = {
        row["step"]: row["ratio"] for row in bias_correction_matters_for_thousands_of_steps()
    }
    return {
        "at_step_one": rows[1],
        "at_step_twenty": rows[20],
        "at_step_seven_thousand": rows[7000],
        "worst_is_not_the_first_step": rows[20] < rows[1],
        "converges_eventually": abs(rows[7000] - 1.0) < 0.01,
    }


def narrow_moments_would_save(optimizer: OptimizerKind = ADAM) -> dict:
    """What keeping the optimiser moments in half precision would be worth.

    A quarter of the total, which is the largest single saving available on the parameter side
    and the reason it keeps being tried. It is also the thing the accumulator argument warns
    against: a moment is a running average and a running average in half precision stops moving
    once the total is large relative to the increment.
    """
    wide = bytes_per_parameter(optimizer)
    narrow_state = optimizer.states * BYTES_PER_FLOAT16
    narrow_total = wide.master + wide.narrow_copy + wide.gradient + narrow_state
    return {
        "now": wide.total,
        "with_narrow_moments": narrow_total,
        "saving": round(wide.total / narrow_total, 4) if narrow_total else 0.0,
    }


def a_narrow_moment_stalls(steps: int = 4096, beta: float = 0.999) -> dict:
    """The failure that argument is about, produced directly.

    A running average with a decay of a thousandth adds a thousandth of the gradient per step.
    In half precision that increment falls below the last bit of the average long before the
    average has converged, so the moment freezes at a value it reached early and the update
    stops adapting.
    """
    if steps < 1:
        raise ConfigError(f"the step count must be positive, got {steps}")
    gradient_square = torch.tensor(1.0)
    wide = torch.zeros(())
    narrow = torch.zeros((), dtype=torch.float16)
    frozen_at = 0
    for step in range(1, steps + 1):
        wide = beta * wide + (1 - beta) * gradient_square
        before = float(narrow)
        narrow = torch.tensor(beta, dtype=torch.float16) * narrow + torch.tensor(
            1 - beta, dtype=torch.float16
        ) * gradient_square.to(torch.float16)
        if float(narrow) == before and not frozen_at and step > 1:
            frozen_at = step
    return {
        "wide": round(float(wide), 6),
        "narrow": round(float(narrow), 6),
        "froze_at": frozen_at,
        "gap": round(abs(float(wide) - float(narrow)), 6),
    }
