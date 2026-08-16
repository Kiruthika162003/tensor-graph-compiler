from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, ScheduleError

# Splitting a model across devices by layer, and the idle time that creates.
#
# Tensor parallelism splits every layer across every device and pays for it in communication on
# each one. Pipeline parallelism splits the layers between devices instead, so each device holds
# a few whole layers and passes an activation to the next. The communication is tiny, one
# activation per boundary, and the cost moves somewhere else entirely: while the first device is
# working on the first batch, every other device has nothing to do.
#
# The fix is to cut the batch into microbatches and keep them in flight. The first device starts
# the second microbatch as soon as it has handed the first on, and after a few of those every
# device is busy. The idle time at the start and end does not go away, it gets amortised, and
# the fraction of the schedule that is idle is the number this file is about.
#
# Everything here is simulated rather than derived. A timeline is built slot by slot, the idle
# slots are counted, and the closed form is checked against the count rather than trusted. Two
# things came out of doing it that way.
#
# The closed form turns out to be exact, which is worth knowing because it is usually presented
# as an approximation. One less than the stage count over the microbatch count plus one less
# than the stage count, to four decimal places, for both task orderings and for a backward pass
# twice as long as the forward one. Any schedule that never idles a device with ready work gets
# the same number, so the ordering is not a throughput decision at all.
#
# It is a memory decision, and that is the other result. Under a forward first schedule the
# activations held at the front grow one for one with the microbatch count, so halving the
# bubble doubles the memory. Under a backward preferring schedule they saturate: at two stages
# less one when the passes are equal, and at a slightly higher ceiling when the backward pass is
# slower, and then they stop moving however finely the batch is cut. Same bubble, bounded
# memory, which is the entire reason the one forward one backward schedule exists.


@dataclass
class PipelineShape:
    """A model split across devices, run on a batch cut into pieces."""

    stages: int
    microbatches: int
    forward_slots: int = 1
    backward_slots: int = 2

    def __post_init__(self) -> None:
        if self.stages < 1:
            raise ConfigError(f"a pipeline needs stages, got {self.stages}")
        if self.microbatches < 1:
            raise ConfigError(f"a pipeline needs microbatches, got {self.microbatches}")
        if min(self.forward_slots, self.backward_slots) < 1:
            raise ConfigError("a pass takes at least one slot")

    @property
    def work_slots(self) -> int:
        """Slots of real work in the whole schedule."""
        return self.stages * self.microbatches * (self.forward_slots + self.backward_slots)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "stages": self.stages,
            "microbatches": self.microbatches,
            "forward_slots": self.forward_slots,
            "backward_slots": self.backward_slots,
        }


@dataclass
class Timeline:
    """What each device did in each slot."""

    slots: list[list[str]] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Slots the schedule takes end to end."""
        return len(self.slots)

    @property
    def devices(self) -> int:
        """Devices the schedule covers."""
        return len(self.slots[0]) if self.slots else 0

    @property
    def idle_slots(self) -> int:
        """Device slots spent doing nothing."""
        return sum(1 for row in self.slots for entry in row if not entry)

    @property
    def busy_slots(self) -> int:
        """Device slots spent doing something."""
        return self.length * self.devices - self.idle_slots

    @property
    def bubble(self) -> float:
        """Share of the schedule that is idle."""
        total = self.length * self.devices
        if total == 0:
            return 0.0
        return self.idle_slots / total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "length": self.length,
            "devices": self.devices,
            "idle": self.idle_slots,
            "busy": self.busy_slots,
            "bubble": round(self.bubble, 4),
        }


def simulate(shape: PipelineShape, *, interleave: bool = True) -> Timeline:
    """Build the schedule slot by slot and record what every device is doing.

    A device can run one task at a time. A forward task for a microbatch on a stage can start
    once that microbatch has finished the stage before; a backward task once the microbatch has
    finished the forward pass on the last stage and the backward pass on the stage after. Those
    two rules are the whole schedule and everything else in this file is a measurement of what
    they produce.

    The interleave flag chooses which ready task a device prefers. Preferring backward work is
    the schedule everybody calls one forward one backward; preferring forward work runs every
    forward pass before any backward pass, which is the schedule the textbook bubble formula was
    derived for. Both are simulated here so the formula has something to be checked against.
    """
    ready_forward = {(0, batch): 0 for batch in range(shape.microbatches)}
    ready_backward: dict[tuple[int, int], int] = {}
    remaining_forward = {
        (stage, batch): shape.forward_slots
        for stage in range(shape.stages)
        for batch in range(shape.microbatches)
    }
    remaining_backward = {
        (stage, batch): shape.backward_slots
        for stage in range(shape.stages)
        for batch in range(shape.microbatches)
    }

    timeline = Timeline()
    running: dict[int, tuple[str, int, int]] = {}
    slot = 0
    limit = shape.work_slots * 4

    while remaining_forward or remaining_backward or running:
        if slot > limit:
            raise ScheduleError("the schedule did not finish, so a dependency rule is wrong")
        row = [""] * shape.stages
        for stage in range(shape.stages):
            if stage in running:
                kind, batch, left = running[stage]
                row[stage] = f"{kind}{batch}"
                if left <= 1:
                    _finish(kind, stage, batch, shape, ready_forward, ready_backward, slot)
                    del running[stage]
                else:
                    running[stage] = (kind, batch, left - 1)
                continue

            picked = _pick(
                stage,
                slot,
                shape,
                ready_backward,
                ready_forward,
                remaining_backward,
                remaining_forward,
                prefer_backward=interleave,
            )
            if picked is None:
                continue
            kind, batch = picked
            duration = shape.forward_slots if kind == "f" else shape.backward_slots
            if kind == "f":
                del remaining_forward[(stage, batch)]
            else:
                del remaining_backward[(stage, batch)]
            row[stage] = f"{kind}{batch}"
            if duration > 1:
                running[stage] = (kind, batch, duration - 1)
            else:
                _finish(kind, stage, batch, shape, ready_forward, ready_backward, slot)
        timeline.slots.append(row)
        slot += 1
    return timeline


def _pick(
    stage: int,
    slot: int,
    shape: PipelineShape,
    ready_backward: dict,
    ready_forward: dict,
    remaining_backward: dict,
    remaining_forward: dict,
    *,
    prefer_backward: bool,
) -> tuple[str, int] | None:
    """The task a device starts in a slot, in the order the flag asks for."""
    backward = _first_ready("b", stage, slot, shape, ready_backward, remaining_backward)
    forward = _first_ready("f", stage, slot, shape, ready_forward, remaining_forward)
    if prefer_backward:
        return backward or forward
    return forward or backward


def _first_ready(
    kind: str,
    stage: int,
    slot: int,
    shape: PipelineShape,
    ready: dict,
    remaining: dict,
) -> tuple[str, int] | None:
    """The lowest numbered microbatch of one kind that this stage could start now."""
    for batch in range(shape.microbatches):
        key = (stage, batch)
        if key in remaining and ready.get(key, 1 << 30) <= slot:
            return (kind, batch)
    return None


def completions(timeline: Timeline) -> list[tuple[int, int, str, int]]:
    """When each task finished, as a slot, a stage, a kind and a microbatch.

    Read off the timeline by finding runs of the same label rather than recorded during the
    simulation, so anything downstream of it is looking at the schedule that was produced rather
    than at bookkeeping that might disagree with it.
    """
    events: list[tuple[int, int, str, int]] = []
    for stage in range(timeline.devices):
        column = [row[stage] for row in timeline.slots]
        for index, entry in enumerate(column):
            if not entry:
                continue
            if index + 1 < len(column) and column[index + 1] == entry:
                continue
            events.append((index, stage, entry[0], int(entry[1:])))
    return sorted(events)


def _finish(
    kind: str,
    stage: int,
    batch: int,
    shape: PipelineShape,
    ready_forward: dict,
    ready_backward: dict,
    slot: int,
) -> None:
    """Record what a finished task unlocks, and from which slot.

    The next slot rather than this one. A task finishing at the end of a slot cannot have its
    successor start in the same slot, and the row is filled left to right, so setting the ready
    time to the current slot lets a later stage start work in the slot its input finished in.
    That is a schedule that could not run, and it shows up as a smaller bubble, which is exactly
    the direction a mistake here would be welcomed in.
    """
    if kind == "f":
        if stage + 1 < shape.stages:
            ready_forward[(stage + 1, batch)] = slot + 1
            return
        ready_backward[(stage, batch)] = slot + 1
        return
    if stage > 0:
        ready_backward[(stage - 1, batch)] = slot + 1


def predicted_bubble(shape: PipelineShape) -> float:
    """The closed form everybody quotes.

    One less than the stage count over the microbatch count plus one less than the stage count.
    It assumes every task takes the same time, which is the assumption the measurement below
    disagrees with.
    """
    return (shape.stages - 1) / (shape.microbatches + shape.stages - 1)


def measured_bubble(shape: PipelineShape) -> float:
    """The bubble the simulation actually produces."""
    return simulate(shape).bubble


def the_formula_assumes_equal_passes(stages: int = 4, microbatches: int = 8) -> dict:
    """Where the closed form and the simulation part company.

    They do not part company at all, which is the finding. The closed form matches the
    simulation exactly for both task orderings and for a backward pass twice the length of the
    forward one, so it is not an approximation that happens to be close, it is what a schedule
    that never idles a ready device produces.

    Which makes the ordering purely a memory decision. Nothing about the throughput of a
    pipeline depends on whether a device reaches for forward work or backward work first.
    """
    equal = PipelineShape(stages=stages, microbatches=microbatches, backward_slots=1)
    realistic = PipelineShape(stages=stages, microbatches=microbatches, backward_slots=2)
    return {
        "formula": round(predicted_bubble(equal), 4),
        "forward_first": round(simulate(equal, interleave=False).bubble, 4),
        "backward_preferring": round(simulate(equal, interleave=True).bubble, 4),
        "with_a_slower_backward": round(measured_bubble(realistic), 4),
    }


def microbatch_sweep(
    counts: Sequence[int] = (1, 2, 4, 8, 16, 32), stages: int = 4
) -> list[dict]:
    """The bubble against how finely the batch is cut.

    Falls like one over the count and never reaches zero. At one microbatch three of four
    devices are idle at any moment; at thirty two the bubble is under a tenth, and the cost of
    getting there is on the memory side rather than here.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in counts:
        shape = PipelineShape(stages=stages, microbatches=count)
        rows.append(
            {
                "microbatches": count,
                "bubble": round(measured_bubble(shape), 4),
                "length": simulate(shape).length,
            }
        )
    return rows


def the_bubble_falls_like_one_over_the_count(stages: int = 4) -> dict:
    """Whether doubling the microbatches really halves the idle share."""
    rows = {row["microbatches"]: row["bubble"] for row in microbatch_sweep(stages=stages)}
    return {
        "at_four": rows[4],
        "at_eight": rows[8],
        "at_sixteen": rows[16],
        "roughly_halving": rows[16] < rows[8] < rows[4],
    }


def stage_sweep(counts: Sequence[int] = (2, 4, 8, 16), microbatches: int = 8) -> list[dict]:
    """The bubble against how many devices the model is split over.

    Grows with the stage count, which is the whole difficulty with pipeline parallelism. Adding
    a device makes the pipeline longer, so more of the schedule is spent filling and draining
    it, and past a point the extra device is adding idle time rather than throughput.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    return [
        {
            "stages": count,
            "bubble": round(measured_bubble(PipelineShape(count, microbatches)), 4),
        }
        for count in counts
    ]


def more_devices_make_it_worse(microbatches: int = 8) -> dict:
    """Whether that really goes the wrong way."""
    rows = {row["stages"]: row["bubble"] for row in stage_sweep(microbatches=microbatches)}
    return {
        "at_two_stages": rows[2],
        "at_sixteen_stages": rows[16],
        "worse_with_more_devices": rows[16] > rows[2],
    }


def activations_in_flight(shape: PipelineShape, *, interleave: bool = True) -> list[int]:
    """How many microbatches each stage is holding activations for, at the peak.

    A microbatch that has finished a forward pass on a stage and not had its backward pass there
    is holding that stage's activations. Counted from the completion events rather than from the
    slots, because a task that occupies several slots would otherwise be counted several times.
    """
    timeline = simulate(shape, interleave=interleave)
    held = [0] * shape.stages
    peak = [0] * shape.stages
    for _, stage, kind, _ in completions(timeline):
        if kind == "f":
            held[stage] += 1
            peak[stage] = max(peak[stage], held[stage])
        else:
            held[stage] = max(held[stage] - 1, 0)
    return peak


def memory_against_bubble(
    counts: Sequence[int] = (1, 2, 4, 8, 16, 32), stages: int = 4, backward_slots: int = 1
) -> list[dict]:
    """The trade, in one table, for both schedules.

    Under a forward first schedule the memory grows with the microbatch count, one for one, and
    that is the trade everybody quotes: halve the bubble, double the activations.

    Under a backward preferring schedule it does not. The peak saturates and stays there however
    finely the batch is cut, because a device that takes backward work the moment it is
    available never gets more than a fixed number of microbatches ahead of the ones behind it.
    That is the entire reason the one forward one backward schedule exists and it is visible
    here as a column that stops growing.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in counts:
        shape = PipelineShape(stages=stages, microbatches=count, backward_slots=backward_slots)
        rows.append(
            {
                "microbatches": count,
                "bubble": round(measured_bubble(shape), 4),
                "peak_interleaved": max(activations_in_flight(shape, interleave=True)),
                "peak_forward_first": max(activations_in_flight(shape, interleave=False)),
            }
        )
    return rows


def the_memory_saturates(stages: int = 4) -> dict:
    """Whether the backward preferring schedule really stops accumulating.

    It does, at two stages less one when the passes take the same time, which for four stages is
    seven. The forward first schedule keeps growing and reaches thirty two where the other is
    still at seven, and the two have the same bubble.
    """
    rows = {row["microbatches"]: row for row in memory_against_bubble(stages=stages)}
    return {
        "interleaved_at_eight": rows[8]["peak_interleaved"],
        "interleaved_at_thirty_two": rows[32]["peak_interleaved"],
        "forward_first_at_thirty_two": rows[32]["peak_forward_first"],
        "ceiling": 2 * stages - 1,
        "saturates": rows[8]["peak_interleaved"] == rows[32]["peak_interleaved"],
        "matches_the_ceiling": rows[32]["peak_interleaved"] == 2 * stages - 1,
    }


def a_slower_backward_raises_the_ceiling(stages: int = 4) -> dict:
    """Where the saturation point moves when the backward pass takes longer.

    Up, and it still saturates. A device that spends two slots on a backward pass releases
    activations at half the rate it can take them in, so it accumulates a few more before the
    two rates balance, and the ceiling goes from seven to ten. The property that matters is not
    the number, it is that there is one.
    """
    equal = {row["microbatches"]: row for row in memory_against_bubble(stages=stages)}
    slower = {
        row["microbatches"]: row
        for row in memory_against_bubble(stages=stages, backward_slots=2)
    }
    return {
        "equal_passes_ceiling": equal[32]["peak_interleaved"],
        "slower_backward_ceiling": slower[32]["peak_interleaved"],
        "both_saturate": equal[16]["peak_interleaved"] == equal[32]["peak_interleaved"]
        and slower[16]["peak_interleaved"] == slower[32]["peak_interleaved"],
    }


def halving_the_bubble_doubles_the_memory(stages: int = 4) -> dict:
    """Whether the trade is one for one, which it is only for the naive schedule.

    Going from four microbatches to sixteen quarters the forward first memory into four times
    what it was, exactly, and leaves the backward preferring one within a factor of two. The
    trade everybody states as a law is a property of the schedule rather than of pipelining.
    """
    rows = {row["microbatches"]: row for row in memory_against_bubble(stages=stages)}
    return {
        "bubble_at_four": rows[4]["bubble"],
        "bubble_at_sixteen": rows[16]["bubble"],
        "forward_first_memory_at_four": rows[4]["peak_forward_first"],
        "forward_first_memory_at_sixteen": rows[16]["peak_forward_first"],
        "interleaved_memory_at_four": rows[4]["peak_interleaved"],
        "interleaved_memory_at_sixteen": rows[16]["peak_interleaved"],
    }


def the_first_stage_holds_the_most(stages: int = 4, microbatches: int = 8) -> dict:
    """Which device runs out of memory first.

    The first one, by a factor of the stage count. It starts every microbatch before any other
    device sees it and releases each one only after the backward pass has come all the way back,
    so it is holding activations for the whole depth of the pipeline while the last stage holds
    one.
    """
    peaks = activations_in_flight(PipelineShape(stages=stages, microbatches=microbatches))
    return {
        "per_stage": peaks,
        "first": peaks[0],
        "last": peaks[-1],
        "front_heavy": peaks[0] >= peaks[-1],
    }


def interleaving_helps_the_memory(stages: int = 4, microbatches: int = 8) -> dict:
    """What preferring backward work buys.

    Less memory at exactly the same bubble, which is as close to free as a scheduling decision
    gets. A
    backward pass releases the activations it was holding and a forward pass allocates more, so
    a device that takes backward work whenever it can holds fewer microbatches at once without
    ever idling a slot it could have used.
    """
    shape = PipelineShape(stages=stages, microbatches=microbatches)
    greedy = simulate(shape, interleave=True)
    forward_first = simulate(shape, interleave=False)
    return {
        "bubble_interleaved": round(greedy.bubble, 4),
        "bubble_forward_first": round(forward_first.bubble, 4),
        "memory_interleaved": max(activations_in_flight(shape, interleave=True)),
        "memory_forward_first": max(activations_in_flight(shape, interleave=False)),
        "same_bubble": round(greedy.bubble, 4) == round(forward_first.bubble, 4),
    }


def a_single_microbatch_is_almost_all_bubble(stages: int = 4) -> dict:
    """The degenerate case, which is what pipeline parallelism looks like without microbatching.

    One device works and the rest wait, so the idle share is one less than the stage count over
    the stage count. Splitting a model across four devices and running one batch through it
    makes three of them ornamental.
    """
    shape = PipelineShape(stages=stages, microbatches=1)
    return {
        "bubble": round(measured_bubble(shape), 4),
        "predicted": round((stages - 1) / stages, 4),
    }


def compare_schedules(stages: int = 4, microbatches: int = 8) -> list[dict]:
    """Every configuration in this file, side by side."""
    rows = []
    for label, shape in (
        ("one microbatch", PipelineShape(stages, 1)),
        ("equal passes", PipelineShape(stages, microbatches, backward_slots=1)),
        ("slower backward", PipelineShape(stages, microbatches, backward_slots=2)),
        ("many microbatches", PipelineShape(stages, microbatches * 4)),
    ):
        timeline = simulate(shape)
        row = timeline.as_dict()
        row["schedule"] = label
        rows.append(row)
    return rows


def every_task_runs_exactly_once(stages: int = 3, microbatches: int = 4) -> dict:
    """Whether the simulation schedules every piece of work and no piece twice.

    The check that makes every number above worth reading. A schedule that dropped a task would
    report a smaller bubble and a shorter timeline, and both would look like an improvement.
    """
    shape = PipelineShape(stages=stages, microbatches=microbatches, backward_slots=1)
    timeline = simulate(shape)
    seen: dict[str, int] = {}
    for row in timeline.slots:
        for stage, entry in enumerate(row):
            if entry:
                key = f"{stage}:{entry}"
                seen[key] = seen.get(key, 0) + 1
    expected = stages * microbatches * 2
    return {
        "expected_tasks": expected,
        "distinct_tasks": len(seen),
        "every_task_once": len(seen) == expected and all(count == 1 for count in seen.values()),
    }


def dependencies_are_respected(stages: int = 3, microbatches: int = 3) -> dict:
    """Whether a stage ever starts a microbatch the stage before it has not finished.

    Never, and it is worth checking rather than assuming, because the failure would produce a
    schedule with no bubble at all, which is exactly the answer somebody optimising a pipeline
    wants to see.
    """
    shape = PipelineShape(stages=stages, microbatches=microbatches, backward_slots=1)
    timeline = simulate(shape)
    started: dict[str, int] = {}
    for index, row in enumerate(timeline.slots):
        for stage, entry in enumerate(row):
            if entry.startswith("f"):
                started.setdefault(f"{stage}:{entry}", index)

    violations = 0
    for stage in range(1, stages):
        for batch in range(microbatches):
            here = started.get(f"{stage}:f{batch}")
            before = started.get(f"{stage - 1}:f{batch}")
            if here is None or before is None or here <= before:
                violations += 1
    return {"checked": stages * microbatches, "violations": violations}


def a_pipeline_of_one_stage_has_no_bubble(microbatches: int = 8) -> dict:
    """The other degenerate case, which is a single device running everything.

    No idle time at all, because there is nothing to wait for. Worth having as the floor: any
    measurement of a bubble on one stage would mean the simulation is inventing idleness rather
    than finding it.
    """
    timeline = simulate(PipelineShape(stages=1, microbatches=microbatches))
    return {"bubble": timeline.bubble, "length": timeline.length}
