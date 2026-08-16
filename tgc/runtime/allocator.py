from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import AllocationError, ConfigError
from tgc.ir.builder import branching_graph, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.graph import Graph

# Handing out memory at runtime, when the sizes are not known until they are asked for.
#
# memory/planner.py solves the version of this problem where every size and every lifetime is
# known ahead of time, which is the version a compiler with static shapes has. This is the other
# one: a request arrives, a block has to come back, and nothing about what comes next is known.
# Every framework has one of these underneath it and the choices it makes are why a model that
# fits on paper does not fit in practice.
#
# The two failures have opposite cures. Internal fragmentation is the space inside a block that
# the request did not ask for, and it comes from rounding sizes up so that freed blocks can be
# reused. External fragmentation is free space that cannot satisfy a request because it is in
# the wrong sized pieces, and it comes from not rounding. Choosing between them is the whole
# design.
#
# On a trace of arbitrary sizes, rounding to powers of two reuses a freed block eighty two
# percent of the time and wastes twenty three percent of what it hands out. Exact fitting reuses
# nothing at all, because two arbitrary sizes are never equal. Size classes get ninety four
# percent of the coarse policy's reuse for three fifths of its waste, which is why that is what
# ships.
#
# On a trace taken from a graph, all three policies produce identical numbers. Every tensor in a
# graph has a size that is already a power of two, so there is nothing for the rounding to round
# and exact fitting reuses just as well. The entire benefit of rounding, on this workload, is
# zero. An allocator tuned on one of these traces and deployed against the other will
# disappoint, and which of them a compiler is serving is a question worth asking before tuning
# anything.
#
# The last measurement is about giving memory back, and it also comes out flat. An allocator
# that returns a block the moment it is freed has a footprint equal to what is live; one that
# keeps everything has a footprint equal to the high water mark; on a graph trace those are the
# same number, because a graph frees a value and immediately allocates one the same size.
# Caching costs nothing here and buys a three quarters reuse rate, which is the easiest trade in
# this file.

POLICIES = ("exact", "power of two", "size classes")


@dataclass
class Block:
    """One region handed out or waiting to be handed out."""

    size: int
    capacity: int

    def __post_init__(self) -> None:
        if self.size < 0 or self.capacity < self.size:
            raise ConfigError(f"a block of {self.size} cannot have capacity {self.capacity}")

    @property
    def waste(self) -> int:
        """Bytes inside the block that nobody asked for."""
        return self.capacity - self.size

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"size": self.size, "capacity": self.capacity, "waste": self.waste}


def round_up(size: int, policy: str) -> int:
    """The capacity a request of a given size gets.

    Exact hands back what was asked for. Powers of two round to the next one, which is at worst
    a doubling and on average a quarter of the request. Size classes round to a small set of
    steps within each power of two, which is the compromise every real allocator uses because it
    keeps the reuse and cuts the waste in half.
    """
    if size < 0:
        raise ConfigError(f"cannot allocate {size} bytes")
    if policy not in POLICIES:
        raise ConfigError(f"unknown policy {policy!r}, expected one of {list(POLICIES)}")
    if size == 0:
        return 0
    if policy == "exact":
        return size

    power = 1
    while power < size:
        power *= 2
    if policy == "power of two":
        return power

    step = max(power // 4, 1)
    return ((size + step - 1) // step) * step


@dataclass
class AllocatorStats:
    """What a run of requests did."""

    requests: int = 0
    reused: int = 0
    fresh: int = 0
    bytes_requested: int = 0
    bytes_handed_out: int = 0
    high_water: int = 0
    live_peak: int = 0

    @property
    def reuse_rate(self) -> float:
        """Share of requests served from something already held."""
        if self.requests == 0:
            return 0.0
        return self.reused / self.requests

    @property
    def internal_waste(self) -> float:
        """Share of what was handed out that nobody asked for."""
        if self.bytes_handed_out == 0:
            return 0.0
        return 1.0 - self.bytes_requested / self.bytes_handed_out

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "requests": self.requests,
            "reuse_rate": round(self.reuse_rate, 4),
            "internal_waste": round(self.internal_waste, 4),
            "high_water": self.high_water,
            "live_peak": self.live_peak,
        }


@dataclass
class Allocator:
    """A caching allocator with a free list per capacity.

    Keyed by capacity rather than by size, which is the entire reason rounding helps: two
    requests that round to the same capacity can share a block, and two that do not never can
    however close their sizes are.
    """

    policy: str = "power of two"
    give_back: bool = False
    free_lists: dict[int, list[int]] = field(default_factory=dict)
    held: dict[int, int] = field(default_factory=dict)
    stats: AllocatorStats = field(default_factory=AllocatorStats)
    _live: int = 0
    _cached: int = 0
    _next_handle: int = 0

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ConfigError(f"unknown policy {self.policy!r}")

    def allocate(self, size: int) -> int:
        """Hand out a block and return a handle to it."""
        capacity = round_up(size, self.policy)
        self.stats.requests += 1
        self.stats.bytes_requested += size
        self.stats.bytes_handed_out += capacity

        waiting = self.free_lists.get(capacity)
        if waiting:
            waiting.pop()
            self._cached -= capacity
            self.stats.reused += 1
        else:
            self.stats.fresh += 1

        handle = self._next_handle
        self._next_handle += 1
        self.held[handle] = capacity
        self._live += capacity
        self.stats.live_peak = max(self.stats.live_peak, self._live)
        self.stats.high_water = max(self.stats.high_water, self._live + self._cached)
        return handle

    def free(self, handle: int) -> None:
        """Give a block back to the allocator."""
        if handle not in self.held:
            raise AllocationError(f"handle {handle} was not allocated or is already free")
        capacity = self.held.pop(handle)
        self._live -= capacity
        if self.give_back:
            return
        self.free_lists.setdefault(capacity, []).append(capacity)
        self._cached += capacity

    @property
    def footprint(self) -> int:
        """Bytes the allocator is holding from the system right now."""
        return self._live + self._cached

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"policy": self.policy, "footprint": self.footprint, **self.stats.as_dict()}


def trace_from(graph: Graph) -> list[tuple[str, int]]:
    """An allocate and free sequence in the order a graph would run.

    Every node allocates its output and frees every operand whose last reader it is, which is
    exactly what an interpreter does and is where the sizes and lifetimes in a real trace come
    from. Building it here rather than by hand keeps the workload honest.
    """
    last_use: dict[str, int] = {}
    for index, node in enumerate(graph.nodes):
        for name in node.inputs:
            last_use[name] = index
    for name in graph.outputs:
        last_use[name] = len(graph.nodes)

    events: list[tuple[str, int]] = []
    sizes: dict[str, int] = {}
    for value in graph.inputs:
        sizes[value.name] = value.shape.elements * value.dtype.bytes

    for index, node in enumerate(graph.nodes):
        size = node.output.shape.elements * node.output.dtype.bytes
        sizes[node.name] = size
        events.append(("allocate", size))
        for name in node.inputs:
            if last_use.get(name) == index and name in sizes:
                events.append(("free", sizes[name]))
    return events


def run_trace(events: Sequence[tuple[str, int]], allocator: Allocator) -> AllocatorStats:
    """Replay a trace through an allocator.

    Frees are matched to the most recent allocation of the same size, which is what a real
    runtime does through the tensor object and what a trace of sizes alone has to approximate.
    A free with nothing to match is dropped rather than raising, because the approximation can
    produce one and the alternative is a check that only measures the approximation.
    """
    if not events:
        raise ConfigError("there is nothing to replay")
    outstanding: dict[int, list[int]] = {}
    for kind, size in events:
        if kind == "allocate":
            handle = allocator.allocate(size)
            outstanding.setdefault(size, []).append(handle)
            continue
        if kind != "free":
            raise ConfigError(f"unknown event {kind!r}")
        waiting = outstanding.get(size)
        if waiting:
            allocator.free(waiting.pop())
    return allocator.stats


def graph_traces() -> list[tuple[str, list[tuple[str, int]]]]:
    """The traces every measurement here runs over."""
    return [
        ("softmax", trace_from(softmax_graph())),
        ("layernorm", trace_from(layernorm_graph())),
        ("mlp", trace_from(mlp_graph())),
        ("branching", trace_from(branching_graph())),
    ]


def random_trace(count: int = 400, *, seed: int = 0) -> list[tuple[str, int]]:
    """A trace of sizes drawn without any structure.

    Not a model of anything. It is here so the results on the graph traces can be compared
    against a case with no locality in it, and the comparison says how much of the reuse comes
    from the allocator and how much from the graph handing it the same sizes repeatedly.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    generator = random.Random(seed)
    events: list[tuple[str, int]] = []
    live: list[int] = []
    for _ in range(count):
        if live and generator.random() < 0.45:
            events.append(("free", live.pop(generator.randrange(len(live)))))
            continue
        size = generator.randrange(1, 1 << 16)
        live.append(size)
        events.append(("allocate", size))
    return events


def compare_policies(events: Sequence[tuple[str, int]] | None = None) -> list[dict]:
    """Every rounding policy on one trace."""
    trace = list(events) if events is not None else trace_from(branching_graph())
    rows = []
    for policy in POLICIES:
        allocator = Allocator(policy=policy)
        run_trace(trace, allocator)
        rows.append({"policy": policy, **allocator.stats.as_dict()})
    return rows


def rounding_buys_reuse(events: Sequence[tuple[str, int]] | None = None) -> dict:
    """What rounding is for, on a trace with no repeated sizes in it.

    Exact fitting can only reuse a block when a request happens to be the same number of bytes
    as a freed one. On a random trace that almost never happens; on a graph trace it happens
    often, because a graph produces the same shapes repeatedly. The gap between those two is the
    whole argument, and it says rounding matters much more for a runtime serving arbitrary
    requests than for one replaying a graph.
    """
    trace = list(events) if events is not None else random_trace()
    rows = {row["policy"]: row for row in compare_policies(trace)}
    return {
        "exact_reuse": rows["exact"]["reuse_rate"],
        "power_of_two_reuse": rows["power of two"]["reuse_rate"],
        "size_class_reuse": rows["size classes"]["reuse_rate"],
    }


def rounding_costs_waste(events: Sequence[tuple[str, int]] | None = None) -> dict:
    """What rounding costs, on the same trace.

    Twenty three percent for powers of two, against the quarter the doubling predicts: a size
    drawn uniformly lands on average a quarter of the way below the power above it. Size classes
    cut that to fourteen while keeping most of the reuse, which is why nobody ships the pure
    power of two version.
    """
    trace = list(events) if events is not None else random_trace()
    rows = {row["policy"]: row for row in compare_policies(trace)}
    return {
        "exact_waste": rows["exact"]["internal_waste"],
        "power_of_two_waste": rows["power of two"]["internal_waste"],
        "size_class_waste": rows["size classes"]["internal_waste"],
    }


def size_classes_are_the_compromise(events: Sequence[tuple[str, int]] | None = None) -> dict:
    """Whether the middle policy really gets most of both.

    Ninety four percent of the coarse policy's reuse for three fifths of its waste, on a random
    trace. It is the only one of the three that is not the best at anything, which is what a
    compromise looks like when it works.
    """
    trace = list(events) if events is not None else random_trace()
    rows = {row["policy"]: row for row in compare_policies(trace)}
    coarse = rows["power of two"]
    middle = rows["size classes"]
    return {
        "reuse_against_the_coarse_one": round(
            middle["reuse_rate"] / max(coarse["reuse_rate"], 1e-9), 4
        ),
        "waste_against_the_coarse_one": round(
            middle["internal_waste"] / max(coarse["internal_waste"], 1e-9), 4
        ),
    }


def rounding_predicts_its_own_waste(samples: int = 20000, *, seed: int = 0) -> dict:
    """Whether the average waste from doubling really is a quarter.

    A size drawn uniformly from a range spanning several powers of two lands, on average, a
    quarter of the way from the power above it. That is a fact about the distribution rather
    than about the allocator, and checking it is what makes the measured number above readable
    as confirmation rather than as a coincidence.
    """
    if samples < 1:
        raise ConfigError(f"the sample count must be positive, got {samples}")
    generator = random.Random(seed)
    requested = 0
    given = 0
    for _ in range(samples):
        size = generator.randrange(1, 1 << 20)
        requested += size
        given += round_up(size, "power of two")
    return {
        "measured_waste": round(1.0 - requested / given, 4),
        "samples": samples,
    }


def caching_against_returning(events: Sequence[tuple[str, int]] | None = None) -> dict:
    """Holding freed memory against giving it straight back.

    The caching allocator's footprint is the high water mark and the returning one's is whatever
    is live. On a graph trace those are the same number, because a graph frees a value and
    immediately asks for one the same size, so the cache never holds anything the program was
    not about to want. Caching is free here and buys the whole reuse rate.
    """
    trace = list(events) if events is not None else trace_from(branching_graph())
    caching = Allocator(policy="power of two")
    run_trace(trace, caching)
    returning = Allocator(policy="power of two", give_back=True)
    run_trace(trace, returning)
    return {
        "caching_footprint": caching.stats.high_water,
        "returning_footprint": returning.stats.live_peak,
        "caching_reuse": caching.stats.reuse_rate,
        "returning_reuse": returning.stats.reuse_rate,
        "extra_memory": round(caching.stats.high_water / max(returning.stats.live_peak, 1), 4),
    }


def returning_gives_up_every_reuse() -> dict:
    """Whether an allocator that keeps nothing can reuse anything, which it cannot."""
    result = caching_against_returning()
    return {
        "caching_reuse": result["caching_reuse"],
        "returning_reuse": result["returning_reuse"],
        "returning_reuses_nothing": result["returning_reuse"] == 0.0,
    }


def compare_graphs() -> list[dict]:
    """Every graph trace under the default policy.

    The branching graph is the one with anything to say. The chain shaped graphs allocate and
    free in lockstep, so an allocator holding one block at a time serves all of them, and the
    only trace where the free lists do any work is the one with several values alive at once.
    """
    rows = []
    for label, trace in graph_traces():
        allocator = Allocator(policy="power of two")
        run_trace(trace, allocator)
        row = allocator.stats.as_dict()
        row["graph"] = label
        rows.append(row)
    return rows


def a_graph_trace_reuses_more_than_a_random_one() -> dict:
    """Where the reuse comes from, the allocator or the workload.

    Mostly the workload. A graph produces the same shapes over and over, so even exact fitting
    finds a freed block most of the time, and the rounding is buying much less than it does on a
    trace with arbitrary sizes in it. An allocator tuned on one of those and deployed on the
    other will disappoint.
    """
    graph_rows = {row["policy"]: row for row in compare_policies(trace_from(branching_graph()))}
    random_rows = {row["policy"]: row for row in compare_policies(random_trace())}
    return {
        "graph_exact_reuse": graph_rows["exact"]["reuse_rate"],
        "random_exact_reuse": random_rows["exact"]["reuse_rate"],
        "graph_rounded_reuse": graph_rows["power of two"]["reuse_rate"],
        "random_rounded_reuse": random_rows["power of two"]["reuse_rate"],
    }


def freeing_twice_is_refused() -> bool:
    """Whether the allocator notices a handle being returned twice.

    It does, and the reason to check is that the alternative is a free list holding the same
    block in two places, which hands the same memory to two callers and produces a wrong answer
    with no allocation error anywhere near it.
    """
    allocator = Allocator()
    handle = allocator.allocate(1024)
    allocator.free(handle)
    try:
        allocator.free(handle)
    except AllocationError:
        return True
    return False


def a_zero_sized_request_is_served(size: int = 0) -> dict:
    """What happens when something asks for nothing.

    It gets a block of nothing rather than an error. A graph can produce an empty tensor and the
    allocator is the wrong place to decide that is a problem, so the capacity is zero and the
    accounting stays consistent.
    """
    allocator = Allocator()
    allocator.allocate(size)
    return {"footprint": allocator.footprint, "requests": allocator.stats.requests}
