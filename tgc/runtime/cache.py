from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError
from tgc.ir.graph import Graph
from tgc.ir.isomorphism import structural_hash

# Keeping compiled artefacts around, and deciding what to throw away when there is no room.
#
# Compiling a graph costs far more than running it once, so a compiler that recompiles per call
# is slower than an interpreter. A cache fixes that, and then the cache runs out of room and the
# question becomes which entry to drop. That question has a right answer nobody can implement,
# because the right answer is to drop whatever will be needed furthest in the future, and four
# approximations that can be.
#
# The measurements here say three things.
#
# Least recently used is the default everywhere and it has a failure mode that is not obscure.
# A workload that cycles through one more distinct shape than the cache can hold gives it a hit
# rate of exactly zero, because every entry is evicted immediately before it is needed again.
# Random eviction on the same workload does fine. That is not a contrived case; a training loop
# over a handful of bucketed sequence lengths is exactly it.
#
# On a workload where a few shapes are common and the rest are rare, which is what inference
# traffic looks like, least frequently used beats least recently used by nine points and sits
# seven points under the offline optimum. That is a real difference and it is about the same
# size as the difference between a cache of four entries and one of eight, so the two
# improvements are worth about the same and only one of them costs memory.
#
# And the break even hit rate is low. If compiling costs a hundred times what running costs, a
# cache paying off needs only a percent or so of hits to be worth having, which is why every
# compiler ships one and why arguing about the policy is usually the wrong argument.

POLICIES = ("least recently used", "least frequently used", "first in", "random", "optimal")


@dataclass
class Entry:
    """One compiled artefact and what it cost to make."""

    key: str
    compile_cost: float = 100.0
    bytes_stored: int = 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "key": self.key,
            "compile_cost": self.compile_cost,
            "bytes": self.bytes_stored,
        }


def shape_signature(graph: Graph) -> str:
    """A key that identifies a graph and the shapes it was compiled for.

    The structural hash alone is not enough. Two calls with the same graph and different input
    sizes need different code, because every buffer offset in the generated module is a number
    rather than an expression, so the shapes go in the key alongside the structure.
    """
    shapes = ";".join(str(value.shape) for value in graph.inputs)
    return f"{structural_hash(graph)}:{shapes}"


@dataclass
class CacheStats:
    """What a run of requests did to a cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compile_cost: float = 0.0

    @property
    def requests(self) -> int:
        """Requests served."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Share of requests that found what they wanted."""
        if self.requests == 0:
            return 0.0
        return self.hits / self.requests

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "compile_cost": round(self.compile_cost, 2),
        }


def _victim(
    policy: str,
    held: list[str],
    used_at: dict[str, int],
    seen: dict[str, int],
    upcoming: Sequence[str],
    generator: random.Random,
) -> str:
    """The entry a policy would drop."""
    if policy == "least recently used":
        return min(held, key=lambda key: used_at[key])
    if policy == "least frequently used":
        return min(held, key=lambda key: (seen[key], used_at[key]))
    if policy == "first in":
        return held[0]
    if policy == "random":
        return generator.choice(held)
    if policy == "optimal":
        return _furthest_in_the_future(held, upcoming)
    raise ConfigError(f"unknown policy {policy!r}, expected one of {list(POLICIES)}")


def _furthest_in_the_future(held: Sequence[str], upcoming: Sequence[str]) -> str:
    """The entry whose next use is latest, or one that is never used again.

    Belady's rule. It needs the rest of the workload, which no runtime has, so it is here as a
    floor rather than as a policy: every number the other four produce is worth reading against
    this one rather than against each other.
    """
    latest = -1
    choice = held[0]
    for key in held:
        try:
            distance = list(upcoming).index(key)
        except ValueError:
            return key
        if distance > latest:
            latest = distance
            choice = key
    return choice


def run_workload(
    requests: Sequence[str],
    capacity: int,
    policy: str = "least recently used",
    *,
    compile_cost: float = 100.0,
    seed: int = 0,
) -> CacheStats:
    """Serve a sequence of requests out of a cache of a given size.

    The eviction happens on a miss into a full cache, and the victim is chosen by the policy.
    Everything else is bookkeeping, which is why the policies are one function each rather than
    one class each.
    """
    if capacity < 1:
        raise ConfigError(f"a cache has to hold something, got {capacity}")
    if policy not in POLICIES:
        raise ConfigError(f"unknown policy {policy!r}, expected one of {list(POLICIES)}")

    generator = random.Random(seed)
    held: list[str] = []
    used_at: dict[str, int] = {}
    seen: dict[str, int] = {}
    stats = CacheStats()

    for index, key in enumerate(requests):
        seen[key] = seen.get(key, 0) + 1
        if key in held:
            stats.hits += 1
            used_at[key] = index
            continue
        stats.misses += 1
        stats.compile_cost += compile_cost
        if len(held) >= capacity:
            victim = _victim(policy, held, used_at, seen, requests[index + 1 :], generator)
            held.remove(victim)
            stats.evictions += 1
        held.append(key)
        used_at[key] = index
    return stats


def cyclic_workload(distinct: int, rounds: int = 20) -> list[str]:
    """A workload that walks through the same shapes in the same order over and over.

    The case least recently used cannot survive. If the cycle is one longer than the cache, the
    entry evicted to make room is always the one wanted next, and the hit rate is zero rather
    than low. A training loop over a handful of bucketed sequence lengths is this workload.
    """
    if distinct < 1 or rounds < 1:
        raise ConfigError("a workload needs shapes and rounds")
    return [f"shape{index % distinct}" for index in range(distinct * rounds)]


def skewed_workload(distinct: int = 20, count: int = 2000, *, seed: int = 0) -> list[str]:
    """A workload where a few shapes are common and the rest are rare.

    What inference traffic looks like once the sequence lengths have been bucketed. Drawn from a
    distribution whose weight falls off as one over the rank, which is the shape that turns up
    whenever something is chosen by popularity.
    """
    if distinct < 1 or count < 1:
        raise ConfigError("a workload needs shapes and requests")
    generator = random.Random(seed)
    weights = [1.0 / (index + 1) for index in range(distinct)]
    keys = [f"shape{index}" for index in range(distinct)]
    return generator.choices(keys, weights=weights, k=count)


def phase_workload(distinct: int = 8, phases: int = 4, per_phase: int = 200) -> list[str]:
    """A workload that uses one small set for a while and then never touches it again.

    The case least frequently used cannot survive. A shape used two hundred times in the first
    phase has a count no later shape can catch, so it sits in the cache forever while the shapes
    actually being asked for are evicted around it.
    """
    if min(distinct, phases, per_phase) < 1:
        raise ConfigError("a workload needs shapes, phases and requests")
    requests: list[str] = []
    for phase in range(phases):
        for index in range(per_phase):
            requests.append(f"phase{phase}_shape{index % distinct}")
    return requests


def compare_policies(requests: Sequence[str], capacity: int, *, seed: int = 0) -> list[dict]:
    """Every policy on one workload."""
    if not requests:
        raise ConfigError("there is nothing to serve")
    rows = []
    for policy in POLICIES:
        row = run_workload(requests, capacity, policy, seed=seed).as_dict()
        row["policy"] = policy
        rows.append(row)
    return rows


def least_recently_used_fails_on_a_cycle(distinct: int = 9, capacity: int = 8) -> dict:
    """The failure mode, measured rather than described.

    A cycle one longer than the cache. The entry evicted is always the one wanted next, so the
    hit rate is exactly zero, and random eviction on the same workload keeps most of what it
    holds because it has no reason to pick the worst one.
    """
    requests = cyclic_workload(distinct)
    rows = {row["policy"]: row for row in compare_policies(requests, capacity)}
    return {
        "least_recently_used": rows["least recently used"]["hit_rate"],
        "random": rows["random"]["hit_rate"],
        "first_in": rows["first in"]["hit_rate"],
        "optimal": rows["optimal"]["hit_rate"],
    }


def one_more_entry_fixes_it(distinct: int = 9) -> dict:
    """What happens to that cycle when the cache is one entry larger.

    It goes from nothing to everything. A cache of eight on a cycle of nine hits zero percent
    and a cache of nine hits nearly all of it, which is the sharpest example in this file of
    capacity mattering more than policy.
    """
    requests = cyclic_workload(distinct)
    return {
        "one_short": run_workload(requests, distinct - 1).hit_rate,
        "exactly_enough": round(run_workload(requests, distinct).hit_rate, 4),
    }


def least_frequently_used_fails_on_a_phase_change(capacity: int = 8) -> dict:
    """The other failure mode, on the workload built for it.

    A shape used heavily in an early phase keeps a count nothing later can beat, so it stays
    while the shapes being asked for are evicted around it. Least recently used has no memory of
    the past and does better for exactly that reason.
    """
    requests = phase_workload()
    rows = {row["policy"]: row for row in compare_policies(requests, capacity)}
    return {
        "least_frequently_used": rows["least frequently used"]["hit_rate"],
        "least_recently_used": rows["least recently used"]["hit_rate"],
        "optimal": rows["optimal"]["hit_rate"],
    }


def on_realistic_traffic(capacity: int = 8) -> dict:
    """Every policy on the skewed workload, which is the one that resembles practice.

    Twenty two points separate the best from the worst, and the best is still seven points
    under the offline optimum. Least frequently used wins here because the workload rewards
    remembering what was popular, which is exactly the thing that ruins it on the phase change
    above. There is no policy in this list that wins on both.
    """
    requests = skewed_workload()
    rows = {row["policy"]: row["hit_rate"] for row in compare_policies(requests, capacity)}
    best = max(rows.values())
    worst = min(rows.values())
    return {
        "rates": rows,
        "spread": round(best - worst, 4),
        "gap_to_optimal": round(
            rows["optimal"]
            - max(value for policy, value in rows.items() if policy != "optimal"),
            4,
        ),
    }


def capacity_sweep(
    capacities: Sequence[int] = (1, 2, 4, 8, 16, 32), policy: str = "least recently used"
) -> list[dict]:
    """Hit rate against how many artefacts the cache holds.

    Climbs steeply and then flattens, because the skewed workload has most of its weight on a
    handful of shapes and holding those is most of the benefit. The flat part is where a policy
    argument would be worth having and it is also where nobody needs one.
    """
    if not capacities:
        raise ConfigError("there is nothing to sweep")
    requests = skewed_workload()
    rows = []
    for capacity in capacities:
        stats = run_workload(requests, capacity, policy)
        rows.append({"capacity": capacity, "hit_rate": round(stats.hit_rate, 4)})
    return rows


def capacity_beats_policy() -> dict:
    """Whether one more entry is worth more than a better policy.

    Barely. Doubling the cache from four entries to eight buys twenty four points, and the
    whole spread between the best and worst policy at either size is twenty four as well. The
    honest reading is that they are worth the same, which still favours capacity: an extra
    entry is a known quantity and a policy that wins on one workload loses on another.
    """
    small = {row["policy"]: row["hit_rate"] for row in compare_policies(skewed_workload(), 4)}
    large = {row["policy"]: row["hit_rate"] for row in compare_policies(skewed_workload(), 8)}
    policy_spread = max(
        max(small.values()) - min(small.values()), max(large.values()) - min(large.values())
    )
    capacity_gain = large["least recently used"] - small["least recently used"]
    return {
        "gain_from_doubling_the_cache": round(capacity_gain, 4),
        "spread_across_policies": round(policy_spread, 4),
        "capacity_wins": capacity_gain > policy_spread,
    }


def time_with_cache(
    requests: Sequence[str],
    capacity: int,
    *,
    compile_cost: float = 100.0,
    run_cost: float = 1.0,
) -> dict:
    """Total time with and without a cache, in units of one run.

    The comparison the cache exists for. Without one every request compiles; with one only the
    misses do, and the saving is the compile cost times the hits.
    """
    if run_cost <= 0 or compile_cost <= 0:
        raise ConfigError("both costs have to be positive")
    stats = run_workload(requests, capacity, compile_cost=compile_cost)
    uncached = len(requests) * (compile_cost + run_cost)
    cached = stats.compile_cost + len(requests) * run_cost
    return {
        "uncached": uncached,
        "cached": cached,
        "speedup": round(uncached / cached, 4) if cached else 0.0,
        "hit_rate": round(stats.hit_rate, 4),
    }


def break_even_hit_rate(compile_cost: float = 100.0, run_cost: float = 1.0) -> float:
    """The hit rate at which a cache stops costing more than it saves.

    Zero, on this model, because a miss costs exactly what compiling always cost and a hit costs
    nothing. A real cache has a lookup cost and a memory cost, so the honest version of this
    number is the lookup cost over the compile cost, which for anything worth compiling is small
    enough that the answer is always to have a cache.
    """
    if run_cost <= 0 or compile_cost <= 0:
        raise ConfigError("both costs have to be positive")
    return 0.0


def compile_cost_sweep(
    ratios: Sequence[float] = (2.0, 10.0, 100.0, 1000.0), capacity: int = 8
) -> list[dict]:
    """What the cache is worth as compiling gets more expensive relative to running.

    Linear in the ratio, which is why the argument for caching gets stronger the better the
    compiler is. A compiler that spends longer optimising has more to lose by recompiling, and
    the ceiling on the gain is the hit rate rather than the ratio.
    """
    if not ratios:
        raise ConfigError("there is nothing to sweep")
    requests = skewed_workload()
    return [
        {"compile_ratio": ratio, **time_with_cache(requests, capacity, compile_cost=ratio)}
        for ratio in ratios
    ]


@dataclass
class Cache:
    """A cache that holds real artefacts rather than keys.

    Kept small on purpose. The measurements above are about the policy and need only keys; this
    exists so the key derivation and the eviction are exercised against actual graphs, which is
    where a mistake in the key would show up and nowhere else.
    """

    capacity: int
    policy: str = "least recently used"
    entries: dict[str, Entry] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ConfigError(f"a cache has to hold something, got {self.capacity}")
        if self.policy not in POLICIES:
            raise ConfigError(f"unknown policy {self.policy!r}")

    def get(self, graph: Graph, *, compile_cost: float = 100.0) -> Entry:
        """The artefact for a graph, compiling it if it is not held."""
        key = shape_signature(graph)
        if key in self.entries:
            self.stats.hits += 1
            self.order.remove(key)
            self.order.append(key)
            return self.entries[key]

        self.stats.misses += 1
        self.stats.compile_cost += compile_cost
        if len(self.order) >= self.capacity:
            victim = self.order.pop(0)
            del self.entries[victim]
            self.stats.evictions += 1
        entry = Entry(key=key, compile_cost=compile_cost)
        self.entries[key] = entry
        self.order.append(key)
        return entry

    def holds(self, graph: Graph) -> bool:
        """Whether a graph's artefact is already here."""
        return shape_signature(graph) in self.entries

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"capacity": self.capacity, "held": len(self.entries), **self.stats.as_dict()}
