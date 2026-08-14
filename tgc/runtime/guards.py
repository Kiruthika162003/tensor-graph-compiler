from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, ScheduleError
from tgc.ir.graph import Graph
from tgc.ir.shape import Shape

# Deciding when a compiled program is allowed to run.
#
# A graph compiled for one set of shapes is not valid for another, so every compiled artifact
# carries a guard: a predicate over the incoming shapes that has to hold before the artifact
# is used. Get the guard too tight and every request recompiles; get it too loose and the
# artifact runs on shapes it was not built for, which is the worst of the two by a distance.
#
# Bucketing is the standard answer and it is a trade rather than a fix. Rounding a sequence
# length up to the next bucket makes one artifact serve a range of shapes, and the padding is
# arithmetic performed on values that are then thrown away. The right bucket count is the one
# where the recompilation saved stops being worth the padding paid, and that crossover depends
# on how long a compile takes, which nobody puts in the model.


@dataclass(frozen=True)
class Guard:
    """A condition an incoming shape has to satisfy."""

    dimension: int
    exact: int | None = None
    lower: int | None = None
    upper: int | None = None

    def __post_init__(self) -> None:
        if self.dimension < 0:
            raise ConfigError(f"a dimension index cannot be negative, got {self.dimension}")
        if self.exact is None and self.lower is None and self.upper is None:
            raise ConfigError("a guard has to constrain something")
        if self.exact is not None and (self.lower is not None or self.upper is not None):
            raise ConfigError("a guard is either exact or a range")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ConfigError(f"an empty range from {self.lower} to {self.upper}")

    @property
    def is_exact(self) -> bool:
        """Whether the guard pins the dimension to one value."""
        return self.exact is not None

    def admits(self, size: int) -> bool:
        """Whether a size satisfies the guard."""
        if self.is_exact:
            return size == self.exact
        if self.lower is not None and size < self.lower:
            return False
        return not (self.upper is not None and size > self.upper)

    @property
    def width(self) -> int:
        """How many sizes the guard admits, when that is finite."""
        if self.is_exact:
            return 1
        if self.lower is None or self.upper is None:
            raise ScheduleError("an open guard admits unboundedly many sizes")
        return self.upper - self.lower + 1

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "dimension": self.dimension,
            "exact": self.exact,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass
class GuardSet:
    """Every condition one compiled artifact requires."""

    guards: list[Guard] = field(default_factory=list)

    def admits(self, shape: Shape) -> bool:
        """Whether a shape satisfies every guard."""
        for guard in self.guards:
            if guard.dimension >= shape.rank:
                return False
            size = shape.sizes[guard.dimension]
            if not size.is_static or not guard.admits(size.value or 0):
                return False
        return True

    @property
    def is_fully_static(self) -> bool:
        """Whether every guard pins its dimension."""
        return bool(self.guards) and all(guard.is_exact for guard in self.guards)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "guards": len(self.guards),
            "fully_static": self.is_fully_static,
        }


def exact_guards(shape: Shape) -> GuardSet:
    """A guard set that admits only the shape it was built from.

    The tightest possible specialisation. Every distinct input shape gets its own artifact,
    which produces the fastest code and the worst compile behaviour, and is the default in
    every compiler until somebody looks at the recompile count.
    """
    guards = []
    for index, size in enumerate(shape.sizes):
        if size.is_static:
            guards.append(Guard(dimension=index, exact=size.value))
    return GuardSet(guards=guards)


def bucket_guards(shape: Shape, dimension: int, buckets: Sequence[int]) -> GuardSet:
    """A guard set that admits a range on one dimension and pins the rest."""
    if not buckets:
        raise ConfigError("there has to be at least one bucket")
    if dimension >= shape.rank:
        raise ConfigError(f"dimension {dimension} is outside a shape of rank {shape.rank}")

    size = shape.sizes[dimension]
    if not size.is_static:
        raise ConfigError("a bucket needs a concrete size to place")
    chosen = bucket_for(size.value or 0, buckets)
    lower = 1
    for boundary in sorted(buckets):
        if boundary == chosen:
            break
        lower = boundary + 1

    guards = [Guard(dimension=dimension, lower=lower, upper=chosen)]
    for index, other in enumerate(shape.sizes):
        if index != dimension and other.is_static:
            guards.append(Guard(dimension=index, exact=other.value))
    return GuardSet(guards=guards)


def bucket_for(size: int, buckets: Sequence[int]) -> int:
    """The smallest bucket that holds a size."""
    if size < 0:
        raise ConfigError("a size cannot be negative")
    ordered = sorted(buckets)
    for boundary in ordered:
        if size <= boundary:
            return boundary
    raise ScheduleError(f"{size} is larger than the largest bucket {ordered[-1]}")


def geometric_buckets(smallest: int = 8, largest: int = 2048, ratio: float = 2.0) -> list[int]:
    """Buckets that grow by a fixed factor.

    Geometric rather than uniform, because the padding a request pays is proportional to its
    own size. Uniform buckets waste a constant number of positions, which is most of a short
    request and a rounding error on a long one.
    """
    if smallest < 1 or largest < smallest:
        raise ConfigError("the bucket range has to be positive and increasing")
    if ratio <= 1.0:
        raise ConfigError(f"the ratio has to grow, got {ratio}")
    buckets = []
    current = smallest
    while current < largest:
        buckets.append(int(current))
        current *= ratio
    buckets.append(largest)
    return buckets


def uniform_buckets(smallest: int = 8, largest: int = 2048, step: int = 128) -> list[int]:
    """Buckets spaced by a fixed number of positions."""
    if step < 1:
        raise ConfigError(f"the step has to be positive, got {step}")
    if smallest < 1 or largest < smallest:
        raise ConfigError("the bucket range has to be positive and increasing")
    buckets = list(range(smallest, largest + 1, step))
    if buckets[-1] != largest:
        buckets.append(largest)
    return buckets


@dataclass
class CacheReport:
    """What a run of requests did to a compiled artifact cache."""

    compiles: int = 0
    hits: int = 0
    padded_positions: int = 0
    real_positions: int = 0

    @property
    def requests(self) -> int:
        """Requests served."""
        return self.compiles + self.hits

    @property
    def hit_rate(self) -> float:
        """Share of requests that found an artifact."""
        if self.requests == 0:
            return 0.0
        return self.hits / self.requests

    @property
    def padding_overhead(self) -> float:
        """Share of the arithmetic performed on padding."""
        if self.real_positions == 0:
            return 0.0
        return self.padded_positions / self.real_positions

    def total_seconds(self, compile_seconds: float, position_seconds: float) -> float:
        """Time the whole run took, compiles and padding included.

        The number the bucket count is actually chosen against, and the one nobody has: it
        needs a compile time, and a compile time depends on the graph, the backend and the
        machine, so a bucket count tuned on one deployment is a guess on the next.
        """
        if compile_seconds < 0 or position_seconds < 0:
            raise ConfigError("the times cannot be negative")
        work = self.real_positions + self.padded_positions
        return self.compiles * compile_seconds + work * position_seconds

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "requests": self.requests,
            "compiles": self.compiles,
            "hit_rate": round(self.hit_rate, 4),
            "padding_overhead": round(self.padding_overhead, 4),
        }


def run_requests(sizes: Sequence[int], buckets: Sequence[int] | None = None) -> CacheReport:
    """Serve a stream of sequence lengths, compiling when nothing admits them.

    With no buckets every distinct length compiles, which is the exact specialisation case.
    With buckets a length is rounded up and the padding is counted, which is arithmetic
    performed on values that are then thrown away.
    """
    if not sizes:
        raise ConfigError("there is nothing to serve")

    report = CacheReport()
    seen: set[int] = set()
    for size in sizes:
        key = bucket_for(size, buckets) if buckets else size
        if key in seen:
            report.hits += 1
        else:
            seen.add(key)
            report.compiles += 1
        report.real_positions += size
        report.padded_positions += key - size
    return report


def realistic_lengths(count: int = 500, seed: int = 0) -> list[int]:
    """A stream of sequence lengths with the shape real traffic has.

    Log normal, because request lengths are: most are short, a few are very long, and the
    mean sits well above the median. A uniform stream makes bucketing look better than it is,
    since uniform lengths land evenly inside their buckets.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    generator = random.Random(seed)
    return [max(1, min(2048, int(math.exp(generator.gauss(4.6, 1.0))))) for _ in range(count)]


def compare_bucketing(count: int = 500, seed: int = 0) -> list[dict]:
    """Exact specialisation against two bucket schemes, on realistic traffic.

    Exact compiles for over half the requests and pads nothing. Geometric buckets compile
    nine times and pad by 45 percent. Uniform buckets compile more often than geometric and
    pad about the same in aggregate, which looks like a tie and is not: the aggregate is
    dominated by long requests, and on requests under sixty four positions uniform pads by
    273 percent against geometric's 36. See short_request_padding.
    """
    lengths = realistic_lengths(count=count, seed=seed)
    schemes = {
        "exact": None,
        "geometric": geometric_buckets(),
        "uniform": uniform_buckets(),
    }
    rows = []
    for name, buckets in schemes.items():
        report = run_requests(lengths, buckets)
        row = report.as_dict()
        row["scheme"] = name
        row["buckets"] = len(buckets) if buckets else 0
        rows.append(row)
    return rows


def choose_bucket_count(
    count: int = 500,
    seed: int = 0,
    compile_seconds: float = 2.0,
    position_seconds: float = 1e-6,
) -> list[dict]:
    """Total time against the number of buckets, which is where the crossover sits.

    Few buckets means little compiling and a lot of padding; many means the reverse. The
    minimum moves with the compile time, which is why a bucket count that was right on one
    deployment is a guess on the next.
    """
    lengths = realistic_lengths(count=count, seed=seed)
    rows = []
    for ratio in (4.0, 3.0, 2.0, 1.5, 1.25, 1.1):
        buckets = geometric_buckets(ratio=ratio)
        report = run_requests(lengths, buckets)
        rows.append(
            {
                "ratio": ratio,
                "buckets": len(buckets),
                "compiles": report.compiles,
                "padding_overhead": round(report.padding_overhead, 4),
                "seconds": round(report.total_seconds(compile_seconds, position_seconds), 4),
            }
        )
    return rows


def guards_for(graph: Graph, *, dimension: int = 0, buckets: Sequence[int] | None = None):
    """The guard set a compiled artifact for this graph would carry."""
    if not graph.inputs:
        raise ConfigError("a graph with no inputs needs no guards")
    shape = graph.inputs[0].shape
    if buckets is None:
        return exact_guards(shape)
    return bucket_guards(shape, dimension, buckets)


def short_request_padding(threshold: int = 64, count: int = 500, seed: int = 0) -> list[dict]:
    """Padding paid by the short requests alone, under each bucket scheme.

    The comparison the aggregate hides. Uniform buckets waste a constant number of positions,
    which is a rounding error on a long request and several times the work on a short one, and
    short requests are the majority of a log normal stream.
    """
    lengths = [size for size in realistic_lengths(count=count, seed=seed) if size < threshold]
    if not lengths:
        raise ScheduleError(f"no request came in under {threshold} positions")
    rows = []
    for name, buckets in (("geometric", geometric_buckets()), ("uniform", uniform_buckets())):
        report = run_requests(lengths, buckets)
        rows.append(
            {
                "scheme": name,
                "requests": len(lengths),
                "padding_overhead": round(report.padding_overhead, 4),
            }
        )
    return rows


def crossover_moves_with_compile_time(count: int = 500, seed: int = 0) -> list[dict]:
    """The best bucket count at several compile times.

    A slow compiler wants few buckets and a fast one wants many, and the optimum moves by a
    factor of two across a range of compile times that any two deployments could plausibly
    differ by. Which is the argument against shipping a bucket count as a constant.
    """
    rows = []
    for compile_seconds in (2.0, 0.05, 0.005):
        sweep = choose_bucket_count(count=count, seed=seed, compile_seconds=compile_seconds)
        best = min(sweep, key=lambda row: row["seconds"])
        rows.append(
            {
                "compile_seconds": compile_seconds,
                "best_ratio": best["ratio"],
                "buckets": best["buckets"],
                "seconds": best["seconds"],
            }
        )
    return rows
