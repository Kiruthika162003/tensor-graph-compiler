from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from tgc.analysis.cost import GPU, Machine, estimate, kendall_agreement
from tgc.errors import ConfigError, ScheduleError
from tgc.ir.graph import Graph
from tgc.runtime.executor import compile_graph
from tgc.verify.reference import evaluate_node, interpret, random_feeds, run

# Timing what actually ran, and comparing it to what the model said.
#
# A cost model is a claim and a timing is a measurement, and the only reason to keep both is
# to find out where they part company. That comparison needs the timing to be worth trusting,
# which on a machine running other things means several samples and a statistic that is not
# the mean.
#
# The median is used throughout. A mean over ten runs where one was interrupted by something
# else on the machine reports the interruption; the median reports the run. The minimum is the
# other defensible choice and says something different again: it is the time when nothing else
# happened, which is a real number and not the one a user experiences.
#
# Warm up is not optional and is not superstition. The first call through this compiler
# allocates an arena, and the first call through torch resolves dispatch and may allocate its
# own workspaces, so a single timed call measures setup and reports it as arithmetic.


@dataclass
class Timing:
    """Several samples of the same thing."""

    label: str
    samples: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.label:
            raise ConfigError("a timing needs a label")
        if any(sample < 0 for sample in self.samples):
            raise ConfigError("a duration cannot be negative")

    @property
    def count(self) -> int:
        """Samples taken."""
        return len(self.samples)

    @property
    def median(self) -> float:
        """The middle sample, which is what the comparisons use."""
        if not self.samples:
            raise ScheduleError(f"{self.label} has no samples")
        return statistics.median(self.samples)

    @property
    def fastest(self) -> float:
        """The best sample, which is the time when nothing else happened."""
        if not self.samples:
            raise ScheduleError(f"{self.label} has no samples")
        return min(self.samples)

    @property
    def spread(self) -> float:
        """How much the samples vary, relative to the median.

        The number that says whether a comparison is meaningful. Two timings differing by five
        percent with a spread of thirty percent have not been distinguished, and reporting the
        difference anyway is how a benchmark ends up measuring the weather.
        """
        if self.count < 2:
            return 0.0
        middle = self.median
        if middle == 0:
            return 0.0
        return (max(self.samples) - min(self.samples)) / middle

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "label": self.label,
            "samples": self.count,
            "median": self.median,
            "fastest": self.fastest,
            "spread": round(self.spread, 4),
        }


def time_call(
    function: Callable[[], object], *, repeats: int = 9, warmups: int = 2, label: str = "call"
) -> Timing:
    """Run something several times and keep the durations.

    The warm up runs are discarded rather than averaged in. Including them makes the first
    call's setup part of every reported number, which is a smaller error than reporting setup
    alone and is still an error.
    """
    if repeats < 1:
        raise ConfigError(f"there has to be at least one repeat, got {repeats}")
    if warmups < 0:
        raise ConfigError(f"the warm up count cannot be negative, got {warmups}")

    for _ in range(warmups):
        function()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return Timing(label=label, samples=samples)


def time_interpreter(graph: Graph, *, repeats: int = 9, seed: int = 0) -> Timing:
    """How long the reference interpreter takes on a graph."""
    feeds = random_feeds(graph, seed=seed, positive=True)
    return time_call(lambda: run(graph, feeds), repeats=repeats, label="interpreter")


def time_compiled(graph: Graph, *, repeats: int = 9, seed: int = 0) -> Timing:
    """How long the compiled program takes on the same graph.

    Compiled once outside the timed region, because compilation is not what is being
    measured and including it makes the first sample a hundred times the others.
    """
    feeds = random_feeds(graph, seed=seed, positive=True)
    compiled = compile_graph(graph)
    return time_call(lambda: compiled(feeds), repeats=repeats, label="compiled")


def time_compiled_reusing_arena(graph: Graph, *, repeats: int = 9, seed: int = 0) -> Timing:
    """The compiled program with one arena kept across calls.

    Which is how a real caller would use it. The default path allocates a fresh arena on every
    call so that nothing leaks between them, and that allocation is a torch.zeros of the whole
    working set, which on a small graph costs more than the arithmetic does.
    """
    feeds = random_feeds(graph, seed=seed, positive=True)
    compiled = compile_graph(graph)
    arena = compiled.new_arena()
    return time_call(
        lambda: compiled.function(arena, feeds), repeats=repeats, label="compiled reusing"
    )


def compare_execution(graph: Graph, *, repeats: int = 9) -> dict:
    """The interpreter against the compiled program, with the spread reported alongside.

    The spread is there so the ratio can be read honestly, and on these graphs it says the two
    have not been distinguished at all. That is the correct reading and the interesting one.
    This backend emits Python that calls torch, and the interpreter is Python calling the same
    torch, so the same kernels run either way and there is no arithmetic to win. What the
    compiled path buys is the memory plan, which the arena measurements elsewhere quantify and
    a stopwatch cannot see.
    """
    interpreted = time_interpreter(graph, repeats=repeats)
    compiled = time_compiled(graph, repeats=repeats)
    return {
        "interpreter_median": interpreted.median,
        "compiled_median": compiled.median,
        "ratio": round(interpreted.median / compiled.median, 3) if compiled.median else 0.0,
        "worst_spread": round(max(interpreted.spread, compiled.spread), 4),
        "distinguishable": abs(interpreted.median - compiled.median)
        > max(interpreted.spread, compiled.spread) * min(interpreted.median, compiled.median),
    }


def arena_allocation_cost(graph: Graph, *, repeats: int = 9) -> dict:
    """What allocating a fresh arena on every call costs.

    The default path calls torch.zeros over the whole working set before running anything, so
    on a small graph the allocation dominates. Reusing one arena is what a caller would
    actually do, and the difference between the two rows is the allocation rather than any
    property of the generated code.
    """
    fresh = time_compiled(graph, repeats=repeats)
    reused = time_compiled_reusing_arena(graph, repeats=repeats)
    return {
        "fresh_arena_median": fresh.median,
        "reused_arena_median": reused.median,
        "allocation_share": round(1.0 - reused.median / fresh.median, 4)
        if fresh.median
        else 0.0,
    }


def model_against_measurement(
    graphs: Sequence[tuple[str, Graph]], machine: Machine = GPU, *, repeats: int = 5
) -> list[dict]:
    """The predicted ranking against the measured one.

    Not the predicted times against the measured times, which would be comparing a model of a
    GPU against a run on a CPU and is meaningless. The ranking is the only thing the model
    claims and the only thing worth checking.
    """
    if not graphs:
        raise ConfigError("there is nothing to compare")
    rows = []
    for name, graph in graphs:
        timing = time_compiled(graph, repeats=repeats)
        rows.append(
            {
                "graph": name,
                "predicted_seconds": estimate(graph, machine).seconds,
                "measured_median": timing.median,
                "spread": round(timing.spread, 4),
            }
        )
    return rows


def ranking_agreement(
    graphs: Sequence[tuple[str, Graph]], machine: Machine = GPU, *, repeats: int = 5
) -> float:
    """How closely the model's order matches the measured one."""
    rows = model_against_measurement(graphs, machine, repeats=repeats)
    if len(rows) < 2:
        raise ConfigError("a ranking of one has no order to compare")
    predicted = [row["graph"] for row in sorted(rows, key=lambda row: row["predicted_seconds"])]
    measured = [row["graph"] for row in sorted(rows, key=lambda row: row["measured_median"])]
    return kendall_agreement(predicted, measured)


@dataclass
class NodeProfile:
    """Time attributed to each node of a graph."""

    per_node: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Time across every node."""
        return sum(self.per_node.values())

    def hottest(self, count: int = 3) -> list[tuple[str, float]]:
        """The nodes that took longest, which is where any optimisation should start."""
        if count < 1:
            raise ConfigError(f"the count must be positive, got {count}")
        return sorted(self.per_node.items(), key=lambda item: -item[1])[:count]

    def share_of(self, name: str) -> float:
        """What fraction of the run one node accounted for."""
        if name not in self.per_node:
            raise ScheduleError(f"{name} was not profiled")
        if self.total == 0:
            return 0.0
        return self.per_node[name] / self.total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "nodes": len(self.per_node),
            "total": self.total,
            "hottest": [name for name, _ in self.hottest()],
        }


def profile_nodes(graph: Graph, *, repeats: int = 5, seed: int = 0) -> NodeProfile:
    """Time each node of a graph separately, through the interpreter.

    Attribution by running the graph one node at a time rather than by instrumenting the
    compiled program. Instrumenting it would change what is being measured: the fused code has
    no per node boundary to put a timer at, which is the entire point of fusing it.
    """
    feeds = random_feeds(graph, seed=seed, positive=True)
    environment = interpret(graph, feeds)

    profile = NodeProfile()
    for node in graph.nodes:
        operands = [environment[name] for name in node.inputs]
        timing = time_call(
            lambda node=node, operands=operands: evaluate_node(node, operands),
            repeats=repeats,
            warmups=1,
            label=node.name,
        )
        profile.per_node[node.name] = timing.median
    return profile


def hot_node_share(graph: Graph, *, repeats: int = 5) -> float:
    """What fraction of the time the single slowest node accounts for.

    Usually most of it, which is the argument for measuring before optimising. A graph where
    one node is eighty percent of the run has one thing worth working on and a dozen that are
    not.
    """
    profile = profile_nodes(graph, repeats=repeats)
    hottest = profile.hottest(1)
    if not hottest:
        raise ScheduleError("the graph has no nodes to profile")
    return profile.share_of(hottest[0][0])


def warmup_matters(graph: Graph, *, repeats: int = 5) -> dict:
    """The first call against the steady state ones.

    The first call through the compiled function allocates an arena and the first through
    torch resolves dispatch, so a benchmark with no warm up reports setup as arithmetic.
    """
    feeds = random_feeds(graph, positive=True)
    compiled = compile_graph(graph)

    start = time.perf_counter()
    compiled(feeds)
    first = time.perf_counter() - start

    steady = time_call(lambda: compiled(feeds), repeats=repeats, warmups=2, label="steady")
    return {
        "first_call": first,
        "steady_median": steady.median,
        "ratio": round(first / steady.median, 3) if steady.median else 0.0,
    }


def deterministic_output(graph: Graph, *, repeats: int = 4) -> bool:
    """Whether repeated runs of the compiled program agree with each other.

    Timing a thing many times is only meaningful if the thing is the same each time, and the
    arena is reused across calls, so this is the property that makes the numbers above mean
    anything at all.
    """
    feeds = random_feeds(graph, positive=True)
    compiled = compile_graph(graph)
    first = [tensor.clone() for tensor in compiled(feeds)]
    for _ in range(repeats - 1):
        again = compiled(feeds)
        if not all(
            torch.equal(before, after) for before, after in zip(first, again, strict=True)
        ):
            return False
    return True
