from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from tgc.analysis.liveness import compute_intervals
from tgc.errors import ConfigError, VerificationError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, validate
from tgc.memory.planner import STRATEGIES, plan_is_valid
from tgc.runtime.executor import compile_graph
from tgc.schedule.order import depth_first_order
from tgc.verify.reference import outputs_agree, random_feeds, run

# Generating graphs nobody thought to write, and shrinking the ones that break.
#
# Every fixture in this repository was written by somebody who had a transformation in mind,
# which means every fixture is a case somebody already thought about. A generator does not
# have that problem, and the graphs it produces are ugly in ways that turn out to matter: a
# value read three times, a reduction feeding a broadcast feeding another reduction, a chain
# that ends in the same node it started from.
#
# The second half is the part that makes fuzzing usable rather than merely alarming. A
# generated counterexample is forty nodes of nonsense and says nothing. Shrinking it removes
# nodes one at a time while the failure survives, and what is left is usually three nodes and
# an obvious bug. Without shrinking a fuzzer produces work; with it, it produces answers.

ELEMENTWISE_UNARY = ("relu", "tanh", "neg", "sigmoid", "abs")
ELEMENTWISE_BINARY = ("add", "sub", "mul", "maximum", "minimum")


@dataclass
class GeneratorConfig:
    """How large and how strange the generated graphs should be."""

    nodes: int = 12
    inputs: int = 2
    rows: int = 8
    columns: int = 8
    reduction_chance: float = 0.15
    reuse_chance: float = 0.3

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ConfigError(f"a graph needs at least one node, got {self.nodes}")
        if self.inputs < 1:
            raise ConfigError(f"a graph needs at least one input, got {self.inputs}")
        if not 0.0 <= self.reduction_chance <= 1.0:
            raise ConfigError("the reduction chance has to be a probability")
        if not 0.0 <= self.reuse_chance <= 1.0:
            raise ConfigError("the reuse chance has to be a probability")

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "nodes": self.nodes,
            "inputs": self.inputs,
            "shape": [self.rows, self.columns],
            "reduction_chance": self.reduction_chance,
        }


def generate(config: GeneratorConfig | None = None, seed: int = 0) -> Graph:
    """A random graph of elementwise operations, reductions and broadcasts.

    Every value is kept at one of two shapes, the full matrix or a column, so that any two
    values can be combined. That restriction is what makes the generator produce valid graphs
    without a constraint solver, and it costs nothing worth having: the shape bugs live in
    shape inference, which has its own tests, and the bugs worth fuzzing for live in the
    passes.
    """
    settings = config or GeneratorConfig()
    generator = random.Random(seed)
    builder = Builder()

    full: list[str] = []
    columns: list[str] = []
    for index in range(settings.inputs):
        full.append(builder.input([settings.rows, settings.columns], name=f"in{index}"))

    for _ in range(settings.nodes):
        if columns and generator.random() < 0.2:
            source = generator.choice(columns)
            full.append(builder.broadcast_to(source, [settings.rows, settings.columns]))
            continue
        if generator.random() < settings.reduction_chance:
            source = generator.choice(full)
            columns.append(builder.sum(source, axes=[1], keepdims=True))
            continue
        if generator.random() < 0.4:
            source = generator.choice(full)
            operation = generator.choice(ELEMENTWISE_UNARY)
            full.append(builder.apply(_op(operation), source))
            continue
        left = generator.choice(full)
        right = generator.choice(full) if generator.random() > settings.reuse_chance else left
        operation = generator.choice(ELEMENTWISE_BINARY)
        full.append(builder.apply(_op(operation), left, right))

    return builder.finish(full[-1])


def _op(name: str):
    """Look up an operation for the generator."""
    return ops.get_op(name)


def generate_many(
    count: int = 50, config: GeneratorConfig | None = None, start: int = 0
) -> list[Graph]:
    """A batch of graphs, one per seed."""
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    return [generate(config, seed=start + index) for index in range(count)]


@dataclass
class Failure:
    """A graph a transformation got wrong."""

    seed: int
    graph: Graph
    detail: str = ""

    @property
    def size(self) -> int:
        """Nodes in the failing graph."""
        return len(self.graph.nodes)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"seed": self.seed, "nodes": self.size, "detail": self.detail}


@dataclass
class FuzzReport:
    """What a fuzzing run found."""

    checked: int = 0
    failures: list[Failure] = field(default_factory=list)

    @property
    def passed(self) -> int:
        """Graphs the transformation handled correctly."""
        return self.checked - len(self.failures)

    @property
    def clean(self) -> bool:
        """Whether anything failed at all."""
        return not self.failures

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "checked": self.checked,
            "passed": self.passed,
            "failures": len(self.failures),
            "smallest_failure": min((f.size for f in self.failures), default=0),
        }


Transform = Callable[[Graph], Graph]


def preserves_semantics(
    transform: Transform, graph: Graph, *, tolerance: float = 0.0, seed: int = 0
) -> bool:
    """Whether a transformation leaves the answer alone on one graph.

    Bit equality by default. A transformation that needs a tolerance is one that changed the
    arithmetic, and having to pass one here is a useful place to be reminded of that.
    """
    feeds = random_feeds(graph, seed=seed, positive=True)
    try:
        rewritten = transform(graph)
        validate(rewritten)
    except Exception:
        return False
    try:
        before = run(graph, feeds)
        after = run(rewritten, feeds)
    except Exception:
        return False
    return outputs_agree(before, after, tolerance=tolerance)


def fuzz_transform(
    transform: Transform,
    *,
    count: int = 50,
    config: GeneratorConfig | None = None,
    tolerance: float = 0.0,
) -> FuzzReport:
    """Run a transformation over many generated graphs and collect what it got wrong."""
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    report = FuzzReport()
    for seed in range(count):
        graph = generate(config, seed=seed)
        report.checked += 1
        if not preserves_semantics(transform, graph, tolerance=tolerance, seed=seed):
            report.failures.append(Failure(seed=seed, graph=graph))
    return report


def shrink(
    transform: Transform, graph: Graph, *, tolerance: float = 0.0, rounds: int = 32
) -> Graph:
    """Remove nodes from a failing graph while the failure survives.

    Greedy and repeated: try dropping each node, keep any drop that still fails, and go round
    again until nothing can be removed. Not minimal in the formal sense and small enough to
    read, which is the whole point. A forty node counterexample says nothing and a three node
    one usually says everything.
    """
    if rounds < 1:
        raise ConfigError(f"the round count must be positive, got {rounds}")
    if preserves_semantics(transform, graph, tolerance=tolerance):
        raise VerificationError("this graph does not fail, so there is nothing to shrink")

    current = graph
    for _ in range(rounds):
        smaller = _one_smaller_failure(transform, current, tolerance)
        if smaller is None:
            return current
        current = smaller
    return current


def _one_smaller_failure(transform: Transform, graph: Graph, tolerance: float) -> Graph | None:
    """A graph one node smaller that still fails, if there is one."""
    for index in range(len(graph.nodes)):
        candidate = _drop_node(graph, index)
        if candidate is None:
            continue
        if not preserves_semantics(transform, candidate, tolerance=tolerance):
            return candidate
    return None


def _drop_node(graph: Graph, index: int) -> Graph | None:
    """The graph without one node, rewiring its readers to one of its inputs.

    Returns nothing when the node cannot be removed without breaking the graph, which is the
    common case and is why the shrinker tries every index rather than picking one.
    """
    node = graph.nodes[index]
    if not node.inputs:
        return None
    replacement = {node.name: node.inputs[0]}
    kept = [
        other.replace_inputs(replacement)
        for position, other in enumerate(graph.nodes)
        if position != index
    ]
    outputs = [replacement.get(name, name) for name in graph.outputs]
    candidate = Graph(nodes=kept, inputs=list(graph.inputs), outputs=outputs)
    try:
        validate(candidate)
    except Exception:
        return None
    if any(
        graph.value(name).shape != candidate.value(name).shape for name in candidate.outputs
    ):
        return None
    return candidate


def differential(
    first: Transform, second: Transform, *, count: int = 50, tolerance: float = 0.0
) -> FuzzReport:
    """Whether two transformations agree with each other on generated graphs.

    Different from checking each against the interpreter, and it catches a different thing: a
    pair of passes that are each correct alone and produce different answers when composed in
    the two possible orders.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    report = FuzzReport()
    for seed in range(count):
        graph = generate(seed=seed)
        report.checked += 1
        feeds = random_feeds(graph, seed=seed, positive=True)
        try:
            left = run(first(graph), feeds)
            right = run(second(graph), feeds)
        except Exception:
            report.failures.append(Failure(seed=seed, graph=graph, detail="raised"))
            continue
        if not outputs_agree(left, right, tolerance=tolerance):
            report.failures.append(Failure(seed=seed, graph=graph, detail="disagreed"))
    return report


def broken_transform(graph: Graph) -> Graph:
    """A pass that is wrong on purpose, so the fuzzer has something to find.

    It replaces the last binary operation with its left operand, which is the shape of a real
    bug: a rewrite rule that fires on a pattern it should not, produces a graph that validates
    perfectly, and quietly drops half the computation.
    """
    for index in range(len(graph.nodes) - 1, -1, -1):
        node = graph.nodes[index]
        if len(node.inputs) == 2:
            candidate = _drop_node(graph, index)
            if candidate is not None:
                return candidate
    return graph


def generated_graph_statistics(count: int = 50) -> dict:
    """What the generator actually produces, so the coverage claim is checkable."""
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    reductions = 0
    reused = 0
    sizes = []
    for seed in range(count):
        graph = generate(seed=seed)
        sizes.append(len(graph.nodes))
        counts = graph.use_counts()
        reductions += sum(1 for node in graph.nodes if node.op.name == "sum")
        reused += sum(1 for name, uses in counts.items() if uses > 1)
    return {
        "graphs": count,
        "mean_nodes": round(sum(sizes) / count, 2),
        "reductions": reductions,
        "values_read_more_than_once": reused,
    }


def feeds_for(graph: Graph, seed: int = 0) -> dict[str, torch.Tensor]:
    """Inputs for a generated graph, kept positive so nothing produces nan."""
    return random_feeds(graph, seed=seed, positive=True)


def check_against_reference(
    transform: Transform, seeds: Sequence[int] = range(20)
) -> list[dict]:
    """One row per graph saying whether the transformation held."""
    if not list(seeds):
        raise ConfigError("there is nothing to check")
    rows = []
    for seed in seeds:
        graph = generate(seed=seed)
        rows.append(
            {
                "seed": seed,
                "nodes": len(graph.nodes),
                "preserved": preserves_semantics(transform, graph, seed=seed),
            }
        )
    return rows


def fuzz_compiler(count: int = 30, config: GeneratorConfig | None = None) -> FuzzReport:
    """Compile and run generated graphs, checking each against the interpreter.

    The end to end version, and the one that catches interactions. A pass can be correct on
    its own and wrong after the scheduler has reordered around it, or the allocator can hand
    two live values the same bytes on a shape no fixture produced. Neither shows up when the
    passes are fuzzed one at a time.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    report = FuzzReport()
    for seed in range(count):
        graph = generate(config, seed=seed)
        report.checked += 1
        feeds = feeds_for(graph, seed=seed)
        try:
            compiled = compile_graph(graph)
            produced = compiled(feeds)
            expected = run(graph, feeds)
        except Exception as failure:
            report.failures.append(Failure(seed=seed, graph=graph, detail=str(failure)))
            continue
        if not outputs_agree(produced, expected):
            report.failures.append(Failure(seed=seed, graph=graph, detail="disagreed"))
    return report


def fuzz_allocation(count: int = 40, config: GeneratorConfig | None = None) -> FuzzReport:
    """Check that every generated graph gets a valid buffer plan.

    A plan that overlaps two live values does not raise, it produces wrong numbers, so this
    checks the plan directly rather than waiting for the arithmetic to disagree.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    report = FuzzReport()
    for seed in range(count):
        graph = generate(config, seed=seed)
        report.checked += 1
        intervals = compute_intervals(graph, depth_first_order(graph))
        for name, strategy in STRATEGIES.items():
            if not plan_is_valid(intervals, strategy(intervals)):
                report.failures.append(Failure(seed=seed, graph=graph, detail=name))
                break
    return report
