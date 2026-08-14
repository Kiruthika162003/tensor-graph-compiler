from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, PassError
from tgc.ir.graph import Graph, validate

# Running transformations, in an order, until they stop changing anything.
#
# Two things here are worth more than the rest of the file. The first is that every pass is
# validated after it runs, so a transformation that breaks an invariant fails at the pass
# that broke it rather than during code generation four passes later. The cost is a linear
# check per pass and it has never once not been worth it.
#
# The second is the iteration limit. Passes enable each other, so running the pipeline once
# leaves work on the table and running it to a fixed point is the obvious answer. The obvious
# answer does not terminate when two passes undo each other, which happens the first time
# somebody adds a rule that canonicalises in the opposite direction to an existing one. The
# limit turns an infinite loop into an error naming the passes that were still fighting.

Transform = Callable[[Graph], Graph]


@dataclass
class PassResult:
    """What one pass did."""

    name: str
    nodes_before: int
    nodes_after: int
    changed: bool

    @property
    def removed(self) -> int:
        """Nodes the pass eliminated, which may be negative."""
        return self.nodes_before - self.nodes_after

    def as_dict(self) -> dict[str, int | str | bool]:
        """Flat mapping for logging."""
        return {
            "pass": self.name,
            "before": self.nodes_before,
            "after": self.nodes_after,
            "removed": self.removed,
            "changed": self.changed,
        }


@dataclass
class PipelineReport:
    """What a whole run of the pipeline did."""

    results: list[PassResult] = field(default_factory=list)
    rounds: int = 0

    @property
    def changed(self) -> bool:
        """Whether anything happened at all."""
        return any(result.changed for result in self.results)

    @property
    def nodes_removed(self) -> int:
        """Net change in node count across the whole run."""
        if not self.results:
            return 0
        return self.results[0].nodes_before - self.results[-1].nodes_after

    def passes_that_fired(self) -> list[str]:
        """Names of the passes that changed something, without repeats."""
        seen = []
        for result in self.results:
            if result.changed and result.name not in seen:
                seen.append(result.name)
        return seen

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rounds": self.rounds,
            "passes_run": len(self.results),
            "nodes_removed": self.nodes_removed,
            "fired": self.passes_that_fired(),
        }


@dataclass
class Pass:
    """One named transformation."""

    name: str
    transform: Transform
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("a pass needs a name")
        if not callable(self.transform):
            raise ConfigError(f"{self.name} is not callable")

    def run(self, graph: Graph, *, check: bool = True) -> tuple[Graph, PassResult]:
        """Apply the transformation and check what came out."""
        before = len(graph.nodes)
        result = self.transform(graph)
        if not isinstance(result, Graph):
            raise PassError(f"{self.name} returned {type(result).__name__} rather than a graph")
        if check:
            try:
                validate(result)
            except Exception as failure:
                raise PassError(
                    f"{self.name} produced an invalid graph: {failure}"
                ) from failure
        return result, PassResult(
            name=self.name,
            nodes_before=before,
            nodes_after=len(result.nodes),
            changed=not graphs_equal(graph, result),
        )


def graphs_equal(left: Graph, right: Graph) -> bool:
    """Whether two graphs are the same node for node.

    Compared structurally rather than by identity, because a pass that rebuilds every node
    while changing nothing is common and reporting it as a change is what makes a fixed point
    loop run forever.
    """
    if len(left.nodes) != len(right.nodes) or left.outputs != right.outputs:
        return False
    for first, second in zip(left.nodes, right.nodes, strict=True):
        if first.op is not second.op or first.inputs != second.inputs:
            return False
        if first.output != second.output or first.attrs != second.attrs:
            return False
    return True


@dataclass
class Pipeline:
    """An ordered list of passes, run until they settle."""

    passes: list[Pass] = field(default_factory=list)
    max_rounds: int = 8
    check_after_each: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ConfigError(f"the pipeline runs at least once, got {self.max_rounds}")
        names = [item.name for item in self.passes]
        if len(set(names)) != len(names):
            raise ConfigError(f"two passes share a name: {sorted(names)}")

    def enabled_passes(self) -> list[Pass]:
        """The passes that will actually run."""
        return [item for item in self.passes if item.enabled]

    def without(self, *names: str) -> Pipeline:
        """The same pipeline with some passes turned off.

        Which is how every measurement in this compiler is made. A pass is worth what the
        pipeline loses when it is removed, and the only way to know that is to run both.
        """
        unknown = [name for name in names if name not in {item.name for item in self.passes}]
        if unknown:
            raise ConfigError(f"no such pass: {unknown}")
        return Pipeline(
            passes=[
                Pass(name=item.name, transform=item.transform, enabled=item.name not in names)
                for item in self.passes
            ],
            max_rounds=self.max_rounds,
            check_after_each=self.check_after_each,
        )

    def only(self, *names: str) -> Pipeline:
        """The same pipeline with everything else turned off."""
        keep = set(names)
        unknown = [name for name in keep if name not in {item.name for item in self.passes}]
        if unknown:
            raise ConfigError(f"no such pass: {unknown}")
        return Pipeline(
            passes=[
                Pass(name=item.name, transform=item.transform, enabled=item.name in keep)
                for item in self.passes
            ],
            max_rounds=self.max_rounds,
            check_after_each=self.check_after_each,
        )

    def run(self, graph: Graph) -> tuple[Graph, PipelineReport]:
        """Run every pass repeatedly until nothing changes.

        Passes enable each other: folding a constant makes a multiplication by one visible,
        removing that makes its operand dead. Running once leaves all of it on the table.
        """
        validate(graph)
        report = PipelineReport()
        current = graph
        for round_number in range(self.max_rounds):
            report.rounds = round_number + 1
            changed_this_round = False
            for item in self.enabled_passes():
                current, result = item.run(current, check=self.check_after_each)
                report.results.append(result)
                changed_this_round = changed_this_round or result.changed
            if not changed_this_round:
                return current, report
        raise PassError(
            f"the pipeline was still changing the graph after {self.max_rounds} rounds, "
            f"which means two of {[item.name for item in self.enabled_passes()]} "
            "are undoing each other"
        )


def run_once(graph: Graph, transform: Transform, name: str = "pass") -> Graph:
    """Apply a single transformation with the usual checking."""
    result, _ = Pass(name=name, transform=transform).run(graph)
    return result


def fixed_point(graph: Graph, transform: Transform, *, limit: int = 16) -> Graph:
    """Apply one transformation until it stops changing the graph."""
    if limit < 1:
        raise ConfigError(f"the limit must be positive, got {limit}")
    current = graph
    for _ in range(limit):
        nxt = transform(current)
        if graphs_equal(current, nxt):
            return nxt
        current = nxt
    raise PassError(f"the transformation was still changing the graph after {limit} rounds")


def compose(*transforms: Transform) -> Transform:
    """One transformation that runs several in order."""
    if not transforms:
        raise ConfigError("there is nothing to compose")

    def combined(graph: Graph) -> Graph:
        current = graph
        for transform in transforms:
            current = transform(current)
        return current

    return combined


def identity(graph: Graph) -> Graph:
    """A pass that does nothing.

    Useful as a control. A measurement that compares a pipeline against itself with one pass
    swapped for this one is measuring that pass, and a measurement that compares two
    different pipelines is measuring something harder to name.
    """
    return graph


def count_ops(graph: Graph, names: Sequence[str]) -> int:
    """How many nodes of the given kinds a graph holds."""
    wanted = set(names)
    return sum(1 for node in graph.nodes if node.op.name in wanted)
