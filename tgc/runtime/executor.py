from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from tgc.analysis.liveness import compute_intervals, peak_bytes
from tgc.codegen.emit import EmittedModule, arena_elements, can_lower, dtype_of, emit
from tgc.errors import CodegenError, ConfigError
from tgc.ir.graph import Graph, Node, validate
from tgc.memory.planner import Plan, plan_largest_first, validate_plan
from tgc.passes.algebraic import simplify
from tgc.passes.constfold import fold_constants
from tgc.passes.cse import eliminate_common_subexpressions
from tgc.passes.dce import eliminate_dead_code
from tgc.passes.layout import cancel_transposes
from tgc.passes.manager import Pass, Pipeline
from tgc.schedule.order import depth_first_order
from tgc.verify.reference import outputs_agree, random_feeds, run, to_torch

# Compiling a graph and running it.
#
# The whole pipeline in one place: optimise, schedule, allocate, generate, execute. Nothing
# here is clever; the value is that it runs end to end, so every claim the earlier files make
# about bytes and peaks is attached to a program that produces the right numbers.
#
# Generated code is executed with exec. That is a deliberate choice and a bounded one: the
# source is built from the graph by this compiler and never from anything a caller supplies,
# and the alternative is an interpreter, which is the thing being compared against.


def default_pipeline() -> Pipeline:
    """The passes a caller gets without asking for anything.

    Exact ones only. Every transformation here either preserves the answer bit for bit or is
    a pure removal of work nobody reads, so a compiled graph and the interpreter have to
    agree exactly, and the tests demand it.
    """
    return Pipeline(
        passes=[
            Pass(name="constant folding", transform=fold_constants),
            Pass(name="algebraic", transform=simplify),
            Pass(name="transposes", transform=cancel_transposes),
            Pass(name="subexpressions", transform=eliminate_common_subexpressions),
            Pass(name="dead code", transform=eliminate_dead_code),
        ]
    )


@dataclass
class CompiledGraph:
    """Everything produced by compiling one graph."""

    graph: Graph
    order: list[Node]
    plan: Plan
    module: EmittedModule
    function: Callable
    input_names: list[str] = field(default_factory=list)

    @property
    def arena_bytes(self) -> int:
        """Storage the compiled program needs."""
        return self.plan.arena_bytes

    def new_arena(self) -> torch.Tensor:
        """A fresh buffer of the right size and type."""
        return torch.zeros(
            arena_elements(self.module),
            dtype=to_torch(self.graph.value(self.input_names[0]).dtype),
        )

    def __call__(self, feeds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Run the compiled program on a set of inputs."""
        missing = [name for name in self.input_names if name not in feeds]
        if missing:
            raise ConfigError(f"no value supplied for {missing}")
        return self.function(self.new_arena(), feeds)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "nodes": len(self.graph.nodes),
            "arena_bytes": self.arena_bytes,
            "generated_lines": self.module.lines,
            "peak_bytes": peak_bytes(compute_intervals(self.graph, self.order)),
        }


def compile_graph(
    graph: Graph, *, pipeline: Pipeline | None = None, optimise: bool = True
) -> CompiledGraph:
    """Optimise, schedule, allocate, and generate code for a graph."""
    validate(graph)
    if not can_lower(graph):
        raise CodegenError("this graph has operations or shapes the backend cannot lower")
    dtype_of(graph)

    optimised = graph
    if optimise:
        chosen = pipeline or default_pipeline()
        optimised, _ = chosen.run(graph)

    order = depth_first_order(optimised)
    intervals = compute_intervals(optimised, order)
    plan = plan_largest_first(intervals)
    validate_plan(intervals, plan)

    module = emit(optimised, order, plan)
    namespace: dict = {}
    exec(compile(module.source, "<tgc>", "exec"), namespace)
    function = namespace["compiled"]

    return CompiledGraph(
        graph=optimised,
        order=list(order),
        plan=plan,
        module=module,
        function=function,
        input_names=[value.name for value in optimised.inputs],
    )


def compiled_matches_reference(
    graph: Graph, feeds: dict[str, torch.Tensor] | None = None, *, optimise: bool = True
) -> dict:
    """Run a graph both ways and report whether they agree.

    Bit equality is the bar. The default pipeline holds only transformations that preserve
    the answer exactly, so anything less means one of them is wrong rather than that a
    tolerance needs widening.
    """
    supplied = feeds if feeds is not None else random_feeds(graph, positive=True)
    compiled = compile_graph(graph, optimise=optimise)
    expected = run(graph, supplied)
    produced = compiled(supplied)
    return {
        "identical": outputs_agree(produced, expected),
        "arena_bytes": compiled.arena_bytes,
        "nodes_before": len(graph.nodes),
        "nodes_after": len(compiled.graph.nodes),
        "generated_lines": compiled.module.lines,
    }


def arena_against_interpreter(graph: Graph) -> dict:
    """What the compiled program allocates against what an interpreter would.

    The interpreter needs a tensor per node and gives them all back to whatever allocator sits
    underneath. The compiled program takes one arena and never asks again, and the size of it
    is the number the planner promised rather than whatever the runtime happened to do.
    """
    compiled = compile_graph(graph)
    interpreter_bytes = sum(graph.value(name).bytes for name in graph.value_names)
    return {
        "compiled_arena": compiled.arena_bytes,
        "interpreter_total": interpreter_bytes,
        "ratio": round(interpreter_bytes / compiled.arena_bytes, 4)
        if compiled.arena_bytes
        else 0.0,
    }


def optimisation_effect(graph: Graph) -> dict:
    """The same graph compiled with the pipeline and without it."""
    optimised = compile_graph(graph, optimise=True)
    plain = compile_graph(graph, optimise=False)
    return {
        "nodes_optimised": len(optimised.graph.nodes),
        "nodes_plain": len(plain.graph.nodes),
        "arena_optimised": optimised.arena_bytes,
        "arena_plain": plain.arena_bytes,
        "lines_optimised": optimised.module.lines,
        "lines_plain": plain.module.lines,
    }


def run_many(graph: Graph, seeds: Sequence[int] = range(8)) -> bool:
    """Compile once and run on several inputs, checking each against the interpreter.

    Compiling once and feeding many is the case the arena has to survive: it is reused across
    calls, so a kernel that reads a buffer before writing it passes on the first call and
    fails on the second.
    """
    if not list(seeds):
        raise ConfigError("there is nothing to run")
    compiled = compile_graph(graph)
    for seed in seeds:
        feeds = random_feeds(graph, seed=seed, positive=True)
        if not outputs_agree(compiled(feeds), run(graph, feeds)):
            return False
    return True


def source_of(graph: Graph) -> str:
    """The generated source for a graph, for reading and for tests."""
    return compile_graph(graph).module.source
