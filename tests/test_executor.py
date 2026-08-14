from __future__ import annotations

import pytest
import torch

from tgc.codegen.emit import (
    EmittedModule,
    arena_elements,
    can_lower,
    check_dtypes_uniform,
    dtype_of,
    emit,
    operation_names,
    unsupported_operations,
    view_expression,
)
from tgc.errors import CodegenError, ConfigError
from tgc.ir.builder import (
    Builder,
    branching_graph,
    diamond_graph,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT16
from tgc.memory.planner import Plan
from tgc.passes.layout import transposed_pair_graph
from tgc.runtime.executor import (
    arena_against_interpreter,
    compile_graph,
    compiled_matches_reference,
    default_pipeline,
    optimisation_effect,
    run_many,
    source_of,
)
from tgc.verify.reference import random_feeds, run

GRAPHS = {
    "chain": elementwise_chain(8, sizes=(8, 8)),
    "softmax": softmax_graph(),
    "layernorm": layernorm_graph(),
    "mlp": mlp_graph(),
    "diamond": diamond_graph(sizes=(8, 8)),
    "branching": branching_graph(4, 2, width=8),
}


class TestLowering:
    def test_every_fixture_can_be_lowered(self):
        assert all(can_lower(graph) for graph in GRAPHS.values())

    def test_a_symbolic_graph_cannot(self):
        assert not can_lower(mlp_graph(batch="batch"))

    def test_and_is_refused_rather_than_generating_broken_code(self):
        with pytest.raises(CodegenError, match="cannot lower"):
            compile_graph(mlp_graph(batch="batch"))

    def test_the_backend_covers_every_operation_the_fixtures_use(self):
        for graph in GRAPHS.values():
            assert unsupported_operations(graph) == []

    def test_it_lists_the_operations_a_graph_uses(self):
        assert "exp" in operation_names(softmax_graph())

    def test_a_graph_of_mixed_widths_is_refused(self):
        # The arena is one typed buffer, so mixing widths needs a reinterpret and a byte
        # offset. Worth doing and not worth doing quietly.
        builder = Builder()
        x = builder.input([4, 4], name="x")
        graph = builder.finish(builder.cast(x, FLOAT16))
        with pytest.raises(CodegenError, match="one element width"):
            check_dtypes_uniform(graph)

    def test_a_uniform_graph_names_its_type(self):
        assert dtype_of(softmax_graph()) == "float32"


class TestGeneration:
    def test_it_produces_a_function(self):
        assert "def compiled(arena, inputs):" in source_of(softmax_graph())

    def test_every_value_gets_a_view_into_the_arena(self):
        source = source_of(softmax_graph())
        assert source.count("arena[") == len(softmax_graph().value_names)

    def test_the_inputs_are_copied_rather_than_aliased(self):
        # Aliasing would mean the caller's tensors are overwritten by any pass that decided
        # an input's storage could be reused, which this compiler is allowed to decide.
        assert "copy_(inputs['x'])" in source_of(softmax_graph())

    def test_the_outputs_are_returned(self):
        graph = softmax_graph()
        assert f"return [buf_{graph.outputs[0]}]" in source_of(graph)

    def test_a_plan_that_places_nothing_is_refused(self):
        graph = softmax_graph()
        with pytest.raises(CodegenError, match="does not place"):
            emit(graph, graph.nodes, Plan())

    def test_a_view_slices_by_elements(self):
        assert "arena[16:32]" in view_expression("b", 16, (4, 4), "float32")

    def test_the_arena_holds_the_planned_bytes(self):
        module = EmittedModule(source="", arena_bytes=64)
        assert arena_elements(module, element_bytes=4) == 16

    def test_a_zero_byte_element_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one byte"):
            arena_elements(EmittedModule(source="", arena_bytes=64), element_bytes=0)

    def test_the_generated_source_is_short(self):
        # Roughly two lines per value plus the header, which is what makes it readable when
        # something has gone wrong.
        graph = softmax_graph()
        assert compile_graph(graph).module.lines < 4 * len(graph.value_names)

    def test_it_serialises(self):
        assert compile_graph(softmax_graph()).module.as_dict()["outputs"]


class TestExecution:
    def test_every_fixture_matches_the_reference_bit_for_bit(self):
        # The default pipeline holds only transformations that preserve the answer exactly,
        # so anything less means one of them is wrong.
        for name, graph in GRAPHS.items():
            assert compiled_matches_reference(graph)["identical"], name

    def test_it_matches_without_the_pipeline_too(self):
        for name, graph in GRAPHS.items():
            assert compiled_matches_reference(graph, optimise=False)["identical"], name

    def test_the_arena_survives_being_reused_across_calls(self):
        # A kernel that reads a buffer before writing it passes on the first call and fails
        # on the second.
        for name, graph in GRAPHS.items():
            assert run_many(graph), name

    def test_a_transpose_pair_compiles_and_matches(self):
        assert compiled_matches_reference(transposed_pair_graph(width=8))["identical"]

    def test_a_missing_input_is_reported(self):
        compiled = compile_graph(softmax_graph())
        with pytest.raises(ConfigError, match="no value supplied"):
            compiled({})

    def test_running_no_seeds_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to run"):
            run_many(softmax_graph(), seeds=[])

    def test_the_compiled_output_has_the_right_shape(self):
        graph = mlp_graph()
        feeds = random_feeds(graph)
        assert compile_graph(graph)(feeds)[0].shape == run(graph, feeds)[0].shape

    def test_a_fresh_arena_is_zeroed(self):
        compiled = compile_graph(softmax_graph())
        assert torch.equal(compiled.new_arena(), torch.zeros_like(compiled.new_arena()))


class TestArena:
    def test_the_compiled_program_allocates_once(self):
        # Against an interpreter that needs a tensor per node and hands them all to whatever
        # allocator sits underneath.
        result = arena_against_interpreter(elementwise_chain(8))
        assert result["ratio"] > 4.0

    def test_a_wide_graph_saves_more(self):
        wide = arena_against_interpreter(branching_graph(6, 2))["ratio"]
        narrow = arena_against_interpreter(softmax_graph())["ratio"]
        assert wide > narrow

    def test_the_arena_is_the_size_the_planner_promised(self):
        compiled = compile_graph(elementwise_chain(8))
        assert compiled.arena_bytes == compiled.plan.arena_bytes

    def test_and_never_less_than_the_liveness_floor(self):
        for graph in GRAPHS.values():
            report = compile_graph(graph).as_dict()
            assert report["arena_bytes"] >= report["peak_bytes"]

    def test_it_serialises(self):
        assert compile_graph(softmax_graph()).as_dict()["nodes"] == 5


class TestPipeline:
    def test_the_default_pipeline_holds_only_exact_passes(self):
        assert {item.name for item in default_pipeline().passes} == {
            "constant folding",
            "algebraic",
            "transposes",
            "subexpressions",
            "dead code",
        }

    def test_optimising_removes_nodes_from_a_redundant_graph(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        y = builder.input([8, 8], name="y")
        first = builder.mul(x, y)
        second = builder.mul(x, y)
        graph = builder.finish(builder.add(first, second))
        result = optimisation_effect(graph)
        assert result["nodes_optimised"] < result["nodes_plain"]

    def test_and_shortens_the_generated_code(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        y = builder.input([8, 8], name="y")
        graph = builder.finish(builder.add(builder.mul(x, y), builder.mul(x, y)))
        result = optimisation_effect(graph)
        assert result["lines_optimised"] < result["lines_plain"]

    def test_and_does_not_change_the_answer(self):
        builder = Builder()
        x = builder.input([8, 8], name="x")
        y = builder.input([8, 8], name="y")
        graph = builder.finish(builder.add(builder.mul(x, y), builder.mul(x, y)))
        feeds = random_feeds(graph)
        assert torch.equal(compile_graph(graph)(feeds)[0], run(graph, feeds)[0])

    def test_an_already_minimal_graph_is_unchanged(self):
        result = optimisation_effect(softmax_graph())
        assert result["nodes_optimised"] == result["nodes_plain"]
