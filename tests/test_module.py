from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError, GraphError
from tgc.frontend.module import (
    LayerNorm,
    Linear,
    Mlp,
    Module,
    ModuleReport,
    Parameter,
    Sequential,
    a_duplicate_name_is_refused,
    a_repeated_layer_object_still_gets_two_names,
    check_layernorm,
    check_linear,
    check_mlp,
    check_stack,
    compare_modules,
    compile_module,
    epsilon_that_is_not_positive_is_refused,
    graph_size,
    names_are_unique,
    parameter_count,
    parameter_share_of_the_graph,
    random_weights,
    scale_matches_the_convention,
    size_by_depth,
    stack,
    the_normalisation_is_the_expensive_one_to_write,
    what_a_zero_epsilon_would_do,
    where_the_epsilon_goes,
)
from tgc.ir import op as ops
from tgc.verify.reference import run


class TestParameters:
    def test_a_parameter_knows_its_size(self):
        assert Parameter("weight", (8, 16)).elements == 128

    def test_an_unnamed_parameter_is_refused(self):
        with pytest.raises(ConfigError, match="needs a name"):
            Parameter("", (8,))

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="cannot have shape"):
            Parameter("weight", (0, 8))

    def test_it_serialises(self):
        assert Parameter("weight", (8, 16)).as_dict()["elements"] == 128

    def test_a_linear_declares_a_weight_and_a_bias(self):
        assert [item.name for item in Linear(8, 16).parameters()] == ["weight", "bias"]

    def test_and_only_a_weight_without_one(self):
        assert [item.name for item in Linear(8, 16, bias=False).parameters()] == ["weight"]

    def test_an_mlp_names_its_children_by_position(self):
        names = [item.name for item in Mlp(8).parameters()]
        assert names == ["up.weight", "up.bias", "down.weight"]

    def test_a_sequence_prefixes_by_index(self):
        names = [item.name for item in Sequential([LayerNorm(8), LayerNorm(8)]).parameters()]
        assert names[0].startswith("0.")
        assert names[-1].startswith("1.")

    def test_the_parameter_count_adds_up(self):
        assert parameter_count(Linear(8, 16)) == 8 * 16 + 16


class TestCorrectness:
    def test_a_linear_matches_torch_exactly(self):
        assert check_linear()["identical"]

    def test_an_mlp_does_too(self):
        assert check_mlp()["identical"]

    def test_and_a_whole_stack(self):
        assert check_stack()["identical"]

    def test_the_normalisation_matches_a_hand_written_one_exactly(self):
        assert check_layernorm()["graph_against_hand"] == 0.0

    def test_and_the_library_to_a_rounding_unit(self):
        # Three implementations rather than two, so a shared misunderstanding between the graph
        # and its reference shows up as a disagreement with the library.
        assert check_layernorm()["hand_against_library"] < 1e-5


class TestEpsilon:
    def test_putting_it_outside_the_root_is_invisible_on_ordinary_data(self):
        # Which is why the mistake survives review.
        assert where_the_epsilon_goes()["ordinary"]["relative_gap"] < 1e-4

    def test_and_wrong_by_a_factor_on_a_nearly_constant_row(self):
        assert where_the_epsilon_goes()["nearly constant"]["relative_gap"] > 1.0

    def test_a_zero_epsilon_is_refused(self):
        assert epsilon_that_is_not_positive_is_refused()

    def test_because_it_would_divide_zero_by_zero(self):
        result = what_a_zero_epsilon_would_do()
        assert result["with_epsilon"]
        assert not result["without"]

    def test_the_deviation_is_the_population_one(self):
        # Matching torch. The two conventions differ by the root of the width over one less.
        result = scale_matches_the_convention()
        assert result["measured_ratio"] == result["predicted_ratio"]

    def test_a_negative_epsilon_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            LayerNorm(16, epsilon=-1.0)

    def test_a_zero_width_normalisation_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            LayerNorm(0)


class TestNaming:
    def test_a_stack_of_identical_layers_gets_distinct_names(self):
        # Two layers sharing a name produce a graph that builds, runs, and computes a different
        # model with tied weights.
        assert names_are_unique()["all_unique"]

    def test_even_when_the_same_object_is_used_twice(self):
        assert a_repeated_layer_object_still_gets_two_names()["distinct"] == 4

    def test_a_module_that_declares_a_duplicate_is_refused(self):
        assert a_duplicate_name_is_refused()

    def test_the_activation_comes_first_in_the_input_list(self):
        graph = compile_module(Linear(8, 16), [4, 8])
        assert graph.inputs[0].name == "x"

    def test_and_the_parameters_follow_in_declaration_order(self):
        module = Linear(8, 16)
        graph = compile_module(module, [4, 8])
        names = [value.name for value in graph.inputs[1:]]
        assert names == [item.name for item in module.parameters()]

    def test_a_rank_three_input_is_refused(self):
        with pytest.raises(ConfigError, match="takes a matrix"):
            compile_module(Linear(8, 16), [2, 4, 8])

    def test_an_empty_sequence_is_refused(self):
        with pytest.raises(ConfigError, match="needs something in it"):
            Sequential([])

    def test_a_stack_of_no_layers_is_refused(self):
        with pytest.raises(ConfigError, match="at least one layer"):
            stack(0)


class TestShapes:
    def test_a_linear_changes_the_width(self):
        assert Linear(8, 16).output_sizes([4, 8]) == [4, 16]

    def test_a_normalisation_does_not(self):
        assert LayerNorm(8).output_sizes([4, 8]) == [4, 8]

    def test_a_sequence_composes_them(self):
        module = Sequential([Linear(8, 16), Linear(16, 4)])
        assert module.output_sizes([2, 8]) == [2, 4]

    def test_the_compiled_graph_has_that_shape(self):
        graph = compile_module(Linear(8, 16), [4, 8])
        assert [size.value for size in graph.value(graph.outputs[0]).shape.sizes] == [4, 16]

    def test_a_zero_width_linear_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            Linear(0, 16)

    def test_a_zero_expansion_mlp_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be"):
            Mlp(16, expansion=0)


class TestSize:
    def test_the_graph_grows_linearly_with_the_stack(self):
        # A frontend that shared structure between identical layers would grow more slowly, and
        # sharing structure is what the outlining pass is for.
        rows = {row["depth"]: row for row in size_by_depth()}
        assert rows[8]["nodes"] == 8 * rows[1]["nodes"]

    def test_and_so_do_the_parameters(self):
        rows = {row["depth"]: row for row in size_by_depth()}
        assert rows[8]["parameter_elements"] == 8 * rows[1]["parameter_elements"]

    def test_almost_every_input_is_a_weight(self):
        result = parameter_share_of_the_graph()
        assert result["activations"] == 1
        assert result["parameters"] > 10

    def test_a_normalisation_is_the_most_nodes_per_parameter(self):
        ratios = the_normalisation_is_the_expensive_one_to_write()
        assert ratios["layernorm"] > ratios["linear"]
        assert ratios["layernorm"] > ratios["mlp"]

    def test_a_linear_is_three_nodes(self):
        rows = {row["module"]: row for row in compare_modules()}
        assert rows["linear"]["nodes"] == 3

    def test_and_a_normalisation_is_fourteen(self):
        rows = {row["module"]: row for row in compare_modules()}
        assert rows["layernorm"]["nodes"] == 14

    def test_four_modules_are_compared(self):
        assert len(compare_modules()) == 4

    def test_a_report_serialises(self):
        assert ModuleReport(label="linear", nodes=3).as_dict()["nodes"] == 3

    def test_an_empty_depth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            size_by_depth(depths=())

    def test_the_graph_holds_the_products_the_module_declared(self):
        graph = compile_module(Mlp(16), [8, 16])
        assert sum(1 for node in graph.nodes if node.op is ops.MATMUL) == 2

    def test_random_weights_cover_every_parameter(self):
        module = stack(2)
        weights = random_weights(module)
        assert set(weights) == {item.name for item in module.parameters()}

    def test_and_have_the_declared_shapes(self):
        module = Linear(8, 16)
        weights = random_weights(module)
        assert list(weights["weight"].shape) == [8, 16]

    def test_a_graph_size_report_counts_its_inputs(self):
        assert graph_size(depth=1)["inputs"] == 6

    def test_an_unbuilt_module_raises(self):
        with pytest.raises(NotImplementedError):
            Module().parameters()

    def test_and_so_does_building_one(self):
        with pytest.raises(NotImplementedError):
            Module().build(None, "x", {})

    def test_a_bias_free_linear_adds_nothing(self):
        graph = compile_module(Linear(8, 16, bias=False), [4, 8])
        assert not any(node.op is ops.ADD for node in graph.nodes)

    def test_the_weights_really_drive_the_answer(self):
        graph = compile_module(Linear(8, 16, bias=False), [4, 8])
        feeds = {"x": torch.ones(4, 8), "weight": torch.zeros(8, 16)}
        assert float(run(graph, feeds)[0].abs().max()) == 0.0

    def test_a_parameter_called_x_would_collide_with_the_activation(self):
        # Which is the failure the naming scheme has to avoid, so it is worth knowing that the
        # builder catches it rather than quietly keeping the first one.
        class Colliding(Module):
            def parameters(self, _prefix: str = "") -> list[Parameter]:
                return [Parameter("x", (8,))]

            def build(self, builder, source, names):
                return builder.add(source, names["x"])

        with pytest.raises(GraphError, match="already defined"):
            compile_module(Colliding(), [4, 8])
