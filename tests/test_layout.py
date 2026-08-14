from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir.builder import Builder, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.graph import validate
from tgc.passes.dce import eliminate_dead_code
from tgc.passes.layout import (
    COLUMN_MAJOR,
    ROW_MAJOR,
    Layout,
    TransposeReport,
    best_global_layout,
    boundary_cost,
    cancel_transposes,
    check_permutation_algebra,
    compare_layout_policies,
    compose_permutations,
    count_transposes,
    global_layout_cost,
    inverse_permutation,
    is_contiguous_under,
    is_identity,
    layout_conflicts,
    preferred_layout,
    report_transposes,
    strides_for,
    transpose_bytes,
    transposed_chain_graph,
    transposed_pair_graph,
)
from tgc.verify.reference import outputs_agree, random_feeds, run


class TestLayout:
    def test_row_major_varies_the_last_dimension_fastest(self):
        assert Layout(ROW_MAJOR).is_row_major

    def test_column_major_does_not(self):
        assert not Layout(COLUMN_MAJOR).is_row_major

    def test_row_major_needs_no_permutation(self):
        assert Layout(ROW_MAJOR).permutation_for(3) == (0, 1, 2)

    def test_column_major_reverses(self):
        assert Layout(COLUMN_MAJOR).permutation_for(3) == (2, 1, 0)

    def test_an_unknown_layout_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown layout"):
            Layout("diagonal")

    def test_a_negative_rank_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot be negative"):
            Layout().permutation_for(-1)

    def test_it_prints_as_its_name(self):
        assert str(Layout(ROW_MAJOR)) == ROW_MAJOR


class TestPermutations:
    def test_a_permutation_composed_with_its_inverse_is_the_identity(self):
        # The pass rests entirely on this. A compiler that gets it wrong produces transposed
        # tensors that pass every shape assertion and hold the wrong numbers.
        result = check_permutation_algebra(4)
        assert result["composes_to_identity"]

    def test_a_full_reversal_is_its_own_inverse(self):
        assert check_permutation_algebra(4)["self_inverse"]

    def test_the_inverse_undoes_the_permutation(self):
        permutation = (2, 0, 1)
        assert is_identity(compose_permutations(permutation, inverse_permutation(permutation)))

    def test_something_that_is_not_a_permutation_is_rejected(self):
        with pytest.raises(ConfigError, match="not a permutation"):
            inverse_permutation([0, 0, 1])

    def test_permutations_of_different_ranks_do_not_compose(self):
        with pytest.raises(ConfigError, match="different ranks"):
            compose_permutations([0, 1], [0, 1, 2])

    def test_the_identity_is_the_identity(self):
        assert is_identity([0, 1, 2])
        assert not is_identity([1, 0, 2])

    def test_a_rank_zero_permutation_has_nothing_to_check(self):
        with pytest.raises(PassError, match="nothing to check"):
            check_permutation_algebra(0)


class TestCancellation:
    def test_two_opposite_transposes_cancel(self):
        graph = transposed_pair_graph()
        assert count_transposes(eliminate_dead_code(cancel_transposes(graph))) == 0

    def test_three_that_compose_to_the_identity_cancel_too(self):
        graph = transposed_chain_graph()
        assert count_transposes(eliminate_dead_code(cancel_transposes(graph))) == 0

    def test_the_answer_is_bit_identical(self):
        # Composing permutations moves no data and changes no arithmetic, so this is exact
        # rather than merely close.
        for graph in (transposed_pair_graph(), transposed_chain_graph()):
            feeds = random_feeds(graph)
            cancelled = eliminate_dead_code(cancel_transposes(graph))
            assert outputs_agree(run(graph, feeds), run(cancelled, feeds))

    def test_a_lone_transpose_survives(self):
        builder = Builder()
        x = builder.input([4, 8], name="x")
        graph = builder.finish(builder.transpose(x, [1, 0]))
        assert count_transposes(cancel_transposes(graph)) == 1

    def test_the_result_still_validates(self):
        for graph in (transposed_pair_graph(), transposed_chain_graph()):
            validate(cancel_transposes(graph))

    def test_a_graph_with_no_transposes_is_left_alone(self):
        graph = softmax_graph()
        assert len(cancel_transposes(graph).nodes) == len(graph.nodes)

    def test_the_transposes_move_data_before_cancelling(self):
        assert transpose_bytes(transposed_pair_graph()) > 0

    def test_and_none_afterwards(self):
        graph = eliminate_dead_code(cancel_transposes(transposed_pair_graph()))
        assert transpose_bytes(graph) == 0

    def test_the_report_counts_what_is_left(self):
        assert report_transposes(transposed_pair_graph()).remaining == 1

    def test_it_serialises(self):
        assert TransposeReport(cancelled=["a"], merged=["b"]).as_dict()["removed"] == 2


class TestPreferences:
    def test_an_elementwise_op_has_no_preference(self):
        # Which is what makes the question tractable at all: a chain of them can take
        # whichever layout its neighbours wanted.
        graph = softmax_graph()
        assert preferred_layout(graph.node("v1")) == ""

    def test_a_matmul_wants_the_column_order(self):
        graph = mlp_graph()
        matmul = next(node for node in graph.nodes if node.op.name == "matmul")
        assert preferred_layout(matmul) == COLUMN_MAJOR

    def test_a_reduction_wants_the_row_order(self):
        graph = softmax_graph()
        assert preferred_layout(graph.node("v0")) == ROW_MAJOR

    def test_a_matmul_graph_prefers_column_major_overall(self):
        assert best_global_layout(mlp_graph()) == COLUMN_MAJOR

    def test_a_reduction_graph_prefers_row_major(self):
        assert best_global_layout(softmax_graph()) == ROW_MAJOR

    def test_the_conflicting_nodes_are_named(self):
        assert len(layout_conflicts(mlp_graph(), ROW_MAJOR)) == 2

    def test_the_preferred_layout_has_no_conflicts(self):
        assert layout_conflicts(mlp_graph(), COLUMN_MAJOR) == []

    def test_an_unknown_layout_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown layout"):
            layout_conflicts(mlp_graph(), "diagonal")


class TestPolicies:
    def test_a_global_layout_pays_for_every_node_that_wanted_the_other(self):
        assert global_layout_cost(mlp_graph(), ROW_MAJOR) > 0

    def test_the_per_node_policy_pays_only_at_the_boundaries(self):
        # Counting satisfied preferences instead reports zero on every graph, which is true
        # and useless.
        assert boundary_cost(mlp_graph()) < global_layout_cost(mlp_graph(), ROW_MAJOR)

    def test_the_elementwise_chains_keep_the_reductions_apart(self):
        for graph in (softmax_graph(), layernorm_graph(), mlp_graph()):
            assert boundary_cost(graph) == 0

    def test_every_policy_is_reported(self):
        assert len(compare_layout_policies(mlp_graph())) == 3

    def test_the_per_node_row_comes_last(self):
        assert compare_layout_policies(mlp_graph())[-1]["policy"] == "per node"


class TestStrides:
    def test_a_packed_tensor_strides_by_the_trailing_sizes(self):
        graph = mlp_graph()
        assert strides_for(graph, "x", Layout(ROW_MAJOR)) == (64, 1)

    def test_the_other_layout_reverses_them(self):
        graph = mlp_graph()
        assert strides_for(graph, "x", Layout(COLUMN_MAJOR)) == (1, 64)

    def test_a_row_major_tensor_is_contiguous_in_row_major(self):
        assert is_contiguous_under(mlp_graph(), "x", Layout(ROW_MAJOR))

    def test_and_not_in_the_other(self):
        assert not is_contiguous_under(mlp_graph(), "x", Layout(COLUMN_MAJOR))

    def test_a_vector_is_contiguous_either_way(self):
        graph = mlp_graph()
        assert is_contiguous_under(graph, "b_up", Layout(COLUMN_MAJOR))
