from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir.builder import elementwise_chain, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.isomorphism import (
    DIGEST_BITS,
    SubgraphReport,
    are_isomorphic,
    attributes_are_part_of_the_identity,
    collision_rate,
    commutativity_is_absorbed,
    dead_code_is_invisible_to_the_hash,
    different_graphs_hash_differently,
    duplication_by_graph,
    hash_is_stable_across_calls,
    layer_shapes_repeat_but_values_do_not,
    node_identities,
    op_pattern_report,
    op_patterns,
    pattern_length_sweep,
    renamed,
    renaming_changes_nothing,
    repeated_subgraphs,
    repeats_are_real_duplicates,
    same_answer_does_not_mean_same_hash,
    same_hash_means_same_answer,
    shapes_are_part_of_the_identity,
    stacked_layers,
    structural_hash,
)


class TestIdentity:
    def test_a_renamed_graph_hashes_the_same(self):
        # Value names come from a counter, so anything that compared them would fail here.
        assert all(row["same_hash"] for row in renaming_changes_nothing())

    def test_every_fixture_gets_its_own_hash(self):
        result = different_graphs_hash_differently()
        assert result["distinct"] == result["graphs"]

    def test_every_value_gets_an_identity(self):
        graph = layernorm_graph()
        identities = node_identities(graph)
        assert len(identities) == len(graph.nodes) + len(graph.inputs)

    def test_hashing_twice_gives_one_answer(self):
        assert hash_is_stable_across_calls()

    def test_a_single_call_has_nothing_to_compare(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            hash_is_stable_across_calls(times=1)

    def test_the_digest_is_sixty_four_bits(self):
        assert len(structural_hash(softmax_graph())) == DIGEST_BITS // 4

    def test_an_empty_prefix_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be empty"):
            renamed(softmax_graph(), prefix="")

    def test_renaming_keeps_the_node_count(self):
        graph = layernorm_graph()
        assert len(renamed(graph).nodes) == len(graph.nodes)


class TestWhatTheHashSees:
    def test_addition_commutes(self):
        assert commutativity_is_absorbed()["addition_commutes"]

    def test_subtraction_does_not(self):
        # Sorting the operands of a subtraction would make two different computations collide.
        assert commutativity_is_absorbed()["subtraction_does_not"]

    def test_the_reduction_axis_is_part_of_the_identity(self):
        assert attributes_are_part_of_the_identity()["different_axes_differ"]

    def test_and_so_is_the_shape(self):
        assert shapes_are_part_of_the_identity()["different_shapes_differ"]

    def test_dead_code_is_not(self):
        # The hash is over the outputs, so a value nobody returns is not part of the identity.
        assert dead_code_is_invisible_to_the_hash()["same_hash"]

    def test_even_though_the_graphs_are_different_sizes(self):
        assert dead_code_is_invisible_to_the_hash()["different_node_counts"]


class TestWhatItPromises:
    def test_one_hash_means_one_answer(self):
        assert same_hash_means_same_answer()["all_agreed"]

    def test_but_one_answer_does_not_mean_one_hash(self):
        # A graph and its optimised form compute the same thing and hash differently, which is
        # what an optimiser is for rather than a defect.
        result = same_answer_does_not_mean_same_hash()
        assert result["same_answer"]
        assert not result["same_hash"]

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            same_hash_means_same_answer(count=0)


class TestCollisions:
    def test_some_generated_graphs_share_a_hash(self):
        assert collision_rate()["repeats"] > 0

    def test_but_none_of_them_is_a_collision(self):
        assert repeats_are_real_duplicates()["collisions"] == 0

    def test_they_are_graphs_that_differ_only_in_unread_work(self):
        assert repeats_are_real_duplicates()["same_once_dead_code_is_removed"] > 0

    def test_most_generated_graphs_are_distinct(self):
        result = collision_rate()
        assert result["distinct_hashes"] > result["graphs"] * 0.8

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            collision_rate(count=0)


class TestRepetition:
    def test_a_branching_graph_repeats_itself(self):
        assert repeated_subgraphs(mlp_graph()).duplicate_nodes == 0
        assert duplication_by_graph()[-1]["duplicate_nodes"] > 0

    def test_a_single_layer_repeats_nothing(self):
        rows = {row["graph"]: row for row in duplication_by_graph()}
        assert rows["softmax"]["duplicate_nodes"] == 0

    def test_an_empty_report_has_nothing_repeated(self):
        assert SubgraphReport().repeated == 0

    def test_it_serialises(self):
        assert repeated_subgraphs(softmax_graph()).as_dict()["distinct"] == 5

    def test_every_fixture_is_reported(self):
        assert len(duplication_by_graph()) == 6


class TestPatterns:
    def test_a_stack_of_layers_repeats_no_values(self):
        # Each layer reads the output of the one before, so nothing computes the same thing.
        assert layer_shapes_repeat_but_values_do_not()["duplicate_values"] == 0

    def test_but_repeats_its_shape_everywhere(self):
        assert layer_shapes_repeat_but_values_do_not()["repeated_patterns"] > 0

    def test_a_longer_pattern_repeats_less(self):
        rows = {row["length"]: row for row in pattern_length_sweep()}
        assert rows[6]["largest_group"] < rows[2]["largest_group"]

    def test_a_pattern_has_to_be_a_chain(self):
        # A window of nodes that do not read each other is not a run worth outlining.
        graph = layernorm_graph()
        assert op_pattern_report(graph, 3)["windows"] <= len(graph.nodes)

    def test_patterns_of_one_are_just_the_ops(self):
        graph = stacked_layers()
        assert len(op_patterns(graph, 1)) == 2

    def test_a_zero_length_pattern_is_refused(self):
        with pytest.raises(ConfigError, match="needs some length"):
            op_patterns(elementwise_chain(4), 0)

    def test_a_stack_needs_at_least_one_layer(self):
        with pytest.raises(ConfigError, match="at least one layer"):
            stacked_layers(0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            pattern_length_sweep(lengths=())

    def test_two_graphs_of_the_same_shape_are_isomorphic(self):
        assert are_isomorphic(elementwise_chain(4), elementwise_chain(4))

    def test_and_two_of_different_depth_are_not(self):
        assert not are_isomorphic(elementwise_chain(4), elementwise_chain(5))
