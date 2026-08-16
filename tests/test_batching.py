from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, mlp_graph
from tgc.passes.batching import (
    BatchReport,
    Candidate,
    a_dependent_chain_is_refused,
    arithmetic_for,
    batch_matmuls,
    batching_gain,
    count_sweep,
    dependent_chain,
    find_candidates,
    find_nothing_to_batch,
    independent,
    is_placeable,
    kernel_counts,
    mismatched_shapes,
    more_products_help_less_and_less,
    parallel_heads,
    rewrite_preserves_the_answer,
    row_sweep,
    size_sweep,
    the_rewrite_adds_nodes_while_removing_kernels,
    traffic_for,
    when_a_group_is_needed,
    when_a_group_is_ready,
    where_batching_stops_paying,
)


class TestMatching:
    def test_products_sharing_a_weight_are_a_group(self):
        assert find_candidates(parallel_heads()).groups == 1

    def test_and_the_group_holds_all_of_them(self):
        assert find_candidates(parallel_heads()).candidates[0].size == 4

    def test_products_that_read_each_other_are_not(self):
        # Every product in the chain shares a weight and has identical shapes, and merging them
        # would change the answer.
        assert a_dependent_chain_is_refused()["groups_found"] == 0

    def test_and_the_dependency_is_what_stops_them(self):
        assert not a_dependent_chain_is_refused()["independent"]

    def test_products_with_different_weights_are_not_a_group(self):
        assert find_candidates(mismatched_shapes()).groups == 0

    def test_a_graph_with_one_product_offers_nothing(self):
        assert find_candidates(mlp_graph()).groups == 0

    def test_every_fixture_is_classified(self):
        assert len(find_nothing_to_batch()) == 5

    def test_independence_is_a_reachability_query(self):
        graph = parallel_heads()
        assert independent(graph, ["x0", "x1"])

    def test_an_empty_report_finds_nothing(self):
        assert BatchReport().groups == 0

    def test_a_candidate_serialises(self):
        candidate = Candidate(nodes=("a", "b"), shared="w", axis=0)
        assert candidate.as_dict()["size"] == 2

    def test_a_symbolic_shape_is_refused(self):
        builder = Builder()
        weight = builder.input([8, 8], name="w")
        first = builder.matmul(builder.input(["n", 8], name="a"), weight)
        second = builder.matmul(builder.input(["n", 8], name="b"), weight)
        graph = builder.finish(builder.add(first, second))
        with pytest.raises(PassError, match="symbolic dimension"):
            find_candidates(graph)


class TestPlacement:
    def test_a_group_of_graph_inputs_is_ready_immediately(self):
        graph = parallel_heads()
        candidate = find_candidates(graph).candidates[0]
        assert when_a_group_is_ready(graph, candidate) == 0

    def test_and_is_needed_after_the_first_product(self):
        graph = parallel_heads()
        candidate = find_candidates(graph).candidates[0]
        assert when_a_group_is_needed(graph, candidate) > 0

    def test_so_it_can_be_placed(self):
        graph = parallel_heads()
        assert is_placeable(graph, find_candidates(graph).candidates[0])

    def test_the_joined_product_lands_before_its_first_consumer(self):
        rewritten = batch_matmuls(parallel_heads())
        product = next(
            index for index, node in enumerate(rewritten.nodes) if node.op is ops.MATMUL
        )
        first_slice = next(
            index for index, node in enumerate(rewritten.nodes) if node.op is ops.SLICE
        )
        assert product < first_slice


class TestRewrite:
    def test_four_products_become_one(self):
        result = kernel_counts()
        assert result["before"] == 4
        assert result["after"] == 1

    def test_and_the_answer_survives(self):
        assert rewrite_preserves_the_answer()["relative_gap"] < 1e-6

    def test_though_not_bit_for_bit(self):
        # Joining four products of eight rows changes the shape the library sees, and the
        # library picks its blocking from the shape.
        assert not rewrite_preserves_the_answer()["identical"]

    def test_the_rewrite_adds_nodes_while_removing_kernels(self):
        result = the_rewrite_adds_nodes_while_removing_kernels()
        assert result["kernels_removed"] == 3
        assert result["nodes_added"] > 0

    def test_a_graph_with_nothing_to_batch_is_returned_unchanged(self):
        graph = mlp_graph()
        assert batch_matmuls(graph) is graph

    def test_a_dependent_chain_is_returned_unchanged(self):
        graph = dependent_chain()
        assert batch_matmuls(graph) is graph

    def test_the_joins_and_windows_appear(self):
        rewritten = batch_matmuls(parallel_heads())
        assert sum(1 for node in rewritten.nodes if node.op is ops.CONCAT) == 3
        assert sum(1 for node in rewritten.nodes if node.op is ops.SLICE) == 4

    def test_a_single_branch_fixture_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to batch"):
            parallel_heads(heads=1)

    def test_a_chain_of_one_is_refused(self):
        with pytest.raises(ConfigError, match="at least two products"):
            dependent_chain(length=1)


class TestCost:
    def test_joining_reads_the_weight_once(self):
        separate = traffic_for(8, 32, 32, 4, batched=False)
        joined = traffic_for(8, 32, 32, 4, batched=True)
        assert separate - joined == 32 * 32 * 4 * 3

    def test_the_arithmetic_does_not_change(self):
        assert arithmetic_for(8, 32, 32, 4) == 4 * arithmetic_for(8, 32, 32, 1)

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            traffic_for(0, 32, 32, 4, batched=True)

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            arithmetic_for(8, 32, 32, 0)

    def test_a_thin_product_gains_a_lot(self):
        assert batching_gain(rows=8, inner=256, columns=256)["speedup"] > 3.0

    def test_a_tall_one_gains_nothing(self):
        # The arithmetic is unchanged by joining them and it is what the time is.
        assert batching_gain(rows=1024, inner=256, columns=256)["speedup"] == 1.0

    def test_and_says_which_regime_it_is_in(self):
        assert batching_gain(rows=1024, inner=256, columns=256)["compute_bound"]
        assert not batching_gain(rows=8, inner=256, columns=256)["compute_bound"]


class TestSweeps:
    def test_the_gain_climbs_with_the_contraction_rather_than_falling(self):
        # Which is not what the usual argument for batching predicts. A thin product is memory
        # bound at every size, so a larger weight means more traffic to remove.
        rows = size_sweep()
        assert rows[-1]["speedup"] > rows[0]["speedup"]

    def test_the_row_count_is_what_stops_it(self):
        rows = row_sweep()
        assert rows[0]["speedup"] > 3.0
        assert rows[-1]["speedup"] == 1.0

    def test_the_crossover_is_findable(self):
        assert where_batching_stops_paying() == 64

    def test_joining_more_products_helps_less_each_time(self):
        assert more_products_help_less_and_less()["flattening"]

    def test_but_never_hurts(self):
        speedups = [row["speedup"] for row in count_sweep()]
        assert speedups == sorted(speedups)

    def test_an_empty_size_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            size_sweep(sizes=())

    def test_an_empty_row_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            row_sweep(rows=())

    def test_an_empty_count_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            count_sweep(counts=())
