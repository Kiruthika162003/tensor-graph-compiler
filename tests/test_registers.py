from __future__ import annotations

import itertools

import pytest

from tgc.codegen.registers import (
    Pressure,
    SpillReport,
    applying_the_rule_to_a_shared_value_is_refused,
    balanced_tree,
    balanced_tree_needs_its_depth,
    best_pressure,
    chain_needs_two_registers,
    elementwise_only,
    fusion_groups_that_are_not_trees,
    how_often_the_rule_helps,
    is_a_tree,
    lighter_first_order,
    lopsided_tree,
    measure_pressure,
    naive_pressure,
    ordering_saves_registers,
    random_tree,
    register_file_sweep,
    register_need,
    sethi_ullman_order,
    shared_values,
    skewed_tree,
    smaller_tiles_avoid_spilling,
    source_order,
    spill_report,
    the_rule_pays_on_a_lopsided_tree,
)
from tgc.errors import CodegenError, ConfigError
from tgc.ir.builder import elementwise_chain, layernorm_graph, softmax_graph


class TestTrees:
    def test_a_chain_is_a_tree(self):
        assert is_a_tree(elementwise_chain(8))

    def test_a_softmax_is_not(self):
        # It reads its exponential twice.
        assert not is_a_tree(softmax_graph())

    def test_and_the_shared_values_are_named(self):
        assert len(shared_values(softmax_graph())) == 2

    def test_a_layernorm_shares_two_values_as_well(self):
        assert len(shared_values(layernorm_graph())) == 2

    def test_the_rule_refuses_a_graph_it_cannot_handle(self):
        # Rather than guessing. Minimising registers with sharing is a harder problem.
        assert applying_the_rule_to_a_shared_value_is_refused()

    def test_the_naive_order_refuses_it_too(self):
        with pytest.raises(CodegenError, match="not a tree"):
            lighter_first_order(softmax_graph())

    def test_which_fixtures_are_trees_is_reported(self):
        rows = {row["graph"]: row for row in fusion_groups_that_are_not_trees()}
        assert rows["chain"]["is_a_tree"]
        assert not rows["layernorm"]["is_a_tree"]


class TestNeeds:
    def test_a_leaf_needs_one_register(self):
        graph = elementwise_chain(4)
        assert register_need(graph, graph.inputs[0].name) == 1

    def test_a_chain_needs_one_whatever_its_length(self):
        graph = elementwise_chain(16)
        assert register_need(graph, graph.outputs[0]) == 1

    def test_a_balanced_join_needs_one_more_than_either_side(self):
        # Because after the first is computed and held, the second has the same requirement
        # with one fewer register available.
        graph = balanced_tree(2)
        assert register_need(graph, graph.outputs[0]) == 3

    def test_a_skewed_tree_stops_at_two_whatever_its_depth(self):
        # Only its first join has two sides of equal weight. Every join after that has a
        # subtree on one side and a leaf on the other, so the rule never adds another.
        assert register_need(skewed_tree(8), "v7") == register_need(skewed_tree(2), "v1")


class TestPressure:
    def test_a_chain_holds_two_values_at_any_length(self):
        # Which is why fusing a chain has no natural stopping point.
        assert all(row["peak"] == 2 for row in chain_needs_two_registers())

    def test_a_balanced_tree_grows_with_its_depth(self):
        rows = balanced_tree_needs_its_depth()
        assert rows[-1]["peak"] > rows[0]["peak"]

    def test_by_exactly_one_per_level(self):
        rows = [row["peak"] for row in balanced_tree_needs_its_depth()]
        assert all(later - earlier == 1 for earlier, later in itertools.pairwise(rows))

    def test_the_order_covers_every_node(self):
        graph = balanced_tree(3)
        assert len(sethi_ullman_order(graph)) == len(graph.nodes)

    def test_the_source_order_is_the_node_list(self):
        graph = balanced_tree(3)
        assert source_order(graph) == [node.name for node in graph.nodes]

    def test_the_pressure_is_recorded_at_every_step(self):
        graph = balanced_tree(3)
        pressure = best_pressure(graph)
        assert len(pressure.at_each_step) == pressure.steps

    def test_an_empty_order_holds_nothing(self):
        assert measure_pressure(balanced_tree(2), []).peak == 0

    def test_it_serialises(self):
        assert best_pressure(balanced_tree(2)).as_dict()["peak"] == 4

    def test_a_machine_with_no_registers_is_refused(self):
        with pytest.raises(ConfigError, match="needs registers"):
            Pressure(order=(), peak=4).spills(0)


class TestOrdering:
    def test_the_rule_buys_nothing_on_a_balanced_tree(self):
        # Both subtrees need the same number, so the two orders are the same order.
        assert ordering_saves_registers()["saving"] == 0

    def test_and_saves_a_register_on_a_lopsided_one(self):
        assert the_rule_pays_on_a_lopsided_tree()["saving"] == 1

    def test_it_wins_on_three_quarters_of_random_trees(self):
        result = how_often_the_rule_helps()
        assert result["rule_wins"] > result["trees"] * 0.7

    def test_and_never_loses(self):
        assert how_often_the_rule_helps()["rule_loses"] == 0

    def test_by_more_than_a_register_on_average(self):
        assert how_often_the_rule_helps()["mean_saving"] > 1.0

    def test_a_random_tree_is_a_tree(self):
        assert is_a_tree(random_tree(12))

    def test_a_zero_node_tree_is_refused(self):
        with pytest.raises(ConfigError, match="needs some nodes"):
            random_tree(0)

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            how_often_the_rule_helps(count=0)

    def test_a_lopsided_tree_needs_both_sides(self):
        with pytest.raises(ConfigError, match="light side needs some depth"):
            lopsided_tree(heavy=3, light=0)

    def test_a_zero_depth_tree_is_refused(self):
        with pytest.raises(ConfigError, match="needs some depth"):
            balanced_tree(0)

    def test_a_zero_depth_skew_is_refused(self):
        with pytest.raises(ConfigError, match="needs some depth"):
            skewed_tree(0)


class TestSpilling:
    def test_a_small_register_file_spills(self):
        assert not spill_report(balanced_tree(5), registers=2).fits

    def test_a_large_enough_one_does_not(self):
        assert spill_report(balanced_tree(5), registers=16).fits

    def test_spilling_stops_once_the_file_holds_the_peak(self):
        rows = register_file_sweep()
        fitting = [row for row in rows if row["fits"]]
        assert all(row["registers"] >= rows[0]["peak"] for row in fitting)

    def test_the_traffic_is_linear_in_the_tile(self):
        # Spilling is not a fixed penalty, it is proportional to everything the kernel holds.
        rows = smaller_tiles_avoid_spilling()
        assert rows[-1]["extra_traffic"] == rows[0]["extra_traffic"] * 64

    def test_a_spilled_value_moves_twice(self):
        report = spill_report(balanced_tree(4), registers=1, elements=100)
        assert report.extra_traffic == report.spills * 100 * 2 * 4

    def test_an_empty_report_fits(self):
        assert SpillReport(registers=8, peak=2, elements=64).fits

    def test_it_serialises(self):
        assert spill_report(balanced_tree(3)).as_dict()["fits"]

    def test_zero_registers_are_refused(self):
        with pytest.raises(ConfigError, match="needs registers"):
            spill_report(balanced_tree(3), registers=0)

    def test_an_empty_tile_is_refused(self):
        with pytest.raises(ConfigError, match="has to hold something"):
            spill_report(balanced_tree(3), elements=0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            register_file_sweep(sizes=())

    def test_an_empty_tile_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            smaller_tiles_avoid_spilling(tiles=())

    def test_an_empty_length_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            chain_needs_two_registers(lengths=())

    def test_an_empty_depth_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            balanced_tree_needs_its_depth(depths=())


class TestFiltering:
    def test_only_elementwise_nodes_reach_a_register_allocator(self):
        # Anything else reaches memory anyway, so it is not part of the register problem.
        assert len(elementwise_only(softmax_graph())) < len(softmax_graph().nodes)

    def test_a_chain_is_entirely_elementwise(self):
        graph = elementwise_chain(6)
        assert len(elementwise_only(graph)) == len(graph.nodes)

    def test_the_naive_order_is_still_a_valid_order(self):
        graph = lopsided_tree(3, 2)
        assert len(lighter_first_order(graph)) == len(graph.nodes)

    def test_and_measures_a_higher_peak(self):
        graph = lopsided_tree(4, 1)
        assert naive_pressure(graph).peak >= best_pressure(graph).peak
