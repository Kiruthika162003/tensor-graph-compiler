from __future__ import annotations

import pytest

from tgc.errors import ConfigError, ScheduleError
from tgc.ir.builder import Builder, branching_graph, elementwise_chain, mlp_graph, softmax_graph
from tgc.parallel.partition import (
    STRATEGIES,
    Assignment,
    PartitionReport,
    best_strategy,
    compare_strategies,
    cut_edges_of,
    cut_fraction,
    device_sweep,
    evaluate,
    get_strategy,
    link_speed_sweep,
    place_by_balance,
    place_contiguous,
    place_everything,
    place_round_robin,
    total_edges,
)

CHAIN = elementwise_chain(16)
WIDE = branching_graph(6, 3)


class TestAssignment:
    def test_it_records_where_each_node_runs(self):
        assignment = place_contiguous(CHAIN, devices=4)
        assert assignment.device_of("x") == 0

    def test_an_unplaced_node_is_reported(self):
        with pytest.raises(ScheduleError, match="has not been placed"):
            Assignment(devices=2).device_of("nothing")

    def test_a_device_that_does_not_exist_is_rejected(self):
        with pytest.raises(ConfigError, match="do not exist"):
            Assignment(devices=2, placement={"a": 5})

    def test_a_fleet_of_none_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one device"):
            Assignment(devices=0)

    def test_a_single_device_placement_uses_one(self):
        assert place_everything(CHAIN, devices=4).used_devices == 1

    def test_a_contiguous_placement_uses_them_all(self):
        assert place_contiguous(CHAIN, devices=4).used_devices == 4

    def test_the_nodes_on_a_device_can_be_listed(self):
        assert place_contiguous(CHAIN, devices=4).nodes_on(0)

    def test_it_serialises(self):
        assert place_contiguous(CHAIN, devices=4).as_dict()["used"] == 4


class TestCuts:
    def test_a_single_device_cuts_nothing(self):
        assert evaluate(CHAIN, place_everything(CHAIN)).cut_edges == 0

    def test_a_contiguous_split_cuts_one_edge_per_boundary(self):
        # The least any partition using every device can cut on a chain.
        assert evaluate(CHAIN, place_contiguous(CHAIN, devices=4)).cut_edges == 3

    def test_round_robin_cuts_almost_everything(self):
        assert cut_fraction(CHAIN, place_round_robin(CHAIN, devices=4)) > 0.9

    def test_and_contiguous_cuts_almost_nothing(self):
        assert cut_fraction(CHAIN, place_contiguous(CHAIN, devices=4)) < 0.2

    def test_the_cut_edges_are_named(self):
        crossing = cut_edges_of(CHAIN, place_contiguous(CHAIN, devices=4))
        assert len(crossing) == 3
        assert all(len(edge) == 2 for edge in crossing)

    def test_a_graph_with_no_edges_cuts_nothing(self):
        builder = Builder()
        graph = builder.finish(builder.constant(1.0))
        assert cut_fraction(graph, place_everything(graph)) == 0.0

    def test_every_edge_is_counted(self):
        graph = softmax_graph()
        assert total_edges(graph) == sum(len(node.inputs) for node in graph.nodes)
        assert total_edges(graph) == 7


class TestBalance:
    def test_a_single_device_is_balanced_over_the_one_it_uses(self):
        # Counting the empty devices instead calls it badly balanced, which is the wrong word:
        # its problem is that it uses one device, and the makespan already says that.
        assert evaluate(CHAIN, place_everything(CHAIN)).balance == 1.0

    def test_round_robin_balances_the_node_count_and_not_the_work(self):
        # An exponential costs eight times an addition, so an equal share of the nodes is not
        # an equal share of the arithmetic.
        assert evaluate(CHAIN, place_round_robin(CHAIN, devices=4)).balance < 0.7

    def test_a_contiguous_split_of_a_uniform_chain_is_even(self):
        assert evaluate(CHAIN, place_contiguous(CHAIN, devices=4)).balance == 1.0

    def test_an_empty_report_is_balanced(self):
        assert PartitionReport(strategy="none").balance == 1.0

    def test_a_negative_rate_is_rejected(self):
        report = evaluate(CHAIN, place_contiguous(CHAIN))
        with pytest.raises(ConfigError, match="have to be positive"):
            report.makespan(flops_per_second=0.0)


class TestComparison:
    def test_every_strategy_is_reported(self):
        assert len(compare_strategies(CHAIN)) == len(STRATEGIES)

    def test_round_robin_is_the_slowest_partition(self):
        rows = {row["strategy"]: row for row in compare_strategies(CHAIN)}
        others = [row["makespan"] for name, row in rows.items() if name != "round robin"]
        assert rows["round robin"]["makespan"] > max(others)

    def test_balancing_by_work_beats_balancing_by_node_count(self):
        rows = {row["strategy"]: row for row in compare_strategies(WIDE)}
        assert rows["balanced"]["makespan"] < rows["round robin"]["makespan"]

    def test_a_single_device_wins_when_the_link_is_slow(self):
        # Which is a statement about the interconnect and not about the graph.
        assert best_strategy(mlp_graph(batch=512, hidden=512)) == "one device"

    def test_and_loses_when_the_link_is_fast(self):
        rows = link_speed_sweep(mlp_graph(batch=512, hidden=512))
        assert rows[0]["winner"] == "one device"
        assert rows[-1]["winner"] != "one device"

    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown strategy"):
            get_strategy("spectral")

    def test_it_looks_up_by_name(self):
        assert get_strategy("contiguous") is place_contiguous


class TestSweeps:
    def test_more_devices_cut_more_edges(self):
        cuts = [row["cut_edges"] for row in device_sweep(CHAIN)]
        assert cuts == sorted(cuts)

    def test_and_move_more_bytes(self):
        moved = [row["transferred_bytes"] for row in device_sweep(CHAIN)]
        assert moved == sorted(moved)

    def test_one_device_moves_nothing(self):
        assert device_sweep(CHAIN, counts=[1])[0]["transferred_bytes"] == 0

    def test_an_empty_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            device_sweep(CHAIN, counts=())

    def test_an_empty_speed_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            link_speed_sweep(CHAIN, speeds=())

    def test_the_strategy_name_reaches_the_report(self):
        assert device_sweep(CHAIN, counts=[2])[0]["strategy"] == "contiguous"


class TestPlacement:
    def test_every_strategy_places_every_value(self):
        for strategy in STRATEGIES.values():
            assignment = strategy(WIDE, 4)
            assert set(assignment.placement) >= WIDE.value_names

    def test_the_inputs_always_start_on_the_first_device(self):
        for strategy in STRATEGIES.values():
            assert strategy(WIDE, 4).device_of("x") == 0

    def test_balancing_by_work_uses_every_device_on_a_uniform_graph(self):
        assert place_by_balance(CHAIN, devices=4).used_devices > 1

    def test_a_fleet_of_none_is_rejected(self):
        for strategy in (place_contiguous, place_round_robin, place_by_balance):
            with pytest.raises(ConfigError, match="at least one device"):
                strategy(CHAIN, 0)
