from __future__ import annotations

import pytest

from tgc.analysis.cost import (
    BANDWIDTH_STARVED,
    CPU,
    GPU,
    MACHINES,
    Machine,
    RooflineEstimate,
    annotate_matmuls,
    compare_machines,
    estimate,
    fusion_speedup,
    get_machine,
    graph_flops,
    kendall_agreement,
    node_flops,
    rank_by_cost,
    size_sweep,
    traffic_against_speedup,
)
from tgc.errors import ConfigError
from tgc.ir.builder import elementwise_chain, mlp_graph, softmax_graph
from tgc.passes.fusion import traffic_ratio


class TestMachine:
    def test_the_ridge_point_is_the_ratio_of_the_peaks(self):
        assert Machine(flops_per_second=100.0, bytes_per_second=10.0).ridge_point == 10.0

    def test_a_bandwidth_starved_machine_has_a_higher_ridge(self):
        assert BANDWIDTH_STARVED.ridge_point > GPU.ridge_point

    def test_a_non_positive_peak_is_rejected(self):
        with pytest.raises(ConfigError, match="have to be positive"):
            Machine(flops_per_second=0.0)

    def test_it_looks_up_by_name(self):
        assert get_machine("gpu") is GPU

    def test_an_unknown_machine_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown machine"):
            get_machine("tpu")

    def test_every_machine_is_registered_under_its_name(self):
        assert all(name == machine.name for name, machine in MACHINES.items())

    def test_it_serialises(self):
        assert CPU.as_dict()["name"] == "cpu"


class TestFlops:
    def test_a_matmul_costs_more_than_its_output(self):
        # Its work is the output times the contracted dimension, and a model that misses that
        # ranks a matmul alongside an addition.
        graph = annotate_matmuls(mlp_graph())
        matmul = next(node for node in graph.nodes if node.op.name == "matmul")
        elements = matmul.output.shape.elements
        assert node_flops(matmul) > elements

    def test_an_uncosted_matmul_is_rejected(self):
        graph = mlp_graph()
        matmul = next(node for node in graph.nodes if node.op.name == "matmul")
        with pytest.raises(ConfigError, match="contracted dimension"):
            node_flops(matmul)

    def test_a_leaf_costs_nothing(self):
        graph = softmax_graph()
        assert graph_flops(graph) > 0

    def test_a_transcendental_costs_more_than_an_addition(self):
        chain = elementwise_chain(8)
        assert graph_flops(chain) > 8 * chain.nodes[0].output.shape.elements

    def test_a_symbolic_graph_cannot_be_costed(self):
        with pytest.raises(ConfigError, match="symbolic shape"):
            graph_flops(mlp_graph(batch="batch"))

    def test_annotating_leaves_other_nodes_alone(self):
        graph = softmax_graph()
        assert annotate_matmuls(graph).nodes == graph.nodes


class TestRoofline:
    def test_a_chain_is_memory_bound(self):
        assert estimate(elementwise_chain(8), GPU).is_memory_bound

    def test_a_large_matmul_is_not(self):
        assert not estimate(mlp_graph(batch=512, hidden=1024), GPU).is_memory_bound

    def test_the_time_is_the_larger_of_the_two_limits(self):
        result = estimate(elementwise_chain(8), GPU)
        assert result.seconds == max(result.compute_seconds, result.memory_seconds)

    def test_a_memory_bound_graph_wastes_the_arithmetic_units(self):
        assert estimate(elementwise_chain(8), GPU).utilisation < 0.5

    def test_a_graph_that_moves_nothing_has_no_intensity(self):
        assert RooflineEstimate(flops=0.0, bytes_moved=0.0, machine=GPU).intensity == 0.0
        assert RooflineEstimate(flops=0.0, bytes_moved=0.0, machine=GPU).utilisation == 0.0

    def test_it_serialises(self):
        assert estimate(elementwise_chain(8)).as_dict()["memory_bound"]


class TestFusionSpeedup:
    def test_a_memory_bound_chain_gains_the_whole_traffic_ratio(self):
        graph = elementwise_chain(8)
        assert fusion_speedup(graph, GPU) == pytest.approx(traffic_ratio(graph))

    def test_a_compute_bound_block_gains_nothing(self):
        # The bytes fusion removes were not the ones anybody was waiting on.
        graph = mlp_graph(batch=512, hidden=1024)
        assert fusion_speedup(graph, GPU) == 1.0

    def test_and_the_byte_saving_is_still_real(self):
        # A compiler that reports the byte number as a speedup is not lying about the bytes.
        graph = mlp_graph(batch=512, hidden=1024)
        assert traffic_ratio(graph) > 1.2

    def test_the_sweep_crosses_the_ridge(self):
        rows = size_sweep()
        assert rows[0]["memory_bound"]
        assert not rows[-1]["memory_bound"]

    def test_the_small_end_gains_what_the_traffic_promised(self):
        first = size_sweep()[0]
        assert first["speedup"] == first["traffic_ratio"]

    def test_the_large_end_does_not(self):
        last = size_sweep()[-1]
        assert last["speedup"] == 1.0
        assert last["traffic_ratio"] > 1.0

    def test_an_empty_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            size_sweep(sizes=())

    def test_it_reports_both_numbers(self):
        row = traffic_against_speedup(elementwise_chain(8))
        assert row["traffic_ratio"] == row["speedup"]


class TestMachineComparison:
    def test_every_machine_is_reported(self):
        assert len(compare_machines(elementwise_chain(8))) == len(MACHINES)

    def test_a_chain_is_memory_bound_everywhere(self):
        assert all(row["memory_bound"] for row in compare_machines(elementwise_chain(8)))

    def test_a_starved_machine_reaches_less_of_its_peak(self):
        rows = {row["machine"]: row for row in compare_machines(elementwise_chain(8))}
        assert rows["bandwidth starved"]["utilisation"] < rows["cpu"]["utilisation"]


class TestRanking:
    def test_it_orders_the_cheapest_first(self):
        candidates = [
            ("wide", elementwise_chain(4, sizes=(128, 128))),
            ("narrow", elementwise_chain(4, sizes=(16, 16))),
        ]
        assert rank_by_cost(candidates)[0] == "narrow"

    def test_two_fused_chains_of_different_lengths_tie(self):
        # Correct rather than a limitation. Fused, both move exactly one input and one
        # output, and while both sit below the ridge point the extra arithmetic in the longer
        # one is free.
        candidates = [
            ("long", elementwise_chain(16, sizes=(64, 64))),
            ("short", elementwise_chain(2, sizes=(64, 64))),
        ]
        costs = {name: estimate(graph).seconds for name, graph in candidates}
        assert costs["long"] == costs["short"]

    def test_ranking_nothing_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to rank"):
            rank_by_cost([])

    def test_a_ranking_agrees_with_itself_completely(self):
        order = ["a", "b", "c"]
        assert kendall_agreement(order, order) == 1.0

    def test_a_reversed_ranking_disagrees_completely(self):
        assert kendall_agreement(["a", "b", "c"], ["c", "b", "a"]) == -1.0

    def test_a_single_swap_is_partial_agreement(self):
        agreement = kendall_agreement(["a", "b", "c"], ["b", "a", "c"])
        assert 0.0 < agreement < 1.0

    def test_two_rankings_of_different_things_are_rejected(self):
        with pytest.raises(ConfigError, match="same candidates"):
            kendall_agreement(["a", "b"], ["a", "c"])

    def test_a_ranking_of_one_has_no_order(self):
        with pytest.raises(ConfigError, match="no order to compare"):
            kendall_agreement(["a"], ["a"])
