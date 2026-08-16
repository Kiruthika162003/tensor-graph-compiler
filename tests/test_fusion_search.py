from __future__ import annotations

import pytest

from tgc.errors import ConfigError, ScheduleError
from tgc.schedule.fusion_search import (
    Dag,
    FusionPlan,
    a_forward_reference_is_refused,
    a_large_graph_is_refused,
    chain_dag,
    cheap_cone,
    compare_graphs,
    compare_strategies,
    cone_dag,
    cost_of,
    duplication_factor,
    duplication_grows_with_the_reader_count,
    exhaustive,
    expensive_cone,
    expensive_memory_means_recompute,
    free_memory_means_no_duplication,
    group_for,
    groups_of,
    intensity_sweep,
    materialise_everything,
    materialise_outputs_only,
    materialise_reused,
    materialising_an_input_is_refused,
    memory_weight_sweep,
    on_a_chain_maximum_fusion_is_optimal,
    report_for,
    search_size,
    the_cone_is_walked_once_per_reader,
    the_crossover_is_a_property_of_the_machine,
    the_rule_loses_on_a_cheap_cone,
    the_rule_wins_on_an_expensive_cone,
    times_computed,
)


class TestGraphs:
    def test_a_chain_has_one_input_and_one_output(self):
        summary = chain_dag().as_dict()
        assert summary["inputs"] == 1
        assert summary["outputs"] == 1

    def test_and_no_barriers(self):
        assert chain_dag().as_dict()["barriers"] == 0

    def test_a_cone_has_one_barrier_per_reader(self):
        assert cone_dag(readers=5).as_dict()["barriers"] == 5

    def test_only_the_tip_of_the_cone_fans_out(self):
        graph = cheap_cone()
        assert [graph.fan_out(index) for index in range(graph.count)].count(3) == 1

    def test_the_search_space_excludes_the_barriers(self):
        graph = cheap_cone()
        assert not set(graph.interior) & graph.required

    def test_and_the_outputs(self):
        graph = cheap_cone()
        assert not set(graph.interior) & set(graph.outputs)

    def test_and_the_inputs(self):
        graph = cheap_cone()
        assert not set(graph.interior) & graph.leaves

    def test_a_forward_reference_is_refused(self):
        assert a_forward_reference_is_refused()

    def test_a_graph_with_no_values_is_refused(self):
        with pytest.raises(ConfigError, match="at least one value"):
            Dag(producers=(), sizes=(), flops=(), outputs=())

    def test_a_missing_flop_count_is_refused(self):
        with pytest.raises(ConfigError, match="a size and a flop count"):
            Dag(producers=((), (0,)), sizes=(1, 1), flops=(1,), outputs=(1,))

    def test_an_input_named_as_an_output_is_refused(self):
        with pytest.raises(ConfigError, match="computes nothing"):
            Dag(producers=((), (0,)), sizes=(1, 1), flops=(0, 1), outputs=(0,))

    def test_a_graph_with_no_outputs_is_refused(self):
        with pytest.raises(ConfigError, match="computes nothing"):
            Dag(producers=((), (0,)), sizes=(1, 1), flops=(0, 1), outputs=())

    def test_a_one_reader_cone_is_refused(self):
        with pytest.raises(ConfigError, match="not a fan out"):
            cone_dag(readers=1)

    def test_a_zero_length_chain_is_refused(self):
        with pytest.raises(ConfigError, match="not a chain"):
            chain_dag(length=0)

    def test_asking_about_a_value_that_does_not_exist_is_refused(self):
        with pytest.raises(ConfigError, match="not a value"):
            cheap_cone().consumers(99)


class TestGrouping:
    def test_a_fully_fused_cone_has_one_group_per_barrier(self):
        graph = cheap_cone()
        assert materialise_outputs_only(graph).groups == len(graph.barriers) + 1

    def test_and_each_of_them_contains_the_whole_cone(self):
        graph = cheap_cone()
        materialised = graph.required
        assert all(4 in group_for(graph, barrier, materialised) for barrier in graph.barriers)

    def test_writing_the_tip_takes_the_cone_out_of_them(self):
        graph = cheap_cone()
        groups = groups_of(graph, (4,))
        assert all(len(groups[barrier]) == 1 for barrier in graph.barriers)

    def test_an_unfused_graph_has_one_group_per_value(self):
        graph = cheap_cone()
        plan = materialise_everything(graph)
        assert plan.groups == len(graph.interior) + len(graph.required)

    def test_a_group_cannot_be_rooted_at_an_input(self):
        with pytest.raises(ScheduleError, match="does not root a group"):
            group_for(chain_dag(), 0, frozenset())

    def test_materialising_an_input_is_refused(self):
        assert materialising_an_input_is_refused()

    def test_a_value_outside_the_graph_is_refused(self):
        with pytest.raises(ScheduleError, match="not a value"):
            cost_of(chain_dag(), (99,))

    def test_asking_how_often_a_missing_value_runs_is_refused(self):
        with pytest.raises(ScheduleError, match="not a value"):
            times_computed(chain_dag(), (), 99)

    def test_a_negative_memory_weight_is_refused(self):
        with pytest.raises(ConfigError, match="not a weight"):
            cost_of(chain_dag(), (), memory_weight=-1.0)


class TestChains:
    def test_maximum_fusion_is_optimal_on_a_chain(self):
        assert on_a_chain_maximum_fusion_is_optimal()["optimal"]

    def test_because_nothing_can_be_duplicated(self):
        assert on_a_chain_maximum_fusion_is_optimal()["duplication"] == 1.0

    def test_and_it_is_worth_seven_times_the_unfused_graph(self):
        assert on_a_chain_maximum_fusion_is_optimal()["against_no_fusion"] > 7.0

    def test_every_plan_on_a_chain_does_the_same_arithmetic(self):
        graph = chain_dag()
        assert materialise_everything(graph).flops == materialise_outputs_only(graph).flops

    def test_so_the_rule_and_the_search_agree(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["chain"]["matches"]


class TestDuplication:
    def test_the_cone_runs_once_per_reader(self):
        assert the_cone_is_walked_once_per_reader(readers=3)["tip_computed"] == 3

    def test_and_once_when_the_tip_is_written(self):
        result = the_cone_is_walked_once_per_reader()
        assert result["tip_computed_under_the_rule"] == 1

    def test_five_readers_means_five_walks(self):
        assert the_cone_is_walked_once_per_reader(readers=5)["tip_computed"] == 5

    def test_the_duplication_rises_with_the_reader_count(self):
        rows = duplication_grows_with_the_reader_count()
        assert [row["duplication"] for row in rows] == sorted(
            row["duplication"] for row in rows
        )

    def test_but_never_reaches_the_reader_count(self):
        # The joins below the reductions are not shared, so they dilute the average.
        rows = {row["readers"]: row for row in duplication_grows_with_the_reader_count()}
        assert rows[5]["duplication"] < 5

    def test_the_fused_flops_grow_faster_than_the_rule_flops(self):
        rows = duplication_grows_with_the_reader_count(counts=(2, 5))
        fused = rows[1]["fused_flops"] / rows[0]["fused_flops"]
        ruled = rows[1]["rule_flops"] / rows[0]["rule_flops"]
        assert fused > ruled

    def test_an_empty_reader_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            duplication_grows_with_the_reader_count(counts=())

    def test_a_plan_with_no_duplication_reports_one(self):
        graph = cheap_cone()
        assert duplication_factor(graph, graph.interior) == 1.0


class TestTheRule:
    def test_it_wins_on_a_cone_that_does_arithmetic(self):
        assert the_rule_wins_on_an_expensive_cone()["matches"]

    def test_and_beats_maximum_fusion_by_most_of_a_factor_of_two(self):
        result = the_rule_wins_on_an_expensive_cone()
        assert result["fully_fused"] / result["rule"] > 1.9

    def test_it_loses_on_a_cone_that_does_not(self):
        assert the_rule_loses_on_a_cheap_cone()["regret"] > 1.5

    def test_by_keeping_a_value_the_search_throws_away(self):
        result = the_rule_loses_on_a_cheap_cone()
        assert set(result["rule_materialised"]) - set(result["best_materialised"]) == {4}

    def test_and_the_search_agrees_with_maximum_fusion_there(self):
        result = the_rule_loses_on_a_cheap_cone()
        assert result["best"] == result["fully_fused"]

    def test_two_of_the_three_shapes_match(self):
        assert sum(1 for row in compare_graphs() if row["matches"]) == 2

    def test_the_one_that_does_not_is_the_cheap_cone(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert not rows["cheap cone"]["matches"]

    def test_four_strategies_are_compared(self):
        assert len(compare_strategies()) == 4

    def test_the_unfused_plan_is_the_worst_of_them(self):
        rows = compare_strategies()
        assert max(rows, key=lambda row: row["cost"])["strategy"] == "everything"


class TestCrossover:
    def test_the_rule_is_wrong_at_one_flop_per_element(self):
        assert not the_crossover_is_a_property_of_the_machine()["at_one"]

    def test_and_right_at_sixty_four(self):
        assert the_crossover_is_a_property_of_the_machine()["at_sixty_four"]

    def test_the_flip_is_between_eight_and_sixteen(self):
        assert the_crossover_is_a_property_of_the_machine()["flips_at"] == 16

    def test_the_regret_shrinks_as_the_cone_gets_expensive(self):
        rows = [row["regret"] for row in intensity_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_and_never_exceeds_half_again(self):
        assert the_crossover_is_a_property_of_the_machine()["worst_regret"] < 1.52

    def test_once_the_rule_is_best_it_stays_best(self):
        rows = intensity_sweep()
        flipped = [index for index, row in enumerate(rows) if row["rule_is_best"]]
        assert flipped == list(range(min(flipped), len(rows)))

    def test_an_empty_intensity_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            intensity_sweep(intensities=())

    def test_the_graph_is_identical_across_the_sweep(self):
        # Only the flop count per element changes, so the flip is the machine and not the shape.
        assert cone_dag(intensity=1).producers == cone_dag(intensity=64).producers


class TestMemoryWeight:
    def test_free_memory_removes_every_duplication(self):
        assert free_memory_means_no_duplication()["duplication"] == 1.0

    def test_without_writing_everything(self):
        result = free_memory_means_no_duplication()
        assert result["materialised"] < result["everything"]

    def test_because_the_extra_writes_buy_no_arithmetic(self):
        result = free_memory_means_no_duplication()
        assert result["flops"] == result["unfused_flops"]

    def test_expensive_memory_gives_values_up(self):
        assert expensive_memory_means_recompute()["fell"]

    def test_and_accepts_the_duplication_that_comes_with_it(self):
        assert expensive_memory_means_recompute()["duplication_at_dear_memory"] > 1.5

    def test_the_sweep_crosses_once(self):
        rows = [row["materialised"] for row in memory_weight_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_an_empty_weight_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            memory_weight_sweep(weights=())

    def test_a_free_write_costs_nothing(self):
        assert cost_of(chain_dag(), (), memory_weight=0.0).cost == 32768


class TestSearch:
    def test_the_search_never_loses_to_a_rule(self):
        for graph in (chain_dag(), cheap_cone(), expensive_cone()):
            assert exhaustive(graph).cost <= materialise_reused(graph).cost

    def test_or_to_either_extreme(self):
        graph = cheap_cone()
        best = exhaustive(graph).cost
        assert best <= materialise_everything(graph).cost
        assert best <= materialise_outputs_only(graph).cost

    def test_the_space_is_two_to_the_interior_count(self):
        summary = search_size()
        assert summary["plans"] == 2 ** summary["interior"]

    def test_a_deep_cone_is_millions_of_plans(self):
        assert search_size()["at_depth_twenty"] > 1_000_000

    def test_a_graph_too_large_to_enumerate_is_refused(self):
        assert a_large_graph_is_refused()

    def test_a_report_packages_the_numbers(self):
        assert report_for(cheap_cone(), "cheap cone").as_dict()["plans"] == 32

    def test_and_carries_the_regret(self):
        assert report_for(cheap_cone(), "cheap cone").regret > 1.5

    def test_an_empty_report_reports_no_regret(self):
        assert FusionPlan().cost == 0.0
