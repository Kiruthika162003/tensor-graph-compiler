from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import layernorm_graph, mlp_graph, softmax_graph
from tgc.runtime.checks import (
    POLICIES,
    RISKY,
    CheckPlan,
    CheckReport,
    a_graph_with_no_risky_operations_gets_no_checks,
    a_nan_at_a_safe_node_escapes_the_narrow_policy,
    an_assert_that_fires_is_an_error,
    an_unknown_policy_is_refused,
    caught_by,
    check_count,
    checks_survive_dead_code_elimination,
    compare_graphs,
    compare_policies,
    cost_sweep,
    every_injection_site,
    first_risky,
    how_many_sites_the_narrow_policy_misses,
    injecting_at_a_value_that_does_not_exist_is_refused,
    insert_checks,
    latency_by_policy,
    plan_for,
    poison,
    report_for,
    the_checked_graph_computes_the_same_thing,
    the_injector_really_poisons,
    the_narrow_policy_reads_a_third,
)
from tgc.verify.reference import random_feeds


class TestPlans:
    def test_the_output_policy_checks_one_value(self):
        assert len(plan_for(layernorm_graph(), "outputs only").checked) == 1

    def test_the_broad_policy_checks_everything(self):
        graph = layernorm_graph()
        assert plan_for(graph, "everything").coverage == 1.0

    def test_the_narrow_policy_checks_the_risky_operations(self):
        graph = layernorm_graph()
        plan = plan_for(graph, "risky operations")
        assert all(graph.node(name).op.name in RISKY for name in plan.checked)

    def test_an_unknown_policy_is_refused(self):
        assert an_unknown_policy_is_refused()

    def test_an_empty_plan_covers_nothing(self):
        assert CheckPlan(policy="none").coverage == 0.0

    def test_and_reads_nothing(self):
        assert CheckPlan(policy="none").read_share == 0.0

    def test_it_serialises(self):
        assert plan_for(layernorm_graph(), "everything").as_dict()["coverage"] == 1.0

    def test_three_policies_are_compared(self):
        assert len(compare_policies()) == len(POLICIES)


class TestInsertion:
    def test_the_checks_appear_in_the_graph(self):
        checked = insert_checks(layernorm_graph(), "everything")
        assert check_count(checked) == len(layernorm_graph().nodes)

    def test_and_the_answer_is_unchanged(self):
        # A check that changed a value would hide the thing it was added to find.
        assert the_checked_graph_computes_the_same_thing()["identical"]

    def test_a_policy_that_checks_nothing_leaves_the_graph_alone(self):
        graph = mlp_graph()
        assert insert_checks(graph, "risky operations") is graph

    def test_the_checks_survive_dead_code_elimination(self):
        # A check silently removed by a later pass is worse than no check.
        assert checks_survive_dead_code_elimination()["all_survived"]

    def test_the_check_is_the_assert_operation(self):
        checked = insert_checks(softmax_graph(), "outputs only")
        assert any(node.op is ops.ASSERT_FINITE for node in checked.nodes)

    def test_a_graph_without_checks_counts_none(self):
        assert check_count(layernorm_graph()) == 0


class TestDetection:
    def test_the_broad_policy_catches_a_nan_immediately(self):
        assert latency_by_policy()["everything"] == 0

    def test_the_narrow_one_does_too_at_a_risky_node(self):
        assert latency_by_policy()["risky_operations"] == 0

    def test_and_the_output_policy_lets_it_run(self):
        assert latency_by_policy()["outputs_only"] > 0

    def test_a_nan_at_a_safe_node_takes_longer_to_find(self):
        result = a_nan_at_a_safe_node_escapes_the_narrow_policy()
        assert result["narrow_latency"] > result["broad_latency"]

    def test_but_is_still_found_on_a_layernorm(self):
        # A nan injected anywhere reaches a division eventually.
        assert a_nan_at_a_safe_node_escapes_the_narrow_policy()["narrow_catches_it"]

    def test_the_narrow_policy_misses_nothing_there(self):
        assert how_many_sites_the_narrow_policy_misses()["missed"] == 0

    def test_every_node_is_tried_as_a_site(self):
        graph = layernorm_graph()
        assert len(every_injection_site(graph)) == len(graph.nodes)

    def test_a_site_the_graph_does_not_have_is_refused(self):
        assert injecting_at_a_value_that_does_not_exist_is_refused()

    def test_a_graph_with_no_risky_operation_has_no_first_one(self):
        with pytest.raises(ConfigError, match="no operation that can manufacture"):
            first_risky(mlp_graph())

    def test_a_caught_nan_is_reported_as_caught(self):
        graph = layernorm_graph()
        assert caught_by(graph, "everything", graph.nodes[0].name)


class TestInjector:
    def test_an_injected_nan_reaches_the_output(self):
        # If it did not, the experiment would be measuring nothing.
        assert the_injector_really_poisons()["output_is_nan"]

    def test_and_the_clean_run_is_finite(self):
        assert the_injector_really_poisons()["clean_output_is_finite"]

    def test_injecting_at_a_missing_value_is_refused(self):
        graph = layernorm_graph()
        feeds = random_feeds(graph, positive=True)
        with pytest.raises(ConfigError, match="not a value"):
            poison(graph, feeds, "nowhere")

    def test_the_injection_leaves_upstream_values_alone(self):
        graph = layernorm_graph()
        feeds = random_feeds(graph, positive=True)
        site = graph.nodes[-1].name
        poisoned = poison(graph, feeds, site)
        assert bool(poisoned[graph.nodes[0].name].isfinite().all())


class TestFixtures:
    def test_an_mlp_gets_no_checks_from_the_narrow_policy(self):
        # A policy written around the operations that manufacture nans has nothing to say about
        # a graph made of products and rectifiers.
        assert a_graph_with_no_risky_operations_gets_no_checks()["checks"] == 0

    def test_and_misses_every_site(self):
        result = a_graph_with_no_risky_operations_gets_no_checks()
        assert result["sites_missed"] == result["nodes"]

    def test_which_shows_up_in_the_comparison(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["mlp"]["missed_share"] == 1.0

    def test_while_the_normalisations_miss_nothing(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["layernorm"]["missed_share"] == 0.0
        assert rows["softmax"]["missed_share"] == 0.0

    def test_three_graphs_are_compared(self):
        assert len(compare_graphs()) == 3


class TestCost:
    def test_the_narrow_policy_reads_a_third(self):
        assert 0.3 < the_narrow_policy_reads_a_third()["risky_operations"] < 0.4

    def test_the_broad_one_reads_everything(self):
        assert the_narrow_policy_reads_a_third()["everything"] == 1.0

    def test_a_free_read_costs_nothing(self):
        assert cost_sweep()[0]["everything"] == 0.0

    def test_and_a_full_price_read_doubles_the_graph(self):
        assert cost_sweep()[-1]["everything"] == 1.0

    def test_an_empty_cost_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            cost_sweep(element_cost=())

    def test_a_report_packages_the_numbers(self):
        report = report_for(layernorm_graph(), "risky operations", "layernorm")
        assert report.as_dict()["policy"] == "risky operations"

    def test_an_empty_report_checks_nothing(self):
        assert CheckReport(graph="none", policy="none").checks == 0

    def test_a_failing_check_raises(self):
        # A check that printed a warning and carried on would produce a run that finished and a
        # line in a log nobody read.
        assert an_assert_that_fires_is_an_error()
