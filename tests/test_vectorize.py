from __future__ import annotations

import pytest

from tgc.codegen.vectorize import (
    WIDTHS,
    GraphVectorReport,
    VectorPlan,
    analyse,
    best_width,
    can_vectorise,
    check_plan,
    compare_graphs,
    every_plan_accounts_for_its_elements,
    length_sweep,
    refusals,
    vectorisable_fraction,
    widest_is_not_always_best,
    width_sweep,
)
from tgc.errors import ConfigError
from tgc.ir.builder import elementwise_chain, layernorm_graph, mlp_graph, softmax_graph


class TestPlan:
    def test_a_length_that_divides_wastes_nothing(self):
        assert VectorPlan(length=64, width=8).wasted_lanes == 0

    def test_and_reaches_the_full_speedup(self):
        assert VectorPlan(length=64, width=8).speedup == 8.0

    def test_a_length_one_past_a_multiple_wastes_almost_a_whole_iteration(self):
        # Nine at width eight issues two iterations and throws away seven lanes.
        plan = VectorPlan(length=9, width=8)
        assert plan.iterations == 2
        assert plan.wasted_lanes == 7

    def test_and_halves_the_speedup(self):
        assert VectorPlan(length=9, width=8).speedup == 4.5

    def test_a_width_of_one_is_the_scalar_loop(self):
        plan = VectorPlan(length=100, width=1)
        assert plan.speedup == 1.0
        assert plan.wasted_lanes == 0

    def test_an_empty_loop_issues_nothing(self):
        plan = VectorPlan(length=0, width=8)
        assert plan.iterations == 0
        assert plan.waste_fraction == 0.0
        assert plan.speedup == 1.0

    def test_a_tail_is_reported(self):
        assert VectorPlan(length=9, width=8).has_tail
        assert not VectorPlan(length=16, width=8).has_tail

    def test_a_negative_length_is_rejected(self):
        with pytest.raises(ConfigError, match="cannot run"):
            VectorPlan(length=-1, width=8)

    def test_a_zero_width_is_rejected(self):
        with pytest.raises(ConfigError, match="at least one element wide"):
            VectorPlan(length=8, width=0)

    def test_it_serialises(self):
        assert VectorPlan(length=9, width=8).as_dict()["wasted_lanes"] == 7


class TestArithmetic:
    def test_every_plan_accounts_for_every_element(self):
        # Swept rather than sampled, for the same reason peeling was: an off by one at a
        # boundary is invisible at whatever size somebody picked.
        assert every_plan_accounts_for_its_elements() == 1000

    def test_a_broken_plan_would_be_caught(self):
        # The check is the arithmetic written out, so a plan that satisfies it accounts for
        # its elements by construction.
        for length in (0, 1, 7, 8, 9, 100):
            for width in WIDTHS:
                check_plan(VectorPlan(length=length, width=width))

    def test_the_full_iterations_and_remainder_partition_the_length(self):
        plan = VectorPlan(length=100, width=16)
        assert plan.full_iterations * plan.width + plan.remainder == plan.length


class TestSweeps:
    def test_a_wider_unit_issues_fewer_iterations(self):
        rows = width_sweep()
        iterations = [row["iterations"] for row in rows]
        assert iterations == sorted(iterations, reverse=True)

    def test_but_wastes_more_lanes(self):
        rows = {row["width"]: row for row in width_sweep()}
        assert rows[16]["wasted_lanes"] > rows[8]["wasted_lanes"]

    def test_the_speedup_stops_keeping_up_with_the_width(self):
        rows = {row["width"]: row for row in width_sweep()}
        assert rows[16]["speedup"] < 16.0
        assert rows[4]["speedup"] == 4.0

    def test_a_short_loop_wastes_a_large_share(self):
        rows = {row["length"]: row for row in length_sweep()}
        assert rows[9]["waste_fraction"] > 0.4

    def test_and_a_long_one_almost_none(self):
        rows = {row["length"]: row for row in length_sweep()}
        assert rows[1000]["waste_fraction"] == 0.0

    def test_a_length_one_short_of_a_multiple_wastes_one_lane(self):
        rows = {row["length"]: row for row in length_sweep()}
        assert rows[7]["wasted_lanes"] == 1

    def test_an_empty_width_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            width_sweep(widths=())

    def test_an_empty_length_sweep_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            length_sweep(lengths=())


class TestWidthChoice:
    def test_a_length_that_divides_takes_the_widest_that_fits(self):
        assert best_width(64) == 16

    def test_a_short_loop_takes_a_narrower_one(self):
        # Seven at width sixteen issues one iteration and wastes nine lanes; at width eight it
        # issues one and wastes one.
        assert best_width(7) == 8

    def test_the_widest_is_not_always_chosen(self):
        rows = widest_is_not_always_best()
        assert any(row["chosen"] != row["widest"] for row in rows)

    def test_and_where_it_is_not_it_wastes_less(self):
        rows = {row["length"]: row for row in widest_is_not_always_best()}
        assert rows[7]["chosen_waste"] < rows[7]["widest_waste"]

    def test_a_zero_length_is_rejected(self):
        with pytest.raises(ConfigError, match="has to run"):
            best_width(0)

    def test_an_empty_choice_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to choose"):
            best_width(64, widths=())

    def test_an_empty_comparison_is_rejected(self):
        with pytest.raises(ConfigError, match="nothing to compare"):
            widest_is_not_always_best(lengths=())


class TestOperations:
    def test_an_elementwise_operation_vectorises(self):
        graph = elementwise_chain(4)
        assert can_vectorise(graph.nodes[0])[0]

    def test_a_reduction_does_not(self):
        # Every lane would have to reach the same accumulator.
        allowed, reason = can_vectorise(softmax_graph().node("v0"))
        assert not allowed
        assert "accumulator" in reason

    def test_nor_a_matrix_product(self):
        graph = mlp_graph()
        matmul = next(node for node in graph.nodes if node.op.name == "matmul")
        allowed, reason = can_vectorise(matmul)
        assert not allowed
        assert "reduction in disguise" in reason

    def test_a_leaf_has_nothing_to_vectorise(self):
        graph = mlp_graph()
        assert vectorisable_fraction(graph) > 0


class TestGraphs:
    def test_a_chain_is_entirely_vectorisable(self):
        assert vectorisable_fraction(elementwise_chain(8)) == 1.0

    def test_a_softmax_is_not(self):
        # Two reductions holding an elementwise chain between them, which is the shape of most
        # tensor code and the reason reduction splitting exists.
        assert vectorisable_fraction(softmax_graph()) < 1.0

    def test_the_refusals_name_the_reason(self):
        assert "accumulator" in next(iter(refusals(softmax_graph()).values()))

    def test_leaves_are_not_counted_as_refusals(self):
        # Counting them made a layernorm look less vectorisable than it is.
        assert all("leaf" not in reason for reason in refusals(layernorm_graph()).values())

    def test_every_fixture_is_compared(self):
        assert len(compare_graphs()) == 4

    def test_the_chain_leads_the_comparison(self):
        rows = {row["graph"]: row for row in compare_graphs()}
        assert rows["chain"]["fraction"] == 1.0
        assert rows["mlp"]["fraction"] < rows["chain"]["fraction"]

    def test_an_empty_report_vectorises_nothing(self):
        assert GraphVectorReport().fraction == 0.0

    def test_it_serialises(self):
        assert analyse(elementwise_chain(8)).as_dict()["vectorisable"] == 8
