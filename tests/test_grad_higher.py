from __future__ import annotations

import pytest

from tgc.errors import ConfigError, PassError
from tgc.grad.higher import (
    CORNERED,
    HvpResult,
    agrees_with_torch,
    compare_with_torch,
    corner_operations,
    growth_by_graph,
    has_a_corner,
    hessian_vector_product,
    measure_quadratic,
    order_of_composition,
    quadratic_graph,
    reverse_over_reverse,
    saturation_flattens_the_curvature,
    smooth_chain,
    symmetry_check,
    symmetry_holds_everywhere,
    two_compositions_agree,
    which_graphs_admit_a_second_derivative,
)
from tgc.ir.builder import layernorm_graph, mlp_graph, softmax_graph


class TestWhatCanBeDifferentiatedTwice:
    def test_a_relu_network_cannot(self):
        # A relu is piecewise linear, so its second derivative is zero on every piece and
        # undefined between them. Saying so at compile time beats a tensor of zeros at runtime.
        assert has_a_corner(mlp_graph())

    def test_a_softmax_cannot_either(self):
        # A max reduction is a corner in disguise.
        assert has_a_corner(softmax_graph())

    def test_a_layernorm_can(self):
        assert not has_a_corner(layernorm_graph())

    def test_a_smooth_chain_can(self):
        assert not has_a_corner(smooth_chain())

    def test_the_blocking_node_is_named(self):
        assert len(corner_operations(mlp_graph())) == 1

    def test_building_one_anyway_is_refused(self):
        with pytest.raises(PassError, match="no derivative"):
            hessian_vector_product(mlp_graph())

    def test_and_so_is_the_other_composition(self):
        with pytest.raises(PassError, match="no derivative"):
            reverse_over_reverse(softmax_graph())

    def test_every_fixture_is_classified(self):
        rows = {row["graph"]: row for row in which_graphs_admit_a_second_derivative()}
        assert rows["mlp"]["blocked_by"] == ["relu"]
        assert rows["softmax"]["blocked_by"] == ["max"]

    def test_the_corner_list_holds_the_reduction_too(self):
        assert "max" in CORNERED

    def test_an_empty_chain_is_refused(self):
        with pytest.raises(ConfigError, match="at least one operation"):
            smooth_chain(0)


class TestTheOneReadableCase:
    def test_a_square_has_a_second_derivative_of_two(self):
        # A hessian vector product against a direction has to come back as twice that
        # direction times the seed, exactly, in every position.
        assert measure_quadratic()["largest_gap"] == 0.0

    def test_and_the_values_are_not_all_zero(self):
        assert measure_quadratic()["largest_value"] > 1.0


class TestAgreementWithTorch:
    def test_a_smooth_chain_matches_a_double_backward(self):
        assert agrees_with_torch(smooth_chain())

    def test_a_quadratic_matches_exactly(self):
        assert compare_with_torch(quadratic_graph())["largest_gap"] == 0.0

    def test_a_layernorm_matches(self):
        assert agrees_with_torch(layernorm_graph())

    def test_a_deeper_chain_matches(self):
        assert agrees_with_torch(smooth_chain(6))


class TestSymmetry:
    def test_the_hessian_is_symmetric_on_every_fixture(self):
        # A second derivative does not depend on the order the two directions were taken in.
        assert all(row["symmetric"] for row in symmetry_holds_everywhere())

    def test_a_quadratic_is_symmetric_to_the_last_bit(self):
        assert symmetry_check(quadratic_graph())["relative_gap"] == 0.0

    def test_both_sides_are_reported(self):
        result = symmetry_check(smooth_chain())
        assert result["one_way"] != 0.0
        assert result["the_other"] != 0.0

    def test_three_fixtures_are_checked(self):
        assert len(symmetry_holds_everywhere()) == 3


class TestComposition:
    def test_the_two_compositions_compute_the_same_product(self):
        # They take their direction from different places, and a confusion between the
        # direction and the seed produces a plausible tensor of the right shape.
        assert all(row["agree"] for row in two_compositions_agree())

    def test_forward_over_reverse_is_not_reliably_the_smaller_graph(self):
        # Counted in nodes the usual advice does not hold. What the second reverse pass costs
        # is keeping the whole first gradient alive, which a node count cannot see.
        ratios = [row["ratio"] for row in order_of_composition()]
        assert min(ratios) < 1.0

    def test_it_is_smaller_on_a_layernorm(self):
        rows = {row["graph"]: row for row in order_of_composition()}
        assert rows["layernorm"]["ratio"] > 1.0

    def test_and_larger_on_a_quadratic(self):
        rows = {row["graph"]: row for row in order_of_composition()}
        assert rows["quadratic"]["ratio"] < 1.0

    def test_a_second_order_graph_is_an_order_of_magnitude_bigger(self):
        assert all(row["growth"] > 10.0 for row in growth_by_graph())

    def test_a_result_serialises(self):
        assert hessian_vector_product(quadratic_graph()).as_dict()["original_nodes"] == 1

    def test_an_empty_result_has_no_growth(self):
        empty = HvpResult(graph=quadratic_graph(), original_nodes=0, gradient_nodes=0)
        assert empty.growth == 0.0


class TestSaturation:
    def test_the_curvature_peaks_in_the_middle_of_the_range(self):
        rows = saturation_flattens_the_curvature()
        peak = max(rows, key=lambda row: row["largest_curvature"])
        assert peak["scale"] == 1.0

    def test_and_falls_away_by_orders_of_magnitude_past_saturation(self):
        rows = {row["scale"]: row for row in saturation_flattens_the_curvature()}
        assert rows[16.0]["largest_curvature"] < rows[1.0]["largest_curvature"] / 1e6

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            saturation_flattens_the_curvature(scales=())
