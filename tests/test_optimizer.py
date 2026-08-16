from __future__ import annotations

import pytest
import torch

from tgc.analysis.optimizer import (
    ADAM,
    MOMENTUM,
    OPTIMIZERS,
    SGD,
    MemoryBreakdown,
    OptimizerKind,
    a_narrow_moment_stalls,
    activation_bytes,
    adam_step,
    batch_sweep,
    bias_correction_matters_for_thousands_of_steps,
    bytes_per_parameter,
    cheaper_arithmetic_would_not_help,
    compare_optimizers,
    matches_the_library,
    memory_split,
    mixed_precision_saves_nothing_on_the_parameters,
    narrow_moments_would_save,
    sharded_bytes_per_parameter,
    sharding_has_a_floor,
    sharding_sweep,
    the_correction_is_not_monotonic,
    the_sixteen_bytes,
    training_memory,
    update_is_memory_bound,
    where_activations_overtake,
)
from tgc.errors import ConfigError


class TestBreakdown:
    def test_adam_in_mixed_precision_costs_sixteen_bytes(self):
        assert the_sixteen_bytes()["total"] == 16

    def test_and_the_moments_are_half_of_it(self):
        result = the_sixteen_bytes()
        assert result["moments"] == 8

    def test_plain_descent_costs_eight(self):
        assert bytes_per_parameter(SGD).total == 8

    def test_and_momentum_twelve(self):
        assert bytes_per_parameter(MOMENTUM).total == 12

    def test_every_optimiser_is_compared(self):
        assert len(compare_optimizers()) == len(OPTIMIZERS)

    def test_mixed_precision_saves_nothing_on_the_parameters(self):
        # The two bytes saved on the gradient are spent on the second copy of the parameter.
        assert mixed_precision_saves_nothing_on_the_parameters()["identical"]

    def test_an_optimiser_with_negative_state_is_refused(self):
        with pytest.raises(ConfigError, match="cannot keep"):
            OptimizerKind(name="broken", states=-1, flops_per_parameter=1.0)

    def test_one_that_does_negative_work_is_refused(self):
        with pytest.raises(ConfigError, match="negative work"):
            OptimizerKind(name="broken", states=1, flops_per_parameter=-1.0)

    def test_an_empty_breakdown_costs_nothing(self):
        assert MemoryBreakdown().total == 0

    def test_it_serialises(self):
        assert bytes_per_parameter(ADAM).as_dict()["total"] == 16

    def test_the_optimiser_serialises_too(self):
        assert ADAM.as_dict()["states"] == 2

    def test_a_model_with_no_parameters_is_refused(self):
        with pytest.raises(ConfigError, match="needs parameters"):
            training_memory(0)


class TestSharding:
    def test_splitting_the_state_across_devices_helps(self):
        assert sharded_bytes_per_parameter(8) < bytes_per_parameter().total

    def test_and_flattens_once_the_state_is_gone(self):
        # Twelve of the sixteen bytes divide and four do not.
        assert sharding_has_a_floor()["floor"] == 4

    def test_sixty_four_devices_get_close_to_the_floor(self):
        result = sharding_has_a_floor()
        assert result["at_sixty_four"] - result["floor"] < 0.25

    def test_one_device_is_the_unsharded_number(self):
        assert sharded_bytes_per_parameter(1) == bytes_per_parameter().total

    def test_the_sweep_falls_the_whole_way(self):
        rows = [row["bytes_per_parameter"] for row in sharding_sweep()]
        assert rows == sorted(rows, reverse=True)

    def test_zero_devices_are_refused(self):
        with pytest.raises(ConfigError, match="at least one device"):
            sharded_bytes_per_parameter(0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            sharding_sweep(counts=())


class TestUpdateCost:
    def test_the_update_is_memory_bound(self):
        assert update_is_memory_bound()["memory_bound"]

    def test_by_more_than_thirty_to_one(self):
        assert update_is_memory_bound()["ratio"] > 30.0

    def test_so_halving_the_arithmetic_changes_nothing(self):
        assert cheaper_arithmetic_would_not_help()["unchanged"]

    def test_a_model_with_no_parameters_is_refused(self):
        with pytest.raises(ConfigError, match="needs parameters"):
            update_is_memory_bound(parameters=0)

    def test_a_machine_that_does_no_work_is_refused(self):
        with pytest.raises(ConfigError, match="some work per second"):
            update_is_memory_bound(flops_per_second=0)

    def test_narrow_moments_would_save_a_quarter(self):
        assert narrow_moments_would_save()["with_narrow_moments"] == 12

    def test_but_a_narrow_moment_freezes(self):
        # A running average with a decay of a thousandth adds an increment that falls below the
        # last bit of the average long before it has converged.
        result = a_narrow_moment_stalls()
        assert result["froze_at"] > 0

    def test_and_lands_far_from_where_it_should(self):
        assert a_narrow_moment_stalls()["gap"] > 0.1

    def test_a_zero_step_count_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            a_narrow_moment_stalls(steps=0)


class TestActivations:
    def test_activations_overtake_the_weights_immediately(self):
        # A single layer at a batch of thirty two and a sequence of two thousand is twice the
        # whole training state of a hundred million parameters.
        assert where_activations_overtake() == 1

    def test_and_are_almost_all_of_the_memory_at_depth(self):
        assert memory_split()["activation_share"] > 0.98

    def test_the_share_grows_with_the_batch(self):
        shares = [row["activation_share"] for row in batch_sweep()]
        assert shares == sorted(shares)

    def test_even_a_batch_of_one_is_mostly_activations(self):
        rows = {row["batch"]: row for row in batch_sweep()}
        assert rows[1]["activation_share"] > 0.7

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            activation_bytes(batch=0, sequence=8, width=8, layers=2)

    def test_an_empty_batch_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            batch_sweep(sizes=())


class TestUpdate:
    def test_the_update_matches_the_library(self):
        assert matches_the_library()["largest_gap"] == 0.0

    def test_over_several_steps(self):
        assert matches_the_library(steps=10)["relative_gap"] < 1e-6

    def test_a_step_count_starting_at_zero_is_refused(self):
        with pytest.raises(ConfigError, match="starts at one"):
            adam_step(torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.zeros(4), 0)

    def test_a_decay_rate_of_one_is_refused(self):
        with pytest.raises(ConfigError, match="have to be in"):
            adam_step(
                torch.zeros(4),
                torch.zeros(4),
                torch.zeros(4),
                torch.zeros(4),
                1,
                beta_one=1.0,
            )

    def test_a_zero_step_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            matches_the_library(steps=0)

    def test_the_first_moment_moves_toward_the_gradient(self):
        gradient = torch.ones(4)
        _, first, _ = adam_step(torch.zeros(4), gradient, torch.zeros(4), torch.zeros(4), 1)
        assert float(first[0]) == pytest.approx(0.1)

    def test_and_the_second_toward_its_square(self):
        gradient = torch.full((4,), 2.0)
        _, _, second = adam_step(torch.zeros(4), gradient, torch.zeros(4), torch.zeros(4), 1)
        assert float(second[0]) == pytest.approx(0.004)


class TestBiasCorrection:
    def test_it_matters_for_thousands_of_steps(self):
        # Not a short warmup, which is what the name suggests.
        sweep = bias_correction_matters_for_thousands_of_steps()
        rows = {row["step"]: row["ratio"] for row in sweep}
        assert rows[1000] < 0.9

    def test_and_the_worst_point_is_not_the_first_step(self):
        assert the_correction_is_not_monotonic()["worst_is_not_the_first_step"]

    def test_because_the_two_moments_decay_at_different_speeds(self):
        result = the_correction_is_not_monotonic()
        assert result["at_step_twenty"] < result["at_step_one"]

    def test_it_does_converge_eventually(self):
        assert the_correction_is_not_monotonic()["converges_eventually"]

    def test_a_step_of_zero_is_refused(self):
        with pytest.raises(ConfigError, match="starts at one"):
            bias_correction_matters_for_thousands_of_steps(steps=(0,))

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            bias_correction_matters_for_thousands_of_steps(steps=())
