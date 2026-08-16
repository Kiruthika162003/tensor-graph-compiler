from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError, PassError
from tgc.parallel.shard import (
    PLANS,
    REPLICATED,
    ShardSpec,
    Traffic,
    a_bigger_product_scales_further,
    arithmetic_per_device,
    check_plan,
    column_parallel,
    compare_plans,
    contraction_parallel,
    even_an_exact_split_is_not_bit_identical,
    every_plan_computes_the_same_product,
    measure_traffic,
    mlp_needs_one_all_reduce,
    only_column_parallel_shrinks_the_weight,
    only_the_reduction_loses_a_bit,
    replicated,
    resharding_cost,
    resharding_table,
    row_parallel,
    run_plan,
    run_sharded_mlp,
    scaling_sweep,
    shard,
    staying_put_is_free,
    the_composition_holds_at_every_device_count,
    the_relu_is_why_it_works,
    where_communication_takes_over,
)


class TestSpec:
    def test_a_sharded_axis_is_divided(self):
        assert ShardSpec(axis=0, devices=4).shard_shape([64, 32]) == [16, 32]

    def test_a_replicated_tensor_is_not(self):
        assert replicated(4).shard_shape([64, 32]) == [64, 32]

    def test_the_pieces_add_back_up(self):
        spec = ShardSpec(axis=1, devices=4)
        assert spec.elements_per_device([64, 32]) * 4 == 64 * 32

    def test_an_axis_that_does_not_divide_is_refused(self):
        with pytest.raises(PassError, match="does not divide evenly"):
            ShardSpec(axis=0, devices=3).shard_shape([64, 32])

    def test_an_axis_beyond_the_rank_is_refused(self):
        with pytest.raises(PassError, match="cannot shard axis"):
            ShardSpec(axis=2, devices=2).shard_shape([64, 32])

    def test_zero_devices_are_refused(self):
        with pytest.raises(ConfigError, match="at least one device"):
            ShardSpec(axis=0, devices=0)

    def test_a_negative_axis_below_replicated_is_refused(self):
        with pytest.raises(ConfigError, match="is not an axis"):
            ShardSpec(axis=REPLICATED - 1, devices=2)

    def test_it_serialises(self):
        assert replicated(4).as_dict()["axis"] == "replicated"

    def test_splitting_gives_one_piece_per_device(self):
        assert len(shard(torch.randn(64, 32), ShardSpec(axis=0, devices=4))) == 4

    def test_replicating_gives_the_whole_thing_to_everyone(self):
        pieces = shard(torch.randn(64, 32), replicated(4))
        assert all(list(piece.shape) == [64, 32] for piece in pieces)


class TestCorrectness:
    def test_every_plan_computes_the_same_product(self):
        assert every_plan_computes_the_same_product()

    def test_every_plan_gives_the_right_shape(self):
        assert all(row["shape_matches"] for row in check_plan())

    def test_column_parallel_is_bit_identical(self):
        assert only_the_reduction_loses_a_bit()["column_parallel_exact"]

    def test_the_all_reduce_is_not(self):
        # Summing four partial sums adds the terms in a different order than one contraction.
        assert not only_the_reduction_loses_a_bit()["contraction_parallel_exact"]

    def test_and_row_parallel_is_not_either(self):
        # Which is the surprise. It performs the same arithmetic in the same groups.
        assert not only_the_reduction_loses_a_bit()["row_parallel_exact"]

    def test_the_row_parallel_gap_survives_double_precision(self):
        # So it is not a rounding budget being exceeded, it is the library blocking a shorter
        # matrix differently.
        rows = even_an_exact_split_is_not_bit_identical()
        assert not rows["float64"]["bit_identical"]

    def test_but_shrinks_to_the_rounding_unit_of_the_type(self):
        rows = even_an_exact_split_is_not_bit_identical()
        assert rows["float64"]["largest_gap"] < rows["float32"]["largest_gap"] / 1e6

    def test_a_single_device_plan_is_the_ordinary_product(self):
        left = torch.randn(8, 16)
        right = torch.randn(16, 8)
        assert torch.equal(run_plan(left, right, row_parallel(1)), left @ right)

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            check_plan(rows=0)


class TestTraffic:
    def test_the_two_gather_free_plans_move_nothing(self):
        rows = {row["plan"]: row for row in compare_plans()}
        assert rows["row parallel"]["communicated"] == 0
        assert rows["column parallel"]["communicated"] == 0

    def test_the_all_reduce_moves_twice_the_output(self):
        traffic = measure_traffic(contraction_parallel(2), 32, 64, 48)
        assert traffic.communicated_elements == 32 * 48

    def test_splitting_the_batch_does_not_make_a_model_smaller(self):
        # However many devices it is split over.
        assert not only_column_parallel_shrinks_the_weight()["row parallel"]

    def test_splitting_the_weight_does(self):
        result = only_column_parallel_shrinks_the_weight()
        assert result["column parallel"]
        assert result["contraction parallel"]

    def test_a_traffic_report_adds_up_what_it_stores(self):
        traffic = Traffic(
            plan="test", weight_elements=10, activation_elements=5, communicated_elements=0
        )
        assert traffic.stored == 15

    def test_it_serialises(self):
        assert measure_traffic(row_parallel(4)).as_dict()["plan"] == "row parallel"

    def test_zero_devices_are_refused(self):
        with pytest.raises(ConfigError, match="at least one device"):
            compare_plans(devices=0)

    def test_three_plans_are_compared(self):
        assert len(compare_plans()) == len(PLANS)


class TestComposition:
    def test_the_composed_mlp_needs_one_all_reduce(self):
        # The piece each device holds after the first product is exactly the piece it needs
        # for the second, so the intermediate never has to be gathered.
        result = mlp_needs_one_all_reduce()
        assert result["composed"] < result["with_a_gather_in_the_middle"]

    def test_and_the_gather_would_cost_four_times_as_much_again(self):
        assert mlp_needs_one_all_reduce()["ratio"] == 5.0

    def test_it_computes_the_same_thing_as_the_unsharded_version(self):
        assert run_sharded_mlp()["relative_gap"] < 1e-5

    def test_at_every_device_count(self):
        assert all(row["agrees"] for row in the_composition_holds_at_every_device_count())

    def test_and_keeps_the_shape(self):
        assert all(
            row["shape_matches"] for row in the_composition_holds_at_every_device_count()
        )

    def test_a_relu_commutes_with_a_column_split(self):
        # Which is why the nonlinearity can happen before anything is gathered.
        assert the_relu_is_why_it_works()["relu_commutes"]

    def test_a_softmax_does_not(self):
        # Its reduction spans the axis the sharding would split, which is why attention is
        # sharded by head instead.
        result = the_relu_is_why_it_works()
        assert result["softmax_does_not"]
        assert result["softmax_gap"] > 0.1

    def test_a_width_that_does_not_divide_is_refused(self):
        with pytest.raises(ConfigError, match="does not divide"):
            run_sharded_mlp(hidden=8, expansion=1, devices=3)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            the_composition_holds_at_every_device_count(counts=())


class TestScaling:
    def test_the_arithmetic_falls_with_the_device_count(self):
        rows = scaling_sweep()
        assert rows[-1]["arithmetic"] < rows[0]["arithmetic"]

    def test_the_communication_flattens_rather_than_falling(self):
        rows = scaling_sweep()
        assert rows[-1]["communication_cost"] > rows[1]["communication_cost"]

    def test_so_the_link_takes_over_eventually(self):
        assert scaling_sweep()[-1]["communication_share"] > 0.9

    def test_a_single_device_communicates_nothing(self):
        assert scaling_sweep()[0]["communication_share"] == 0.0

    def test_the_crossover_is_findable(self):
        assert where_communication_takes_over() > 1

    def test_a_bigger_product_scales_further(self):
        # The arithmetic grows with three dimensions and the communication with two.
        rows = a_bigger_product_scales_further()
        assert rows[-1]["crossover"] > rows[0]["crossover"]

    def test_the_arithmetic_is_divided_evenly(self):
        assert arithmetic_per_device(8, 8, 8, 4) == arithmetic_per_device(8, 8, 8, 1) / 4

    def test_zero_devices_are_refused(self):
        with pytest.raises(ConfigError, match="at least one device"):
            arithmetic_per_device(8, 8, 8, 0)

    def test_a_zero_link_ratio_is_refused(self):
        with pytest.raises(ConfigError, match="has to be positive"):
            scaling_sweep(link_ratio=0.0)

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            scaling_sweep(counts=())


class TestResharding:
    def test_staying_in_the_same_layout_is_free(self):
        spec = ShardSpec(axis=0, devices=4)
        assert resharding_cost(spec, spec, [64, 64]) == 0

    def test_and_nothing_else_is(self):
        result = staying_put_is_free()
        assert result["free"] == 3
        assert result["all_free_ones_are_no_ops"]

    def test_changing_the_split_axis_moves_the_whole_tensor(self):
        source = ShardSpec(axis=0, devices=4)
        target = ShardSpec(axis=1, devices=4)
        assert resharding_cost(source, target, [64, 64]) == 64 * 64

    def test_a_mismatched_device_count_is_refused(self):
        with pytest.raises(PassError, match="cannot reshard between"):
            resharding_cost(replicated(2), replicated(4), [8, 8])

    def test_every_pair_is_costed(self):
        assert len(resharding_table()) == 9

    def test_zero_devices_are_refused(self):
        with pytest.raises(ConfigError, match="at least one device"):
            resharding_table(devices=0)

    def test_the_named_plans_use_the_layouts_in_the_table(self):
        axes = {plan(4).right.as_dict()["axis"] for plan in (row_parallel, column_parallel)}
        assert axes == {"replicated", 1}
