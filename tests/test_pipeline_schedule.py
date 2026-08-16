from __future__ import annotations

import pytest

from tgc.errors import ConfigError
from tgc.schedule.pipeline import (
    PipelineShape,
    Timeline,
    a_pipeline_of_one_stage_has_no_bubble,
    a_single_microbatch_is_almost_all_bubble,
    a_slower_backward_raises_the_ceiling,
    activations_in_flight,
    compare_schedules,
    completions,
    dependencies_are_respected,
    every_task_runs_exactly_once,
    halving_the_bubble_doubles_the_memory,
    interleaving_helps_the_memory,
    measured_bubble,
    memory_against_bubble,
    microbatch_sweep,
    more_devices_make_it_worse,
    predicted_bubble,
    simulate,
    stage_sweep,
    the_bubble_falls_like_one_over_the_count,
    the_first_stage_holds_the_most,
    the_formula_assumes_equal_passes,
    the_memory_saturates,
)


class TestShape:
    def test_the_work_is_every_task_on_every_stage(self):
        shape = PipelineShape(stages=4, microbatches=8, backward_slots=1)
        assert shape.work_slots == 4 * 8 * 2

    def test_a_pipeline_with_no_stages_is_refused(self):
        with pytest.raises(ConfigError, match="needs stages"):
            PipelineShape(stages=0, microbatches=4)

    def test_one_with_no_microbatches_is_refused(self):
        with pytest.raises(ConfigError, match="needs microbatches"):
            PipelineShape(stages=4, microbatches=0)

    def test_a_pass_that_takes_no_time_is_refused(self):
        with pytest.raises(ConfigError, match="at least one slot"):
            PipelineShape(stages=4, microbatches=4, forward_slots=0)

    def test_it_serialises(self):
        assert PipelineShape(stages=4, microbatches=8).as_dict()["stages"] == 4

    def test_an_empty_timeline_has_no_bubble(self):
        assert Timeline().bubble == 0.0


class TestSimulation:
    def test_every_task_runs_exactly_once(self):
        # A schedule that dropped a task would report a smaller bubble and a shorter timeline.
        assert every_task_runs_exactly_once()["every_task_once"]

    def test_no_stage_starts_before_its_input_is_ready(self):
        assert dependencies_are_respected()["violations"] == 0

    def test_a_single_stage_never_idles(self):
        assert a_pipeline_of_one_stage_has_no_bubble()["bubble"] == 0.0

    def test_and_takes_exactly_its_work(self):
        result = a_pipeline_of_one_stage_has_no_bubble()
        assert result["length"] == 8 * 3

    def test_the_timeline_covers_every_device(self):
        timeline = simulate(PipelineShape(stages=4, microbatches=4))
        assert timeline.devices == 4

    def test_the_busy_slots_are_the_work(self):
        shape = PipelineShape(stages=3, microbatches=4, backward_slots=1)
        assert simulate(shape).busy_slots == shape.work_slots

    def test_completions_are_read_off_the_timeline(self):
        shape = PipelineShape(stages=3, microbatches=4, backward_slots=1)
        assert len(completions(simulate(shape))) == 3 * 4 * 2

    def test_it_serialises(self):
        assert simulate(PipelineShape(2, 2)).as_dict()["devices"] == 2


class TestBubble:
    def test_the_closed_form_is_exact(self):
        # Not an approximation that happens to be close: it is what a schedule that never idles
        # a ready device produces.
        result = the_formula_assumes_equal_passes()
        assert result["formula"] == result["forward_first"]

    def test_for_both_orderings(self):
        result = the_formula_assumes_equal_passes()
        assert result["forward_first"] == result["backward_preferring"]

    def test_and_for_a_slower_backward_pass(self):
        result = the_formula_assumes_equal_passes()
        assert result["with_a_slower_backward"] == result["formula"]

    def test_one_microbatch_leaves_every_other_device_idle(self):
        result = a_single_microbatch_is_almost_all_bubble()
        assert result["bubble"] == result["predicted"]

    def test_the_bubble_falls_as_the_batch_is_cut_finer(self):
        result = the_bubble_falls_like_one_over_the_count()
        assert result["roughly_halving"]

    def test_and_never_reaches_zero(self):
        rows = microbatch_sweep()
        assert all(row["bubble"] > 0.0 for row in rows)

    def test_more_devices_make_it_worse(self):
        # Adding a device makes the pipeline longer to fill and drain.
        assert more_devices_make_it_worse()["worse_with_more_devices"]

    def test_by_a_lot(self):
        result = more_devices_make_it_worse()
        assert result["at_sixteen_stages"] > 5 * result["at_two_stages"]

    def test_the_prediction_matches_the_measurement(self):
        shape = PipelineShape(stages=4, microbatches=8, backward_slots=1)
        assert round(predicted_bubble(shape), 4) == round(measured_bubble(shape), 4)

    def test_an_empty_microbatch_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            microbatch_sweep(counts=())

    def test_an_empty_stage_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            stage_sweep(counts=())


class TestMemory:
    def test_a_forward_first_schedule_grows_with_the_microbatches(self):
        rows = {row["microbatches"]: row for row in memory_against_bubble()}
        assert rows[32]["peak_forward_first"] == 32

    def test_a_backward_preferring_one_saturates(self):
        assert the_memory_saturates()["saturates"]

    def test_at_two_stages_less_one(self):
        assert the_memory_saturates()["matches_the_ceiling"]

    def test_a_slower_backward_raises_that_ceiling(self):
        result = a_slower_backward_raises_the_ceiling()
        assert result["slower_backward_ceiling"] > result["equal_passes_ceiling"]

    def test_but_it_still_saturates(self):
        assert a_slower_backward_raises_the_ceiling()["both_saturate"]

    def test_the_first_stage_holds_the_most(self):
        # It starts every microbatch first and releases each one last.
        assert the_first_stage_holds_the_most()["front_heavy"]

    def test_and_the_last_stage_holds_one(self):
        assert the_first_stage_holds_the_most()["last"] == 1

    def test_interleaving_costs_no_throughput(self):
        assert interleaving_helps_the_memory()["same_bubble"]

    def test_the_trade_is_one_for_one_only_for_the_naive_schedule(self):
        result = halving_the_bubble_doubles_the_memory()
        assert (
            result["forward_first_memory_at_sixteen"]
            == 4 * result["forward_first_memory_at_four"]
        )

    def test_while_the_other_one_barely_moves(self):
        result = halving_the_bubble_doubles_the_memory()
        assert (
            result["interleaved_memory_at_sixteen"] < 2 * result["interleaved_memory_at_four"]
        )

    def test_the_peak_is_reported_per_stage(self):
        assert len(activations_in_flight(PipelineShape(4, 8))) == 4

    def test_an_empty_memory_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            memory_against_bubble(counts=())

    def test_four_schedules_are_compared(self):
        assert len(compare_schedules()) == 4

    def test_and_the_finest_one_has_the_smallest_bubble(self):
        rows = {row["schedule"]: row for row in compare_schedules()}
        assert rows["many microbatches"]["bubble"] < rows["one microbatch"]["bubble"]
