from __future__ import annotations

import pytest
import torch

from tgc.errors import ConfigError, GraphError
from tgc.ir.builder import (
    Builder,
    elementwise_chain,
    layernorm_graph,
    mlp_graph,
    softmax_graph,
)
from tgc.ir.dtype import FLOAT16, FLOAT32, INT32, DType
from tgc.verify.reference import (
    interpret,
    largest_difference,
    outputs_agree,
    random_feeds,
    relative_difference,
    run,
    to_torch,
)


class TestTypes:
    def test_it_maps_onto_torch(self):
        assert to_torch(FLOAT32) is torch.float32
        assert to_torch(FLOAT16) is torch.float16

    def test_an_unmappable_type_is_rejected(self):
        with pytest.raises(ConfigError, match="no torch equivalent"):
            to_torch(DType(name="float8", bits=8, kind="float"))


class TestInterpretation:
    def test_a_softmax_graph_produces_a_softmax(self):
        graph = softmax_graph()
        feeds = random_feeds(graph)
        assert torch.allclose(run(graph, feeds)[0], torch.softmax(feeds["x"], dim=1), atol=1e-6)

    def test_a_layernorm_graph_produces_unit_variance(self):
        graph = layernorm_graph()
        result = run(graph, random_feeds(graph))[0]
        assert result.mean(dim=1).abs().max() < 1e-6
        assert torch.allclose(result.std(dim=1, unbiased=False), torch.ones(8), atol=1e-3)

    def test_an_mlp_graph_produces_the_mlp(self):
        graph = mlp_graph()
        feeds = random_feeds(graph)
        expected = torch.relu(feeds["x"] @ feeds["w_up"] + feeds["b_up"]) @ feeds["w_down"]
        assert torch.allclose(run(graph, feeds)[0], expected, atol=1e-5)

    def test_every_intermediate_is_kept(self):
        # Which is what lets a broken pass be caught at the value it broke rather than at
        # the output, where several errors may have cancelled.
        graph = softmax_graph()
        environment = interpret(graph, random_feeds(graph))
        assert set(environment) == graph.value_names

    def test_a_missing_input_is_reported(self):
        with pytest.raises(GraphError, match="no value supplied"):
            run(softmax_graph(), {})

    def test_an_input_of_the_wrong_type_is_reported(self):
        graph = softmax_graph()
        feeds = random_feeds(graph)
        feeds["x"] = feeds["x"].to(torch.float64)
        with pytest.raises(GraphError, match="was declared"):
            run(graph, feeds)

    def test_a_symbolic_graph_cannot_be_filled(self):
        with pytest.raises(GraphError, match="symbolic shape"):
            random_feeds(mlp_graph(batch="batch"))


class TestOperations:
    def test_a_chain_applies_in_order(self):
        graph = elementwise_chain(2, sizes=(4,))
        feeds = random_feeds(graph)
        assert torch.allclose(run(graph, feeds)[0], torch.relu(torch.exp(feeds["x"])))

    def test_a_reduction_widens_before_accumulating(self):
        # Summing float16 in float16 stalls once the running total is large enough that each
        # addend falls below its last bit. The compiler says the sum is float32 and the
        # interpreter has to agree, or nothing downstream can be checked against it.
        builder = Builder()
        x = builder.input([4096], dtype=FLOAT16, name="x")
        graph = builder.finish(builder.sum(x, axes=[0]))
        feeds = {"x": torch.full((4096,), 1.0, dtype=torch.float16)}
        assert run(graph, feeds)[0].dtype is torch.float32
        assert run(graph, feeds)[0].item() == 4096.0

    def test_a_narrow_accumulator_would_have_stalled(self):
        # The measurement the widening exists to avoid, done directly rather than asserted.
        values = torch.full((4096,), 1.0, dtype=torch.float16)
        naive = torch.zeros((), dtype=torch.float16)
        for value in values:
            naive = naive + value
        assert naive.item() < 4096.0

    def test_a_maximum_keeps_its_type(self):
        builder = Builder()
        x = builder.input([4, 8], dtype=FLOAT16)
        graph = builder.finish(builder.max(x, axes=[1]))
        assert run(graph, random_feeds(graph))[0].dtype is torch.float16

    def test_a_transpose_moves_the_axes(self):
        builder = Builder()
        x = builder.input([2, 3, 4])
        graph = builder.finish(builder.transpose(x, [2, 0, 1]))
        assert tuple(run(graph, random_feeds(graph))[0].shape) == (4, 2, 3)

    def test_a_reshape_keeps_the_elements(self):
        builder = Builder()
        x = builder.input([4, 8], name="x")
        graph = builder.finish(builder.reshape(x, [32]))
        feeds = random_feeds(graph)
        assert torch.equal(run(graph, feeds)[0], feeds["x"].reshape(32))

    def test_a_broadcast_repeats(self):
        builder = Builder()
        x = builder.input([1, 8], name="x")
        graph = builder.finish(builder.broadcast_to(x, [4, 8]))
        feeds = random_feeds(graph)
        assert torch.equal(run(graph, feeds)[0], feeds["x"].expand(4, 8))

    def test_a_constant_holds_its_value(self):
        builder = Builder()
        x = builder.input([4], name="x")
        two = builder.constant(2.0)
        graph = builder.finish(builder.mul(x, two))
        feeds = random_feeds(graph)
        assert torch.allclose(run(graph, feeds)[0], feeds["x"] * 2)

    def test_an_integer_graph_stays_integral(self):
        builder = Builder()
        x = builder.input([4], dtype=INT32)
        graph = builder.finish(builder.add(x, x))
        assert run(graph, random_feeds(graph))[0].dtype is torch.int32


class TestFeeds:
    def test_the_same_seed_gives_the_same_inputs(self):
        graph = softmax_graph()
        first = random_feeds(graph, seed=3)
        second = random_feeds(graph, seed=3)
        assert torch.equal(first["x"], second["x"])

    def test_a_different_seed_does_not(self):
        graph = softmax_graph()
        assert not torch.equal(random_feeds(graph, seed=1)["x"], random_feeds(graph)["x"])

    def test_positive_inputs_are_safe_for_logarithms(self):
        # Otherwise every comparison becomes nan against nan, which compares unequal, and
        # the failure says nothing about the transformation being tested.
        graph = softmax_graph()
        assert (random_feeds(graph, positive=True)["x"] > 0).all()

    def test_the_type_matches_the_declaration(self):
        builder = Builder()
        builder.input([4], dtype=FLOAT16, name="a")
        builder.input([4], dtype=INT32, name="b")
        graph = builder.finish(builder.neg("a"))
        feeds = random_feeds(graph)
        assert feeds["a"].dtype is torch.float16
        assert feeds["b"].dtype is torch.int32


class TestComparison:
    def test_two_identical_runs_agree_exactly(self):
        graph = softmax_graph()
        feeds = random_feeds(graph)
        assert outputs_agree(run(graph, feeds), run(graph, feeds))

    def test_different_answers_do_not(self):
        graph = softmax_graph()
        first = run(graph, random_feeds(graph, seed=1))
        second = run(graph, random_feeds(graph, seed=2))
        assert not outputs_agree(first, second)

    def test_a_tolerance_forgives_a_small_gap(self):
        left = [torch.tensor([1.0, 2.0])]
        right = [torch.tensor([1.0, 2.0 + 1e-5])]
        assert not outputs_agree(left, right)
        assert outputs_agree(left, right, tolerance=1e-3)

    def test_different_shapes_never_agree(self):
        assert not outputs_agree([torch.zeros(4)], [torch.zeros(5)])

    def test_different_counts_never_agree(self):
        assert not outputs_agree([torch.zeros(4)], [torch.zeros(4), torch.zeros(4)])

    def test_the_largest_difference_is_reported(self):
        left = [torch.tensor([1.0, 2.0])]
        right = [torch.tensor([1.0, 2.5])]
        assert largest_difference(left, right) == pytest.approx(0.5)

    def test_the_relative_difference_scales_by_the_values(self):
        left = [torch.tensor([100.0, 200.0])]
        right = [torch.tensor([100.0, 200.5])]
        assert relative_difference(left, right) == pytest.approx(0.0025)

    def test_comparing_different_counts_is_rejected(self):
        with pytest.raises(ConfigError, match="different numbers of outputs"):
            largest_difference([torch.zeros(4)], [])

    def test_an_all_zero_reference_falls_back_to_the_absolute_gap(self):
        assert relative_difference([torch.zeros(4)], [torch.ones(4)]) == 1.0
