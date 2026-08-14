from __future__ import annotations

from tgc.ir import op as ops
from tgc.ir.builder import Builder, layernorm_graph, softmax_graph
from tgc.ir.dtype import FLOAT16
from tgc.ir.graph import validate
from tgc.passes.cse import (
    duplicate_count,
    eliminate_common_subexpressions,
    find_duplicates,
    is_candidate,
    report_common_subexpressions,
    signature_groups,
)
from tgc.verify.reference import outputs_agree, random_feeds, run


def repeated_graph():
    """The same expression written twice, which a frontend emits constantly."""
    builder = Builder()
    x = builder.input([4, 4], name="x")
    y = builder.input([4, 4], name="y")
    first = builder.mul(x, y)
    second = builder.mul(x, y)
    return builder.finish(builder.add(first, second))


def commuted_graph():
    """The same expression with its operands the other way round."""
    builder = Builder()
    x = builder.input([4, 4], name="x")
    y = builder.input([4, 4], name="y")
    first = builder.add(x, y)
    second = builder.add(y, x)
    return builder.finish(builder.mul(first, second))


def repeated_chain():
    """Two identical chains, which should collapse in one pass rather than needing several."""
    builder = Builder()
    x = builder.input([4, 4], name="x")
    left = builder.relu(builder.exp(x))
    right = builder.relu(builder.exp(x))
    return builder.finish(builder.add(left, right))


class TestCandidates:
    def test_a_pure_op_is_a_candidate(self):
        graph = softmax_graph()
        assert is_candidate(graph.node("v2"))

    def test_a_side_effecting_one_is_not(self):
        # Two prints of the same tensor are two prints.
        builder = Builder()
        x = builder.input([4], name="x")
        printed = builder.apply(ops.PRINT, x)
        graph = builder.finish(printed)
        assert not is_candidate(graph.node(printed))

    def test_a_constant_is_not(self):
        builder = Builder()
        x = builder.input([4], name="x")
        two = builder.constant(2.0)
        graph = builder.finish(builder.mul(x, two))
        assert not is_candidate(graph.node(two))


class TestElimination:
    def test_a_repeated_expression_is_computed_once(self):
        assert len(eliminate_common_subexpressions(repeated_graph()).nodes) == 2

    def test_the_reader_is_rewired_to_the_survivor(self):
        merged = eliminate_common_subexpressions(repeated_graph())
        assert merged.nodes[-1].inputs == ("v0", "v0")

    def test_commuted_operands_still_collide(self):
        # Where most of the wins come from, once a frontend has emitted the same expression
        # with its arguments in whichever order the user wrote them.
        assert len(eliminate_common_subexpressions(commuted_graph()).nodes) == 2

    def test_a_chain_collapses_in_one_pass(self):
        # Once the first duplicate is merged, the nodes reading it become identical to their
        # counterparts, so they merge in the same pass rather than needing another round.
        assert len(eliminate_common_subexpressions(repeated_chain()).nodes) == 3

    def test_a_graph_with_nothing_repeated_is_left_alone(self):
        graph = softmax_graph()
        assert len(eliminate_common_subexpressions(graph).nodes) == len(graph.nodes)

    def test_the_result_still_validates(self):
        for graph in (repeated_graph(), commuted_graph(), repeated_chain()):
            validate(eliminate_common_subexpressions(graph))

    def test_the_answer_does_not_change(self):
        for graph in (repeated_graph(), commuted_graph(), repeated_chain()):
            feeds = random_feeds(graph)
            merged = eliminate_common_subexpressions(graph)
            assert outputs_agree(run(graph, feeds), run(merged, feeds))

    def test_running_it_twice_changes_nothing_the_second_time(self):
        once = eliminate_common_subexpressions(repeated_graph())
        assert len(eliminate_common_subexpressions(once).nodes) == len(once.nodes)

    def test_an_output_that_was_merged_away_is_redirected(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        y = builder.input([4, 4], name="y")
        first = builder.add(x, y)
        second = builder.add(x, y)
        graph = builder.finish(first, second)
        merged = eliminate_common_subexpressions(graph)
        assert merged.outputs == [first, first]
        validate(merged)


class TestSignatures:
    def test_different_axes_do_not_collide(self):
        # A sum over rows and a sum over columns are the same op on the same input and are
        # not the same value.
        builder = Builder()
        x = builder.input([4, 4], name="x")
        rows = builder.sum(x, axes=[0])
        columns = builder.sum(x, axes=[1])
        graph = builder.finish(builder.add(rows, columns))
        assert duplicate_count(graph) == 0

    def test_different_output_types_do_not_collide(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        narrow = builder.cast(x, FLOAT16)
        graph = builder.finish(builder.add(builder.neg(narrow), builder.neg(narrow)))
        assert duplicate_count(graph) == 1

    def test_a_non_commutative_op_keeps_operand_order(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        y = builder.input([4, 4], name="y")
        graph = builder.finish(builder.add(builder.sub(x, y), builder.sub(y, x)))
        assert duplicate_count(graph) == 0

    def test_the_groups_show_why_a_merge_did_not_happen(self):
        builder = Builder()
        x = builder.input([4, 4], name="x")
        rows = builder.sum(x, axes=[0])
        columns = builder.sum(x, axes=[1])
        graph = builder.finish(builder.add(rows, columns))
        groups = signature_groups(graph)
        assert all(len(names) == 1 for names in groups.values())

    def test_identical_nodes_share_a_group(self):
        groups = signature_groups(repeated_graph())
        assert any(len(names) == 2 for names in groups.values())


class TestReporting:
    def test_it_maps_each_duplicate_to_its_survivor(self):
        assert find_duplicates(repeated_graph()) == {"v1": "v0"}

    def test_a_layernorm_has_nothing_to_merge(self):
        assert duplicate_count(layernorm_graph()) == 0

    def test_the_count_matches_the_removal(self):
        graph = repeated_chain()
        before = len(graph.nodes)
        after = len(eliminate_common_subexpressions(graph).nodes)
        assert duplicate_count(graph) == before - after

    def test_it_serialises(self):
        assert report_common_subexpressions(repeated_graph()).as_dict()["merged"] == 1
