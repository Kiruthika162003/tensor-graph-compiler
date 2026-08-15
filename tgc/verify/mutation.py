from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.graph import Graph, Node, reachable_from_outputs, validate
from tgc.verify.fuzz import generate_many
from tgc.verify.reference import outputs_agree, random_feeds, run

# Breaking the passes on purpose, to find out what the tests would have caught.
#
# A test suite that passes says nothing on its own. It says something once the passes have been
# broken in the ways a person would plausibly break them and the suite is asked again. Every
# mutant below is a pass with one specific mistake in it, and every mistake is one that has
# actually shipped somewhere: a subexpression pass that ignores attributes, a transpose
# canceller that assumes any two transposes undo each other, a dead code pass that treats a
# print like any other node.
#
# The result is the reason this file exists. Comparing outputs on random inputs, which is what
# almost every compiler test does, catches four of the six. It misses the dead code one
# entirely, because a print returns its input and deleting it leaves every number where it was,
# and no amount of comparing values will ever find that. It misses the multiply by zero rule for
# a different reason: the rule is correct on every number a random generator produces and wrong
# only on an infinity, so the check has to be handed one.
#
# Two of the four it does catch were only caught after the fixtures grew. The transpose mutant
# is not a mistake at rank two and every fixture here was a matrix; the subexpression mutant
# needs two nodes with one op, one input and different attributes, and no fixture had a pair.
# Both were invisible for the same reason, which is not that the check was weak but that
# nothing it ran on contained the thing the pass is about.
#
# Fuzzing does not rescue either of them. Four of the six mutants never change the answer on a
# generated graph at all and the best of the rest shows up on a quarter of them.

Transform = Callable[[Graph], Graph]


@dataclass
class Mutant:
    """A pass with one deliberate mistake in it."""

    name: str
    transform: Transform
    mistake: str

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"mutant": self.name, "mistake": self.mistake}


def dce_that_forgets_side_effects(graph: Graph) -> Graph:
    """Dead code elimination seeded only from the outputs.

    The one line version everybody writes first. A print whose value nobody reads is not dead,
    because printing is why it is there, and the author of the print has no way to tell that a
    compiler pass is why it stopped happening.
    """
    live = set(reachable_from_outputs(graph))
    changed = True
    while changed:
        changed = False
        for node in reversed(graph.nodes):
            if node.name in live:
                for name in node.inputs:
                    if name not in live:
                        live.add(name)
                        changed = True
    return graph.with_nodes([node for node in graph.nodes if node.name in live])


def cse_that_ignores_attributes(graph: Graph) -> Graph:
    """Subexpression elimination keyed on the op and its inputs only.

    A sum over axis zero and a sum over axis one are the same op reading the same value, and
    they are not the same value. Leaving the attributes out of the key makes the second one
    disappear into the first.
    """
    seen: dict[tuple, str] = {}
    mapping: dict[str, str] = {}
    rewritten: list[Node] = []
    for node in graph.nodes:
        updated = node.replace_inputs(mapping)
        key = (updated.op.name, updated.inputs)
        if updated.op.can_be_removed_if_unused and key in seen:
            mapping[node.name] = seen[key]
            continue
        seen[key] = node.name
        rewritten.append(updated)
    return graph.with_nodes(rewritten)


def transposes_that_always_cancel(graph: Graph) -> Graph:
    """Transpose cancellation that does not check the permutations.

    True at rank two, where the only permutation worth writing is its own inverse, and false at
    every higher rank. A fixture set made of matrices cannot tell the difference, which is
    exactly why this one survives the usual checks.
    """
    mapping: dict[str, str] = {}
    rewritten: list[Node] = []
    for node in graph.nodes:
        updated = node.replace_inputs(mapping)
        if updated.op is ops.TRANSPOSE:
            inner = graph.producer_of(node.inputs[0])
            if inner is not None and inner.op is ops.TRANSPOSE:
                mapping[node.name] = mapping.get(inner.inputs[0], inner.inputs[0])
                continue
        rewritten.append(updated)
    return graph.with_nodes(rewritten)


def folding_that_ignores_overflow(graph: Graph) -> Graph:
    """Constant folding that keeps the result whatever it came out as.

    Folding at compile time in double precision and storing the answer as a float32 literal.
    The graph would have produced an infinity at runtime and now produces the largest finite
    float instead, which is a different number and a much harder bug to find later.
    """
    rewritten: list[Node] = []
    values: dict[str, float] = {}
    for node in graph.nodes:
        if node.op is ops.CONSTANT:
            values[node.name] = float(node.attrs["value"])
            rewritten.append(node)
            continue
        if node.op is ops.MUL and all(name in values for name in node.inputs):
            product = values[node.inputs[0]] * values[node.inputs[1]]
            largest = 3.4028234663852886e38
            clamped = math.copysign(largest, product) if math.isinf(product) else product
            values[node.name] = clamped
            rewritten.append(
                Node(
                    op=ops.CONSTANT,
                    inputs=(),
                    output=node.output,
                    attrs={"value": clamped},
                )
            )
            continue
        rewritten.append(node)
    return graph.with_nodes(rewritten)


def multiply_by_zero_is_zero(graph: Graph) -> Graph:
    """The algebraic rule that is true in arithmetic and false in floating point.

    Zero times an infinity is a nan and zero times a nan is a nan, so replacing the product
    with a literal zero changes the answer exactly where somebody is already debugging.
    """
    rewritten: list[Node] = []
    for node in graph.nodes:
        zero_name = _zero_operand(graph, node)
        if zero_name:
            rewritten.append(
                Node(
                    op=ops.BROADCAST_TO,
                    inputs=(zero_name,),
                    output=node.output,
                    attrs={"shape": node.output.shape},
                )
            )
            continue
        rewritten.append(node)
    return graph.with_nodes(rewritten)


def _zero_operand(graph: Graph, node: Node) -> str:
    """The name of a literal zero this node multiplies by, if there is one.

    Written as a broadcast of the existing literal rather than as a fresh scalar constant,
    because a scalar output where a matrix was expected is a shape error and would be caught by
    the validator rather than by the numbers. The mutant is only interesting if the graph it
    produces is a legal graph that computes the wrong thing.
    """
    if node.op is not ops.MUL:
        return ""
    for name in node.inputs:
        producer = graph.producer_of(name)
        if producer is not None and producer.op is ops.CONSTANT:
            zero = float(producer.attrs["value"]) == 0.0
            if zero:
                return name
    return ""


def reductions_that_lose_an_axis(graph: Graph) -> Graph:
    """A rewrite that changes which axis a reduction runs over.

    Stands in for the whole family of off by one mistakes in axis handling. It is included
    because it is the easiest mutant to catch and its being easy is the point of comparison for
    the ones that are not.
    """
    rewritten: list[Node] = []
    for node in graph.nodes:
        if node.op.category == ops.REDUCTION and node.attrs.get("axes"):
            axes = tuple(int(axis) for axis in node.attrs["axes"])
            shifted = tuple(max(axis - 1, 0) for axis in axes)
            attrs = dict(node.attrs)
            attrs["axes"] = shifted
            rewritten.append(
                Node(op=node.op, inputs=node.inputs, output=node.output, attrs=attrs)
            )
            continue
        rewritten.append(node)
    return graph.with_nodes(rewritten)


MUTANTS = (
    Mutant(
        name="dce keeps no side effects",
        transform=dce_that_forgets_side_effects,
        mistake="seeded liveness from the outputs only",
    ),
    Mutant(
        name="cse ignores attributes",
        transform=cse_that_ignores_attributes,
        mistake="keyed the signature on the op and inputs only",
    ),
    Mutant(
        name="transposes always cancel",
        transform=transposes_that_always_cancel,
        mistake="assumed any two transposes undo each other",
    ),
    Mutant(
        name="folding clamps an overflow",
        transform=folding_that_ignores_overflow,
        mistake="stored the largest finite float instead of the infinity",
    ),
    Mutant(
        name="multiply by zero is zero",
        transform=multiply_by_zero_is_zero,
        mistake="applied a rule that is false for infinities and nans",
    ),
    Mutant(
        name="reduction axis off by one",
        transform=reductions_that_lose_an_axis,
        mistake="shifted every reduction axis down by one",
    ),
)


def fixtures() -> list[tuple[str, Graph]]:
    """The graphs the usual checks run over."""
    return [
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ]


def caught_by_outputs(transform: Transform, graph: Graph, *, seed: int = 0) -> bool:
    """Whether comparing outputs on random inputs notices a mutant.

    The check almost every compiler test is. It runs the original and the transformed graph on
    the same random inputs and compares bit for bit, and it counts a crash or an invalid graph
    as a catch, because both are the pass being noticed.
    """
    feeds = random_feeds(graph, positive=True, seed=seed)
    try:
        mutated = transform(graph)
        validate(mutated)
    except Exception:
        return True
    try:
        return not outputs_agree(run(graph, feeds), run(mutated, feeds))
    except Exception:
        return True


def caught_by_side_effects(transform: Transform, graph: Graph) -> bool:
    """Whether the side effects in a graph survived the pass.

    The check that the value comparison cannot make. A print returns its input unchanged, so
    deleting it leaves every output exactly where it was, and no amount of comparing numbers
    will ever notice.
    """
    before = _effect_names(graph)
    if not before:
        return False
    try:
        after = _effect_names(transform(graph))
    except Exception:
        return True
    return before != after


def _effect_names(graph: Graph) -> set[str]:
    """Every node in a graph that exists for something other than its value."""
    return {node.name for node in graph.nodes if not node.op.can_be_removed_if_unused}


def caught_by_extreme_inputs(transform: Transform, graph: Graph) -> bool:
    """Whether a mutant shows up on inputs a random generator does not produce.

    Infinities and nans, which is where the arithmetic identities stop holding. Every rule in
    passes/algebraic.py that is listed as inexact is inexact here and nowhere else, and a suite
    that only ever feeds it ordinary numbers is testing the easy half.
    """
    feeds = random_feeds(graph, positive=True)
    for name, tensor in feeds.items():
        extreme = tensor.clone()
        flat = extreme.flatten()
        if flat.numel() >= 2:
            flat[0] = float("inf")
            flat[1] = float("nan")
        feeds[name] = flat.reshape(tensor.shape)
    try:
        mutated = transform(graph)
        validate(mutated)
        before = run(graph, feeds)
        after = run(mutated, feeds)
    except Exception:
        return True
    return not _agree_with_nans(before, after)


def _agree_with_nans(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> bool:
    """Bit equality that treats a nan in the same place as agreement."""
    if len(left) != len(right):
        return False
    for first, second in zip(left, right, strict=True):
        if first.shape != second.shape:
            return False
        both_nan = first.isnan() & second.isnan()
        if not bool((torch.eq(first, second) | both_nan).all()):
            return False
    return True


def rank_three_graph(size: int = 4) -> Graph:
    """A graph whose transposes do not undo each other.

    Two rotations of three axes compose into a third rotation rather than into the identity.
    Nothing in the fixture set is above rank two, so nothing in the fixture set can tell a
    transpose canceller that checks permutations from one that does not.
    """
    if size < 1:
        raise ConfigError(f"the size has to be positive, got {size}")
    builder = Builder()
    x = builder.input([size, size + 1, size + 2], name="x")
    once = builder.transpose(x, [1, 2, 0])
    twice = builder.transpose(once, [1, 2, 0])
    return builder.finish(builder.relu(twice))


def graph_with_a_side_effect() -> Graph:
    """A graph holding a print whose value nothing reads."""
    builder = Builder()
    x = builder.input([8, 8], name="x")
    scaled = builder.mul(x, builder.constant(2.0))
    builder.apply(ops.PRINT, scaled)
    return builder.finish(builder.relu(scaled))


def overflowing_graph() -> Graph:
    """A literal chain whose product does not fit in a float32."""
    builder = Builder()
    x = builder.input([4, 4], name="x")
    big = builder.mul(builder.constant(1e30), builder.constant(1e30))
    return builder.finish(builder.mul(x, big))


def two_reductions_of_one_value(rows: int = 8, columns: int = 8) -> Graph:
    """One value reduced along two different axes.

    Added after the fact, and the reason is the interesting part. The subexpression mutant
    survived every check in this file on the fixtures it was given, not because the checks were
    weak but because no fixture contained two nodes with the same op and the same input and
    different attributes. There was nothing there for it to break.
    """
    if min(rows, columns) < 1:
        raise ConfigError("the shape has to be positive")
    builder = Builder()
    x = builder.input([rows, columns], name="x")
    down = builder.sum(x, axes=[0], keepdims=True)
    across = builder.sum(x, axes=[1], keepdims=True)
    wide_down = builder.broadcast_to(down, [rows, columns])
    wide_across = builder.broadcast_to(across, [rows, columns])
    return builder.finish(builder.mul(wide_down, wide_across))


def graph_with_a_zero(rows: int = 8) -> Graph:
    """A product against a literal zero, for the rule that is false in floating point."""
    builder = Builder()
    x = builder.input([rows, rows], name="x")
    return builder.finish(builder.mul(x, builder.constant(0.0)))


@dataclass
class MutationReport:
    """Which checks caught which mutants."""

    rows: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Mutants tried."""
        return len(self.rows)

    @property
    def caught_by_the_usual_check(self) -> int:
        """Mutants a random output comparison notices."""
        return sum(1 for row in self.rows if row["outputs"])

    @property
    def survivors(self) -> list[str]:
        """Mutants the usual check misses."""
        return [row["mutant"] for row in self.rows if not row["outputs"]]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "mutants": self.total,
            "caught_by_outputs": self.caught_by_the_usual_check,
            "survivors": self.survivors,
        }


def run_all_checks(mutant: Mutant) -> dict:
    """Every check against one mutant, over every fixture it applies to.

    A mutant counts as caught by a check if that check notices it on any graph. Requiring every
    graph would be measuring the fixtures rather than the check.
    """
    graphs = [graph for _, graph in fixtures()]
    graphs.extend(
        [
            rank_three_graph(),
            graph_with_a_side_effect(),
            overflowing_graph(),
            graph_with_a_zero(),
            two_reductions_of_one_value(),
        ]
    )
    return {
        "mutant": mutant.name,
        "mistake": mutant.mistake,
        "outputs": any(caught_by_outputs(mutant.transform, item) for item in graphs),
        "side_effects": any(caught_by_side_effects(mutant.transform, item) for item in graphs),
        "extremes": any(caught_by_extreme_inputs(mutant.transform, item) for item in graphs),
    }


def mutation_report() -> MutationReport:
    """Every mutant against every check."""
    return MutationReport(rows=[run_all_checks(mutant) for mutant in MUTANTS])


def which_check_is_needed() -> list[dict]:
    """For each mutant, the cheapest check that catches it.

    The output of this file. A comparison on random inputs is the cheapest thing available and
    it is enough for most of them; the rest need a check that looks at something other than the
    numbers, or numbers a random generator will not produce.
    """
    rows = []
    for row in mutation_report().rows:
        if row["outputs"]:
            needed = "comparing outputs"
        elif row["extremes"]:
            needed = "comparing outputs on infinities and nans"
        elif row["side_effects"]:
            needed = "checking the side effects survived"
        else:
            needed = "nothing here catches it"
        rows.append({"mutant": row["mutant"], "cheapest_check": needed})
    return rows


def every_mutant_is_caught_by_something() -> dict:
    """Whether the checks in this compiler between them catch everything tried."""
    rows = mutation_report().rows
    missed = [
        row["mutant"]
        for row in rows
        if not (row["outputs"] or row["side_effects"] or row["extremes"])
    ]
    return {"mutants": len(rows), "missed": missed, "all_caught": not missed}


def random_inputs_are_not_enough() -> dict:
    """How much of the work the usual check does, and how much it leaves.

    Most of it, which is why it is the usual check. Not all of it, which is why a compiler with
    only that check has a class of bug it cannot see rather than a low chance of seeing one.
    """
    report = mutation_report()
    return {
        "mutants": report.total,
        "caught_by_outputs": report.caught_by_the_usual_check,
        "survivors": report.survivors,
        "share": round(report.caught_by_the_usual_check / report.total, 3)
        if report.total
        else 0.0,
    }


def fixture_rank_hides_a_mutant() -> dict:
    """Whether the transpose mutant is invisible on matrices and visible above them.

    The most useful line in this file. It is not that the check was too weak, it is that every
    graph it ran on was rank two, and at rank two the mutant is not a mistake. A pass can be
    correct on every graph anybody tested it with and wrong.
    """
    mutant = transposes_that_always_cancel
    on_matrices = any(caught_by_outputs(mutant, graph) for _, graph in fixtures())
    on_rank_three = caught_by_outputs(mutant, rank_three_graph())
    return {
        "caught_on_matrices": on_matrices,
        "caught_at_rank_three": on_rank_three,
        "the_fixtures_were_the_problem": not on_matrices and on_rank_three,
    }


def mutants_on_generated_graphs(count: int = 16) -> list[dict]:
    """How often each mutant shows up on graphs nobody wrote.

    A different question from whether it can be caught. A mutant that changes the answer on one
    graph in sixteen is as wrong as one that changes it on all of them, and much less likely to
    be noticed by whoever is looking.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    graphs = list(generate_many(count))
    rows = []
    for mutant in MUTANTS:
        caught = sum(1 for graph in graphs if caught_by_outputs(mutant.transform, graph))
        rows.append(
            {
                "mutant": mutant.name,
                "graphs": len(graphs),
                "caught_on": caught,
                "share": round(caught / len(graphs), 3) if graphs else 0.0,
            }
        )
    return rows


def generated_graphs_catch_almost_nothing() -> dict:
    """What fuzzing alone would have found, which is not much.

    Four of the six mutants never show up on a generated graph at all, and the other two show
    up on a quarter and a sixteenth of them. Fuzzing is not a substitute for a fixture built to
    contain the pattern a pass is about; it is a way of finding patterns nobody thought to
    write down, which is a different job.
    """
    rows = mutants_on_generated_graphs()
    return {
        "mutants": len(rows),
        "never_caught": [row["mutant"] for row in rows if row["share"] == 0.0],
        "best_share": max((row["share"] for row in rows), default=0.0),
    }


def fixture_set_hides_a_mutant() -> dict:
    """Whether the subexpression mutant is invisible until a fixture holds the pattern.

    It is. On the three ordinary fixtures nothing catches it, because none of them reduces one
    value along two axes and so none of them has a pair of nodes the mutant would wrongly
    merge. Add a graph that does and the plain output comparison catches it immediately.
    """
    mutant = cse_that_ignores_attributes
    on_fixtures = any(caught_by_outputs(mutant, graph) for _, graph in fixtures())
    on_the_new_one = caught_by_outputs(mutant, two_reductions_of_one_value())
    return {
        "caught_on_the_old_fixtures": on_fixtures,
        "caught_on_the_new_one": on_the_new_one,
        "the_fixtures_were_the_problem": not on_fixtures and on_the_new_one,
    }
