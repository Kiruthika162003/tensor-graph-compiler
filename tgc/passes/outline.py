from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, PassError
from tgc.ir import op as ops
from tgc.ir.builder import Builder
from tgc.ir.graph import Graph, Node
from tgc.ir.isomorphism import structural_hash
from tgc.passes.fusion import find_groups
from tgc.verify.reference import evaluate_node, outputs_agree, random_feeds, run

# Pulling a repeated run of operations out into something called several times.
#
# A model is the same layer applied twenty times. A traced graph of it is twenty copies of the
# same sequence of operations on different values, and a subexpression pass finds nothing in it,
# because nothing computes the same thing twice. What repeats is the shape of the computation,
# not any value in it, and outlining is the pass that exploits that.
#
# The saving is in code rather than in time. Twenty copies of a layer become one copy and twenty
# calls, and everything downstream of that gets smaller in the same proportion: the generated
# source, the compile time, the instruction cache footprint. The arithmetic is identical.
#
# The cost is real and it is not the call overhead. A call is a boundary, and a fusion pass will
# not merge across one, so a chain that ran as a single loop through what used to be a layer
# boundary now runs as two. The measurement below puts a number on that, and the number is the
# reason this pass belongs after fusion rather than before it.
#
# There is no call operation in the IR. Adding one would mean an op with variable arity holding
# a whole graph, which is a much larger change than the two this compiler has already made to
# its op set, so an outlined program is a separate structure with its own interpreter and the
# check is that the interpreter agrees with the graph it came from, bit for bit.


@dataclass(frozen=True)
class Call:
    """One place a function is used."""

    function: str
    arguments: tuple[str, ...]
    result: str

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "function": self.function,
            "arguments": list(self.arguments),
            "result": self.result,
        }


@dataclass
class Program:
    """A main body with calls in it, and the functions those calls reach."""

    functions: dict[str, Graph] = field(default_factory=dict)
    steps: list[Node | Call] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        """Calls in the body."""
        return sum(1 for step in self.steps if isinstance(step, Call))

    @property
    def total_nodes(self) -> int:
        """Nodes anywhere in the program, function bodies included."""
        inline = sum(1 for step in self.steps if isinstance(step, Node))
        return inline + sum(len(graph.nodes) for graph in self.functions.values())

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "functions": len(self.functions),
            "calls": self.call_count,
            "steps": len(self.steps),
            "total_nodes": self.total_nodes,
        }


def _is_a_chain(window: Sequence[Node]) -> bool:
    """Whether each node in a window reads the one before it."""
    return all(earlier.name in later.inputs for earlier, later in itertools.pairwise(window))


def parameters_of(window: Sequence[Node]) -> list[str]:
    """The values a window reads from outside itself, in order of first use.

    Everything a function would need as an argument. Values produced inside the window are not
    parameters, which is what makes a longer window take fewer of them and is most of the reason
    a longer pattern is worth more.
    """
    produced = {node.name for node in window}
    seen: list[str] = []
    for node in window:
        for name in node.inputs:
            if name not in produced and name not in seen:
                seen.append(name)
    return seen


def window_key(graph: Graph, window: Sequence[Node]) -> str:
    """A key two windows share when one function could serve both.

    The operations and their attributes, the shape of every intermediate, and the shapes of the
    parameters. The parameter shapes are the part that is easy to leave out: two windows with
    identical operations reading differently shaped values need different code, and a matcher
    that missed that would outline them together and produce a function that cannot be called.
    """
    body = "|".join(
        f"{node.op.name}:{sorted(node.attrs.items())}:{node.output.shape}" for node in window
    )
    signature = ";".join(str(graph.value(name).shape) for name in parameters_of(window))
    return f"{body}||{signature}"


@dataclass
class Occurrence:
    """One run of nodes a function could replace."""

    start: int
    nodes: tuple[str, ...]
    parameters: tuple[str, ...]
    result: str

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "start": self.start,
            "nodes": list(self.nodes),
            "parameters": list(self.parameters),
        }


def find_occurrences(graph: Graph, length: int = 3) -> dict[str, list[Occurrence]]:
    """Every run of a given length, grouped by the key that says which are interchangeable.

    Overlapping runs are kept here and resolved later. Dropping them at this point would make
    the answer depend on which end the scan started from, and the greedy choice belongs with the
    rewrite rather than with the search.
    """
    if length < 1:
        raise ConfigError(f"a pattern needs some length, got {length}")
    groups: dict[str, list[Occurrence]] = {}
    for start in range(len(graph.nodes) - length + 1):
        window = graph.nodes[start : start + length]
        if not _is_a_chain(window):
            continue
        if any(node.op.is_leaf for node in window):
            continue
        occurrence = Occurrence(
            start=start,
            nodes=tuple(node.name for node in window),
            parameters=tuple(parameters_of(window)),
            result=window[-1].name,
        )
        groups.setdefault(window_key(graph, window), []).append(occurrence)
    return groups


def _escapes(graph: Graph, occurrence: Occurrence) -> bool:
    """Whether anything outside a window reads one of its intermediates.

    A function returns one value. A window whose middle is read from outside would need to
    return two, so it is refused rather than handled, and refusing it is the whole of the
    legality check for this pass.
    """
    inside = set(occurrence.nodes)
    for node in graph.nodes:
        if node.name in inside:
            continue
        for name in node.inputs:
            if name in inside and name != occurrence.result:
                return True
    return any(name in inside and name != occurrence.result for name in graph.outputs)


def choose_occurrences(
    graph: Graph, length: int = 3, *, minimum: int = 2
) -> tuple[str, list[Occurrence]]:
    """The best group of non overlapping, legal occurrences.

    Best by how many nodes it removes, which is the count of occurrences times one less than the
    length. Ties go to the longer key so the choice does not depend on dictionary order, which
    would make the pass produce different programs on different runs of the same input.
    """
    best_key = ""
    best: list[Occurrence] = []
    best_score = 0
    for key, occurrences in sorted(find_occurrences(graph, length).items()):
        legal = [item for item in occurrences if not _escapes(graph, item)]
        chosen: list[Occurrence] = []
        used: set[str] = set()
        for item in legal:
            if used & set(item.nodes):
                continue
            chosen.append(item)
            used |= set(item.nodes)
        if len(chosen) < minimum:
            continue
        score = len(chosen) * (length - 1)
        if score > best_score:
            best_key, best, best_score = key, chosen, score
    return best_key, best


def build_function(graph: Graph, occurrence: Occurrence) -> Graph:
    """A standalone graph computing what one window computes.

    Built by replaying the window's operations against fresh inputs rather than by copying
    nodes, so the function's shapes come out of the same inference the rest of the compiler uses
    and cannot drift from what the window meant.

    The parameters are renamed by position. Keeping the caller's names collides with the names
    the builder invents for the body, because an occurrence halfway down a graph takes a
    parameter called something the builder is about to hand out.
    """
    builder = Builder()
    mapping: dict[str, str] = {}
    for position, name in enumerate(occurrence.parameters):
        value = graph.value(name)
        mapping[name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=f"arg{position}",
        )
    for name in occurrence.nodes:
        node = graph.node(name)
        if node.op is ops.CONSTANT:
            mapping[name] = builder.constant(
                float(node.attrs["value"]), dtype=node.output.dtype
            )
            continue
        operands = [mapping[item] for item in node.inputs]
        mapping[name] = builder.apply(node.op, *operands, **node.attrs)
    return builder.finish(mapping[occurrence.result])


def outline(graph: Graph, length: int = 3, *, minimum: int = 2) -> Program:
    """Replace the best repeated run with a function and calls to it.

    One function, not several. Outlining every pattern in one pass would need a conflict
    resolution between overlapping groups that is more machinery than the measurement below
    justifies, and the interesting number is what one function is worth rather than what all of
    them together are.
    """
    _, occurrences = choose_occurrences(graph, length, minimum=minimum)
    program = Program(
        inputs=[value.name for value in graph.inputs], outputs=list(graph.outputs)
    )
    if not occurrences:
        program.steps = list(graph.nodes)
        return program

    program.functions["body"] = build_function(graph, occurrences[0])
    replaced = {item.nodes[0]: item for item in occurrences}
    inside = {name for item in occurrences for name in item.nodes}

    for node in graph.nodes:
        if node.name in replaced:
            item = replaced[node.name]
            program.steps.append(
                Call(function="body", arguments=item.parameters, result=item.result)
            )
            continue
        if node.name in inside:
            continue
        program.steps.append(node)
    return program


def run_program(program: Program, feeds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Evaluate an outlined program.

    The interpreter that makes the rewrite checkable. A call binds its arguments to the
    function's inputs by position and runs the function's graph, which is exactly what a call
    means and is three lines because the function is an ordinary graph.
    """
    missing = [name for name in program.inputs if name not in feeds]
    if missing:
        raise PassError(f"no value supplied for {missing}")

    environment = dict(feeds)
    for step in program.steps:
        if isinstance(step, Call):
            function = program.functions[step.function]
            if len(step.arguments) != len(function.inputs):
                raise PassError(
                    f"{step.function} takes {len(function.inputs)} arguments, "
                    f"given {len(step.arguments)}"
                )
            bound = {
                value.name: environment[name]
                for value, name in zip(function.inputs, step.arguments, strict=True)
            }
            environment[step.result] = run(function, bound)[0]
            continue
        operands = [environment[name] for name in step.inputs]
        environment[step.name] = evaluate_node(step, operands)
    return [environment[name] for name in program.outputs]


def stacked_layers(count: int = 4, width: int = 16) -> Graph:
    """The same two operations applied several times, which is what a model is."""
    if count < 1:
        raise ConfigError(f"there has to be at least one layer, got {count}")
    builder = Builder()
    current = builder.input([width, width], name="x")
    weight = builder.input([width, width], name="w")
    for _ in range(count):
        current = builder.tanh(builder.matmul(current, weight))
    return builder.finish(current)


def outlining_preserves_the_answer(graph: Graph | None = None, length: int = 2) -> dict:
    """The outlined program against the graph it came from.

    Bit equality, and it should be: a call performs the same operations on the same values in
    the same order, so anything short of exact agreement would mean the extraction changed
    something. This is the one rewrite in the compiler where a tolerance would be a bug.
    """
    target = graph if graph is not None else stacked_layers()
    program = outline(target, length)
    feeds = random_feeds(target, positive=True)
    return {
        "calls": program.call_count,
        "identical": outputs_agree(run(target, feeds), run_program(program, feeds)),
    }


def code_size(graph: Graph | None = None, length: int = 2) -> dict:
    """Nodes before and after, which is the whole point of the pass."""
    target = graph if graph is not None else stacked_layers()
    program = outline(target, length)
    return {
        "before": len(target.nodes),
        "after": program.total_nodes,
        "calls": program.call_count,
        "saved": len(target.nodes) - program.total_nodes,
    }


def size_by_layer_count(counts: Sequence[int] = (2, 4, 8, 16, 32)) -> list[dict]:
    """How the saving grows with how many times the pattern appears.

    Linearly, which is the argument for the pass. A model of thirty two identical layers
    compiles from the code of one, and every call after the first is free in code size.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    return [{"layers": count, **code_size(stacked_layers(count))} for count in counts]


def a_single_occurrence_is_left_alone() -> dict:
    """What the pass does with a pattern that appears once.

    Nothing, and it has to. A function called once is strictly worse than the code inlined: the
    same nodes, plus a call boundary that stops a fusion pass, for no saving at all.
    """
    graph = stacked_layers(1)
    program = outline(graph, 2)
    return {"calls": program.call_count, "steps": len(program.steps)}


def pattern_length_sweep(lengths: Sequence[int] = (1, 2, 3, 4, 6)) -> list[dict]:
    """The saving against how long the outlined run is.

    Peaks at the length of the repeating unit, which for this fixture is two. A run of one
    saves nothing at all, because a body of one node replaces a call site of one node. A run
    longer than the unit matches half as often, since it only lines up every second layer, and
    the saving falls with it.
    """
    if not lengths:
        raise ConfigError("there is nothing to sweep")
    graph = stacked_layers(8)
    rows = []
    for length in lengths:
        program = outline(graph, length)
        rows.append(
            {
                "length": length,
                "calls": program.call_count,
                "total_nodes": program.total_nodes,
                "saved": len(graph.nodes) - program.total_nodes,
            }
        )
    return rows


def best_pattern_length(graph: Graph | None = None) -> int:
    """The run length that removes the most nodes."""
    target = graph if graph is not None else stacked_layers(8)
    best = 0
    chosen = 1
    for length in (1, 2, 3, 4, 6):
        program = outline(target, length)
        saved = len(target.nodes) - program.total_nodes
        if saved > best:
            best, chosen = saved, length
    return chosen


def elementwise_layers(count: int = 8, width: int = 16) -> Graph:
    """Layers with nothing in them a fusion pass would stop at.

    Needed because the obvious fixture cannot show the cost. A layer with a matrix product in it
    already has a fusion boundary at the product, so outlining it loses nothing, and measuring
    the loss on that graph would have reported a cost of zero and been believed.
    """
    if count < 1:
        raise ConfigError(f"there has to be at least one layer, got {count}")
    builder = Builder()
    current = builder.input([width, width], name="x")
    for _ in range(count):
        current = builder.sigmoid(builder.tanh(current))
    return builder.finish(current)


def fusion_lost_to_the_boundary(graph: Graph | None = None, length: int = 2) -> dict:
    """What a call boundary costs a fusion pass.

    An unbroken elementwise graph is one fusion group and runs as one loop. Outlining it into a
    function called eight times gives eight separate loops, because nothing merges across a
    call, and each of them writes its result to memory for the next call to read back.

    So the pass has to run after fusion rather than before it, and even then it is a trade: the
    code got eight times smaller and the traffic went up by every intermediate the single loop
    was keeping in registers.
    """
    target = graph if graph is not None else elementwise_layers()
    program = outline(target, length)
    inline_nodes = [step for step in program.steps if isinstance(step, Node)]
    inline_groups = (
        len(find_groups(Graph(nodes=inline_nodes, inputs=list(target.inputs), outputs=[])))
        if inline_nodes
        else 0
    )
    return {
        "loops_before": len(find_groups(target)),
        "loops_after": program.call_count + inline_groups,
        "calls": program.call_count,
        "inline_nodes": len(inline_nodes),
    }


def the_cost_is_zero_where_there_was_already_a_boundary() -> dict:
    """The same measurement on a graph whose layers hold a matrix product.

    Nothing is lost, because the product was already breaking the fusion at exactly the point
    the call boundary now sits. Which layers are worth outlining depends on what is in them, and
    the ones with a contraction in them are free.
    """
    return {
        "elementwise": fusion_lost_to_the_boundary(elementwise_layers()),
        "with_a_product": fusion_lost_to_the_boundary(stacked_layers(8)),
    }


def outlining_is_worth_it_when_the_pattern_repeats(
    counts: Sequence[int] = (1, 2, 4, 8, 16),
) -> list[dict]:
    """Where the pass starts paying, as a function of how many layers there are.

    At two. One layer gives one occurrence and the pass declines; two gives a saving equal to
    one layer's worth of nodes, and every layer after that adds the same again.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for count in counts:
        result = code_size(stacked_layers(count))
        result["layers"] = count
        result["worth_it"] = result["saved"] > 0
        rows.append(result)
    return rows


def a_shared_intermediate_is_refused() -> dict:
    """A window whose middle is read from outside, which cannot become a function.

    A function returns one value. The graph here reads the product inside the layer as well as
    the layer's output, so the window would have to return two, and the pass declines it rather
    than inventing a convention for multiple results.
    """
    builder = Builder()
    x = builder.input([16, 16], name="x")
    weight = builder.input([16, 16], name="w")
    product = builder.matmul(x, weight)
    activated = builder.tanh(product)
    graph = builder.finish(builder.add(activated, product))

    groups = find_occurrences(graph, 2)
    legal = sum(
        1
        for occurrences in groups.values()
        for item in occurrences
        if not _escapes(graph, item)
    )
    return {"windows": sum(len(items) for items in groups.values()), "legal": legal}


def the_function_is_the_same_graph_every_time(graph: Graph | None = None) -> dict:
    """Whether the occurrences really are one function rather than several.

    Checked by structural hash rather than by the matcher that grouped them, so a bug in the key
    would show up here instead of being confirmed by the thing that caused it.
    """
    target = graph if graph is not None else stacked_layers(4)
    _, occurrences = choose_occurrences(target, 2)
    hashes = {structural_hash(build_function(target, item)) for item in occurrences}
    return {"occurrences": len(occurrences), "distinct_functions": len(hashes)}
