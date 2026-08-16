from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import CodegenError, ConfigError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, elementwise_chain, layernorm_graph, softmax_graph
from tgc.ir.graph import Graph, Node

# How many values a fused kernel has to hold at once, and the order that minimises it.
#
# Once an elementwise chain is fused into a single loop, its intermediates never reach memory.
# They live in registers, and a machine has a fixed number of those. Exceeding it does not fail,
# it spills: the compiler writes a value to memory and reads it back, which is exactly the
# traffic fusion was meant to remove. A fusion pass that does not know how many registers it is
# using can fuse its way back to where it started.
#
# The number it needs is the peak of the live count over the evaluation order, and that peak
# depends on the order. For an expression tree there is an order that minimises it and a rule
# for finding it: evaluate the subtree that needs more registers first, because whatever the
# other subtree needs afterwards has to sit alongside only one held value rather than several.
# The rule is fifty years old and it is still the right answer for a tree.
#
# Two things the measurements say that the rule does not. A chain of any length needs two
# registers, so fusing a chain is unbounded and a pass has no reason to stop. A balanced tree
# grows one register per level of depth, so fusing a wide expression is bounded and a pass has
# every reason to stop. Those are different regimes and a single heuristic cannot serve both.
#
# The rule itself buys nothing on a balanced tree, which is worth knowing before reaching for
# it: both subtrees need the same number, so heavier first and lighter first are the same order.
# It pays on lopsided shapes, which is most of what a real expression is, and over forty random
# trees it wins on three quarters of them and never loses.


@dataclass
class Pressure:
    """How many values are live at once over one evaluation order."""

    order: tuple[str, ...]
    peak: int
    at_each_step: tuple[int, ...] = ()

    @property
    def steps(self) -> int:
        """Operations evaluated."""
        return len(self.order)

    def spills(self, registers: int) -> int:
        """How many values have to go to memory on a machine of a given size."""
        if registers < 1:
            raise ConfigError(f"a machine needs registers, got {registers}")
        return max(self.peak - registers, 0)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"steps": self.steps, "peak": self.peak}


def is_a_tree(graph: Graph) -> bool:
    """Whether every value in a graph is read at most once.

    The condition under which the ordering rule below applies. A value read twice has to stay
    live across everything between its two readers, and the minimum register count for a graph
    with that in it is a harder problem than for a tree.
    """
    counts = graph.use_counts()
    return all(count <= 1 for name, count in counts.items() if name not in graph.outputs)


def shared_values(graph: Graph) -> list[str]:
    """Values read more than once, which are why a graph is not a tree."""
    return [
        name
        for name, count in graph.use_counts().items()
        if count > 1 and name not in graph.outputs
    ]


def register_need(graph: Graph, name: str) -> int:
    """The Sethi Ullman number of a value: registers needed to compute it alone.

    A leaf needs one. A node whose two subtrees need different amounts needs the larger, because
    the smaller can be computed afterwards into whatever is left. A node whose two subtrees need
    the same amount needs one more than that, because after the first is computed and held, the
    second has the same requirement with one fewer register available.
    """
    node = graph.producer_of(name)
    if node is None or node.op.is_leaf:
        return 1
    needs = [register_need(graph, operand) for operand in node.inputs]
    if not needs:
        return 1
    if len(needs) == 1:
        return needs[0]
    first, second = max(needs), min(needs)
    return first if first != second else first + 1


def sethi_ullman_order(graph: Graph) -> list[str]:
    """An evaluation order that needs as few registers as any order does.

    Heavier subtree first. The whole rule, and the proof that it is optimal for a tree is the
    reason nobody has needed a better one: whatever the lighter subtree needs, it needs it while
    exactly one value is held, and doing it the other way round holds several.
    """
    if not is_a_tree(graph):
        raise CodegenError(f"{shared_values(graph)} are read twice, so this is not a tree")

    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        node = graph.producer_of(name)
        if node is None or name in visited:
            return
        visited.add(name)
        for operand in sorted(node.inputs, key=lambda item: -register_need(graph, item)):
            visit(operand)
        order.append(name)

    for output in graph.outputs:
        visit(output)
    return order


def source_order(graph: Graph) -> list[str]:
    """The order the nodes were written in."""
    return [node.name for node in graph.nodes]


def lighter_first_order(graph: Graph) -> list[str]:
    """The rule read backwards, kept so the difference can be measured.

    Not a straw man. Evaluating the cheap side first is what a recursive descent over the
    operands does when nobody has thought about it, and it is the order a naive code generator
    produces.
    """
    if not is_a_tree(graph):
        raise CodegenError(f"{shared_values(graph)} are read twice, so this is not a tree")

    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        node = graph.producer_of(name)
        if node is None or name in visited:
            return
        visited.add(name)
        for operand in sorted(node.inputs, key=lambda item: register_need(graph, item)):
            visit(operand)
        order.append(name)

    for output in graph.outputs:
        visit(output)
    return order


def measure_pressure(graph: Graph, order: Sequence[str]) -> Pressure:
    """How many values are live at each point of an order, and the peak.

    A value becomes live when it is produced and dies after its last reader in this order. The
    peak is what the machine has to have; anything above the peak is a register that sits empty
    for the whole kernel.
    """
    positions = {name: index for index, name in enumerate(order)}
    last_read: dict[str, int] = {}
    for name in order:
        node = graph.producer_of(name)
        if node is None:
            continue
        for operand in node.inputs:
            last_read[operand] = positions[name]
    for output in graph.outputs:
        last_read[output] = len(order)

    live: set[str] = set()
    counts: list[int] = []
    for index, name in enumerate(order):
        node = graph.producer_of(name)
        if node is not None:
            for operand in node.inputs:
                if operand not in live and graph.producer_of(operand) is None:
                    live.add(operand)
        live.add(name)
        counts.append(len(live))
        for held in list(live):
            if last_read.get(held, -1) <= index:
                live.discard(held)
    return Pressure(order=tuple(order), peak=max(counts, default=0), at_each_step=tuple(counts))


def best_pressure(graph: Graph) -> Pressure:
    """The peak under the ordering rule."""
    return measure_pressure(graph, sethi_ullman_order(graph))


def naive_pressure(graph: Graph) -> Pressure:
    """The peak under the order a recursive descent produces."""
    return measure_pressure(graph, lighter_first_order(graph))


def balanced_tree(depth: int = 4, size: int = 64) -> Graph:
    """A balanced expression tree, which is the shape that needs registers.

    Every node has two subtrees of equal weight, which is exactly the case where the ordering
    rule says one more register is needed than either side alone. So the requirement grows with
    the depth, and a fusion pass working on one of these has a limit.

    Measured, the peak is two more than the depth rather than one, because the accounting here
    counts both operands and the result. A machine that can write a result over one of its
    operands needs one fewer, and most can, so treat the number as an upper bound.
    """
    if depth < 1:
        raise ConfigError(f"a tree needs some depth, got {depth}")
    builder = Builder()
    leaves = [builder.input([size, size], name=f"x{index}") for index in range(2**depth)]
    level = leaves
    while len(level) > 1:
        level = [
            builder.add(level[index], level[index + 1]) for index in range(0, len(level), 2)
        ]
    return builder.finish(level[0])


def skewed_tree(depth: int = 8, size: int = 64) -> Graph:
    """A tree that leans entirely one way, which is a chain with extra inputs.

    Every node has one leaf and one subtree, so the two sides never need the same number and
    the rule never has to add one. Two registers whatever the depth, which is the other regime.
    """
    if depth < 1:
        raise ConfigError(f"a tree needs some depth, got {depth}")
    builder = Builder()
    current = builder.input([size, size], name="x0")
    for index in range(1, depth + 1):
        other = builder.input([size, size], name=f"x{index}")
        current = builder.add(current, other)
    return builder.finish(current)


def chain_needs_two_registers(lengths: Sequence[int] = (2, 4, 8, 16, 32)) -> list[dict]:
    """The peak for a chain of unary operations, at several lengths.

    Two, at every length. A chain reads one value and writes one, so the old value dies the
    moment the new one exists and nothing accumulates. That is why fusing a chain has no natural
    stopping point and why the fusion pass in this compiler does not look for one.
    """
    if not lengths:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for length in lengths:
        graph = elementwise_chain(length, sizes=(64, 64))
        rows.append({"length": length, "peak": best_pressure(graph).peak})
    return rows


def balanced_tree_needs_its_depth(depths: Sequence[int] = (1, 2, 3, 4, 5)) -> list[dict]:
    """The peak for a balanced tree, at several depths.

    Two more than the depth under this accounting, growing without limit while the chain stays
    at two. A fusion pass has to know which of the two shapes it is looking at, because the same
    decision is free in one and unaffordable in the other.
    """
    if not depths:
        raise ConfigError("there is nothing to sweep")
    return [
        {"depth": depth, "peak": best_pressure(balanced_tree(depth)).peak} for depth in depths
    ]


def ordering_saves_registers(depth: int = 4) -> dict:
    """The two orders on the same tree.

    Written to find out how much the rule buys on a balanced tree, and the answer is nothing:
    both sides of a balanced tree need the same number, so heavier first and lighter first are
    the same order. The rule pays on a lopsided tree and a balanced one is the case it cannot
    help with, which is worth knowing before reaching for it.
    """
    graph = balanced_tree(depth)
    return {
        "with_the_rule": best_pressure(graph).peak,
        "lighter_first": naive_pressure(graph).peak,
        "saving": naive_pressure(graph).peak - best_pressure(graph).peak,
    }


def lopsided_tree(heavy: int = 3, light: int = 2, size: int = 64) -> Graph:
    """A deep subtree on one side and a shallow one on the other.

    The shape the ordering rule is for. Doing the heavy side first holds one value while the
    light side is computed; doing the light side first holds one value while the heavy side
    needs all of its own, so the peak is one higher.

    Both sides have to be computed subtrees. An earlier version put a bare input on the light
    side, which needs no registers to produce and made the two orders identical, so the fixture
    measured nothing and reported a saving of zero.
    """
    if heavy < 1:
        raise ConfigError(f"the heavy side needs some depth, got {heavy}")
    if light < 1:
        raise ConfigError(f"the light side needs some depth, got {light}")
    builder = Builder()
    leaves = [builder.input([size, size], name=f"h{index}") for index in range(2**heavy)]
    level = leaves
    while len(level) > 1:
        level = [
            builder.add(level[index], level[index + 1]) for index in range(0, len(level), 2)
        ]

    shallow = [builder.input([size, size], name=f"l{index}") for index in range(2**light)]
    while len(shallow) > 1:
        shallow = [
            builder.add(shallow[index], shallow[index + 1])
            for index in range(0, len(shallow), 2)
        ]
    return builder.finish(builder.add(level[0], shallow[0]))


def the_rule_pays_on_a_lopsided_tree(heavy: int = 4, light: int = 1) -> dict:
    """The two orders on the shape the rule was written for."""
    graph = lopsided_tree(heavy, light)
    return {
        "with_the_rule": best_pressure(graph).peak,
        "lighter_first": naive_pressure(graph).peak,
        "saving": naive_pressure(graph).peak - best_pressure(graph).peak,
    }


def random_tree(nodes: int = 12, size: int = 64, *, seed: int = 0) -> Graph:
    """An expression tree nobody designed.

    Grown by repeatedly joining two values chosen at random, which produces a mix of balanced
    and lopsided shapes rather than either extreme. The sweep over these is what says how often
    the rule is worth applying rather than how much it helps when it is.
    """
    if nodes < 1:
        raise ConfigError(f"a tree needs some nodes, got {nodes}")
    generator = random.Random(seed)
    builder = Builder()
    available = [builder.input([size, size], name=f"x{index}") for index in range(nodes + 1)]
    while len(available) > 1:
        first = available.pop(generator.randrange(len(available)))
        second = available.pop(generator.randrange(len(available)))
        available.append(builder.add(first, second))
    return builder.finish(available[0])


def how_often_the_rule_helps(count: int = 40, nodes: int = 12) -> dict:
    """The rule against the naive order on a sample of random trees.

    It never loses, which is the guarantee, and it wins on about three quarters of them by an
    average of one and a half registers. The quarter it does not win on are shapes where the two
    subtrees need the same number and the two orders are the same order.
    """
    if count < 1:
        raise ConfigError(f"the count must be positive, got {count}")
    wins = 0
    losses = 0
    total_saving = 0
    for seed in range(count):
        graph = random_tree(nodes, seed=seed)
        best = best_pressure(graph).peak
        naive = naive_pressure(graph).peak
        if naive > best:
            wins += 1
            total_saving += naive - best
        elif naive < best:
            losses += 1
    return {
        "trees": count,
        "rule_wins": wins,
        "rule_loses": losses,
        "mean_saving": round(total_saving / count, 3),
    }


@dataclass
class SpillReport:
    """What a kernel costs on a machine with a fixed register file."""

    registers: int
    peak: int
    elements: int
    spills: int = 0
    extra_traffic: int = 0
    steps: list[dict] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        """Whether the kernel runs without touching memory."""
        return self.spills == 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "registers": self.registers,
            "peak": self.peak,
            "spills": self.spills,
            "extra_traffic": self.extra_traffic,
            "fits": self.fits,
        }


def spill_report(graph: Graph, registers: int = 8, elements: int = 4096) -> SpillReport:
    """What exceeding the register file costs, in bytes moved.

    Each spilled value is written once and read once, so the traffic is twice its size, and its
    size is the whole tile the kernel is working on rather than one number. That is the reason
    register pressure decides tile size: a kernel that spills at a tile of four thousand
    elements does not spill at a tile of four hundred, and the smaller tile is often faster for
    exactly that reason.
    """
    if registers < 1:
        raise ConfigError(f"a machine needs registers, got {registers}")
    if elements < 1:
        raise ConfigError(f"a tile has to hold something, got {elements}")
    pressure = best_pressure(graph)
    spills = pressure.spills(registers)
    return SpillReport(
        registers=registers,
        peak=pressure.peak,
        elements=elements,
        spills=spills,
        extra_traffic=spills * elements * 2 * 4,
    )


def register_file_sweep(
    graph: Graph | None = None, sizes: Sequence[int] = (2, 4, 8, 16, 32)
) -> list[dict]:
    """Spilling against the size of the register file.

    Falls to nothing once the file is as large as the peak and stays at nothing after, which is
    the shape of every resource limit. What matters is where it lands, and for a balanced tree
    of depth five that is six registers, which any real machine has and no real machine has many
    times over once a kernel is holding tiles rather than numbers.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    target = graph if graph is not None else balanced_tree(5)
    return [spill_report(target, registers=size).as_dict() for size in sizes]


def smaller_tiles_avoid_spilling(
    tiles: Sequence[int] = (256, 1024, 4096, 16384), registers: int = 4
) -> list[dict]:
    """The traffic a spill costs, against how much the kernel is holding.

    Linear in the tile, which is the point. Spilling is not a fixed penalty, it is a penalty
    proportional to everything the kernel is working on, so the decision to fuse further and the
    decision about tile size are the same decision.
    """
    if not tiles:
        raise ConfigError("there is nothing to sweep")
    graph = balanced_tree(4)
    return [
        {"tile": tile, **spill_report(graph, registers=registers, elements=tile).as_dict()}
        for tile in tiles
    ]


def fusion_groups_that_are_not_trees() -> list[dict]:
    """Which of the fixtures a tree rule can be applied to at all.

    Not many. A softmax reads its exponential twice and a layernorm reads its centred value
    twice, so neither is a tree and the ordering rule does not apply to either as written.
    Reported rather than worked around, because the honest answer for a graph with sharing is
    that minimising registers is a harder problem and this file solves the easier one.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(8)),
        ("balanced tree", balanced_tree(3)),
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
    ):
        rows.append(
            {
                "graph": label,
                "is_a_tree": is_a_tree(graph),
                "shared_values": len(shared_values(graph)),
            }
        )
    return rows


def applying_the_rule_to_a_shared_value_is_refused() -> bool:
    """Whether the rule refuses a graph it cannot handle rather than guessing."""
    try:
        sethi_ullman_order(softmax_graph())
    except CodegenError:
        return True
    return False


def elementwise_only(graph: Graph) -> list[Node]:
    """The nodes of a graph a register allocator would see.

    Anything that is not elementwise reaches memory anyway, so it is not part of a fused
    kernel's register problem. Filtering here rather than at the caller keeps the definition of
    what fuses in one place, which is ir/op.py.
    """
    return [node for node in graph.nodes if node.op.is_elementwise or node.op is ops.INPUT]
