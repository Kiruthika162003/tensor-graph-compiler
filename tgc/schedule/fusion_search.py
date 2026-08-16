from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.errors import ConfigError, ScheduleError

# Choosing what to keep in memory, when the alternative is computing it twice.
#
# passes/fusion.py merges elementwise runs and stops at a reduction, because a reduction has a
# different iteration space and cannot share a loop nest with what reads it. That is what makes
# fusion a decision rather than a preference. A value that feeds two reductions feeds two
# separate kernels, and it is either written to memory and read by both of them or computed
# again inside each one. There is no third option.
#
# So a plan is a set of values to materialise. Reductions and graph outputs are in it whatever
# the plan says. Everything else is recomputed wherever it is needed, and the cost is the
# traffic of the materialised set plus the flops of every group, with a duplicated operation
# counted once per group it lands in.
#
# The rule every compiler uses is to materialise anything with more than one reader, and the
# measurement here says it is right on two of the three shapes it was given. On a chain it is
# optimal and so is everything else, because nothing can be duplicated. On a cone of four
# elementwise operations feeding three reductions it depends entirely on how much arithmetic the
# cone does per element: somewhere between eight and sixteen flops the answer flips, and below
# that the cheaper plan walks the whole cone three times while the rule pays half again on top
# of the best. The crossover is where the extra passes cost what the tensor the rule writes and
# reads costs, which is a property of the machine rather than of the graph.

BYTES_PER_ELEMENT = 4
MEMORY_WEIGHT = 10.0


@dataclass(frozen=True)
class Dag:
    """A graph flattened to producer lists, with a size and a flop count per value."""

    producers: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]
    flops: tuple[int, ...]
    outputs: tuple[int, ...]
    barriers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.producers)
        if count == 0:
            raise ConfigError("a graph needs at least one value")
        if len(self.sizes) != count or len(self.flops) != count:
            raise ConfigError("every value needs a size and a flop count")
        for index, sources in enumerate(self.producers):
            for source in sources:
                if not 0 <= source < index:
                    raise ConfigError(
                        f"value {index} reads {source}, which is not an earlier value"
                    )
        if not self.outputs:
            raise ConfigError("a graph with no outputs computes nothing")
        for value in (*self.outputs, *self.barriers):
            if not 0 <= value < count:
                raise ConfigError(f"{value} is not a value in this graph")
            if not self.producers[value]:
                raise ConfigError(f"value {value} is an input, so it computes nothing")

    @property
    def count(self) -> int:
        """Values in the graph, inputs included."""
        return len(self.producers)

    @property
    def leaves(self) -> frozenset[int]:
        """Values that arrive from memory rather than being computed."""
        return frozenset(index for index, sources in enumerate(self.producers) if not sources)

    @property
    def required(self) -> frozenset[int]:
        """Values that are written whatever the plan says.

        Graph outputs, because somebody outside asked for them, and reductions, because the
        kernel that produces one cannot be merged into the kernel that reads it.
        """
        return frozenset(self.outputs) | frozenset(self.barriers)

    @property
    def interior(self) -> tuple[int, ...]:
        """Computed values that are not already required. The search space."""
        required = self.required
        return tuple(
            index
            for index in range(self.count)
            if self.producers[index] and index not in required
        )

    def consumers(self, value: int) -> tuple[int, ...]:
        """Everything that reads a value."""
        if not 0 <= value < self.count:
            raise ConfigError(f"{value} is not a value in this graph")
        return tuple(index for index, sources in enumerate(self.producers) if value in sources)

    def fan_out(self, value: int) -> int:
        """How many readers a value has, counting a graph output as a reader."""
        extra = 1 if value in self.outputs else 0
        return len(self.consumers(value)) + extra

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "values": self.count,
            "inputs": len(self.leaves),
            "interior": len(self.interior),
            "barriers": len(self.barriers),
            "outputs": len(self.outputs),
        }


def group_for(graph: Dag, root: int, materialised: frozenset[int]) -> frozenset[int]:
    """The operations one materialised value is computed from.

    Everything upstream of the root, stopping at a leaf or at another materialised value. Two
    groups can overlap, and the overlap is the duplicated work that makes this a decision.
    """
    if not graph.producers[root]:
        raise ScheduleError(f"value {root} is an input, so it does not root a group")
    members: set[int] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in members:
            continue
        members.add(current)
        for source in graph.producers[current]:
            if source in materialised or not graph.producers[source]:
                continue
            pending.append(source)
    return frozenset(members)


def normalise(graph: Dag, chosen: Sequence[int]) -> frozenset[int]:
    """The materialised set a choice implies, with the required values added back.

    An output is written whether or not it was chosen and so is a reduction, so a plan that
    leaves one out is the same plan as one that includes it. Folding that in here means the
    search never counts the same plan twice.
    """
    for value in chosen:
        if not 0 <= value < graph.count:
            raise ScheduleError(f"{value} is not a value in this graph")
        if not graph.producers[value]:
            raise ScheduleError(f"value {value} is an input and is already in memory")
    return frozenset(chosen) | graph.required


@dataclass
class FusionPlan:
    """A set of materialised values, and what it costs."""

    materialised: frozenset[int] = frozenset()
    groups: int = 0
    flops: int = 0
    traffic: int = 0
    memory_weight: float = MEMORY_WEIGHT

    @property
    def cost(self) -> float:
        """Flops plus weighted bytes."""
        return self.flops + self.traffic * self.memory_weight

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "materialised": sorted(self.materialised),
            "groups": self.groups,
            "flops": self.flops,
            "traffic": self.traffic,
            "cost": round(self.cost, 2),
        }


def groups_of(graph: Dag, chosen: Sequence[int]) -> dict[int, frozenset[int]]:
    """The kernels a plan produces, keyed by the value each one writes."""
    materialised = normalise(graph, chosen)
    return {root: group_for(graph, root, materialised) for root in sorted(materialised)}


def cost_of(
    graph: Dag, chosen: Sequence[int], memory_weight: float = MEMORY_WEIGHT
) -> FusionPlan:
    """What one materialisation choice costs.

    The write of every materialised value, one read per group that needs it, and the flops of
    every group with duplication counted. Reads are charged per group rather than per edge
    because two consumers inside the same kernel read the value once.
    """
    if memory_weight < 0:
        raise ConfigError(f"a memory weight of {memory_weight} is not a weight")
    materialised = normalise(graph, chosen)
    groups = groups_of(graph, chosen)
    flops = sum(graph.flops[member] for members in groups.values() for member in members)
    traffic = sum(graph.sizes[root] for root in materialised)
    for root, members in groups.items():
        needed = {
            source
            for member in members
            for source in graph.producers[member]
            if source not in members and source != root
        }
        traffic += sum(graph.sizes[source] for source in needed)
    return FusionPlan(
        materialised=materialised,
        groups=len(groups),
        flops=flops,
        traffic=traffic * BYTES_PER_ELEMENT,
        memory_weight=memory_weight,
    )


def times_computed(graph: Dag, chosen: Sequence[int], value: int) -> int:
    """How many kernels contain one operation under a plan."""
    if not 0 <= value < graph.count:
        raise ScheduleError(f"{value} is not a value in this graph")
    return sum(1 for members in groups_of(graph, chosen).values() if value in members)


def duplication_factor(graph: Dag, chosen: Sequence[int]) -> float:
    """How many times the average operation is computed under a plan.

    One when nothing is duplicated. The number is the whole reason a fusion decision exists, and
    on the cone below it rises with the number of readers rather than staying near one.
    """
    groups = list(groups_of(graph, chosen).values())
    computed = sum(len(members) for members in groups)
    distinct = len({member for members in groups for member in members})
    if distinct == 0:
        raise ScheduleError("a plan with no groups computes nothing")
    return computed / distinct


def materialise_everything(graph: Dag, memory_weight: float = MEMORY_WEIGHT) -> FusionPlan:
    """The plan an unfused graph runs: every value written and read back."""
    return cost_of(graph, graph.interior, memory_weight)


def materialise_outputs_only(graph: Dag, memory_weight: float = MEMORY_WEIGHT) -> FusionPlan:
    """Maximum fusion: nothing kept beyond what has to be, everything else recomputed."""
    return cost_of(graph, (), memory_weight)


def materialise_reused(graph: Dag, memory_weight: float = MEMORY_WEIGHT) -> FusionPlan:
    """The rule compilers use: keep anything read more than once.

    It draws the line exactly where duplication becomes possible, which is the right place if
    duplication is always the expensive side. That is often true and not always, and the case
    where it is false is measured below.
    """
    chosen = [value for value in graph.interior if graph.fan_out(value) > 1]
    return cost_of(graph, chosen, memory_weight)


def exhaustive(graph: Dag, memory_weight: float = MEMORY_WEIGHT, limit: int = 16) -> FusionPlan:
    """Every subset of the interior values, and the cheapest.

    Two to the interior count, so it stops being runnable at about sixteen values. It is the
    floor the three rules are measured against, not a method.
    """
    interior = graph.interior
    if len(interior) > limit:
        raise ScheduleError(
            f"{len(interior)} interior values give {2 ** len(interior)} plans, too many"
        )
    best = materialise_outputs_only(graph, memory_weight)
    for size in range(len(interior) + 1):
        for chosen in itertools.combinations(interior, size):
            plan = cost_of(graph, chosen, memory_weight)
            if plan.cost < best.cost:
                best = plan
    return best


def chain_dag(length: int = 8, size: int = 4096, intensity: int = 1) -> Dag:
    """A run of unary operations, each reading the one before it.

    The shape with no decision in it. Nothing has two readers, so no plan can duplicate
    anything and every plan does the same arithmetic.
    """
    if length < 1:
        raise ConfigError(f"a chain of {length} operations is not a chain")
    if intensity < 1:
        raise ConfigError(f"an intensity of {intensity} is not work")
    producers: list[tuple[int, ...]] = [()]
    for index in range(1, length + 1):
        producers.append((index - 1,))
    flops = [0] + [size * intensity for _ in range(length)]
    return Dag(
        producers=tuple(producers),
        sizes=tuple(size for _ in producers),
        flops=tuple(flops),
        outputs=(length,),
    )


def cone_dag(
    readers: int = 3,
    depth: int = 4,
    size: int = 4096,
    intensity: int = 1,
    reduced: int = 64,
) -> Dag:
    """An elementwise cone feeding several reductions, whose results are combined.

    The shape the decision lives in. The tip of the cone is read by every reduction and each
    reduction is its own kernel, so the tip is either written once and read by all of them or
    the whole cone is walked again inside each one. Both are reasonable and which is cheaper is
    a number rather than an opinion.
    """
    if readers < 2:
        raise ConfigError(f"{readers} readers is not a fan out")
    if depth < 1:
        raise ConfigError(f"a cone of {depth} operations is not a cone")
    if reduced < 1:
        raise ConfigError(f"a reduction to a {reduced}th is not a reduction")
    small = max(size // reduced, 1)
    producers: list[tuple[int, ...]] = [()]
    sizes = [size]
    flops = [0]
    for index in range(depth):
        producers.append((index,))
        sizes.append(size)
        flops.append(size * intensity)
    tip = depth
    barriers = []
    for _ in range(readers):
        producers.append((tip,))
        sizes.append(small)
        flops.append(size)
        barriers.append(len(producers) - 1)
    joined = barriers[0]
    for barrier in barriers[1:]:
        producers.append((joined, barrier))
        sizes.append(small)
        flops.append(small * intensity)
        joined = len(producers) - 1
    return Dag(
        producers=tuple(producers),
        sizes=tuple(sizes),
        flops=tuple(flops),
        outputs=(joined,),
        barriers=tuple(barriers),
    )


def cheap_cone() -> Dag:
    """A cone that does one flop per element. Recomputing it is nearly free."""
    return cone_dag(intensity=1)


def expensive_cone() -> Dag:
    """A cone that does sixty four flops per element. Recomputing it is not."""
    return cone_dag(intensity=64)


def compare_strategies(graph: Dag | None = None) -> list[dict]:
    """The three rules and the search, on one graph."""
    target = graph if graph is not None else cheap_cone()
    return [
        {"strategy": "everything", **materialise_everything(target).as_dict()},
        {"strategy": "outputs only", **materialise_outputs_only(target).as_dict()},
        {"strategy": "reused values", **materialise_reused(target).as_dict()},
        {"strategy": "exhaustive", **exhaustive(target).as_dict()},
    ]


def on_a_chain_maximum_fusion_is_optimal() -> dict:
    """Whether the extreme plan is right when nothing fans out.

    It is. A chain has no value with two readers, so no plan can duplicate anything and the
    flops are identical under every plan. That leaves the cost as pure traffic, and the plan
    that writes the least is the one that writes only the output.
    """
    graph = chain_dag()
    fused = materialise_outputs_only(graph)
    best = exhaustive(graph)
    return {
        "fused": round(fused.cost, 2),
        "best": round(best.cost, 2),
        "optimal": abs(fused.cost - best.cost) < 1e-9,
        "duplication": round(duplication_factor(graph, ()), 4),
        "against_no_fusion": round(materialise_everything(graph).cost / fused.cost, 3),
    }


def the_cone_is_walked_once_per_reader(readers: int = 3) -> dict:
    """What maximum fusion actually does to the shared cone.

    Walks it once for each reduction that reads it. The tip is not written down, so each kernel
    rebuilds it from the input, and with three readers every operation in the cone runs three
    times. That is the cost the rule exists to avoid.
    """
    graph = cone_dag(readers=readers)
    tip = 4
    return {
        "readers": readers,
        "tip_computed": times_computed(graph, (), tip),
        "tip_computed_under_the_rule": times_computed(graph, [tip], tip),
        "duplication": round(duplication_factor(graph, ()), 3),
    }


def duplication_grows_with_the_reader_count(
    counts: Sequence[int] = (2, 3, 4, 5),
) -> list[dict]:
    """How the recomputation scales as more reductions read the same cone."""
    if not counts:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for readers in counts:
        graph = cone_dag(readers=readers)
        rows.append(
            {
                "readers": readers,
                "duplication": round(duplication_factor(graph, ()), 3),
                "fused_flops": materialise_outputs_only(graph).flops,
                "rule_flops": materialise_reused(graph).flops,
            }
        )
    return rows


def the_rule_wins_on_an_expensive_cone() -> dict:
    """Whether writing the tip is right when the cone does real arithmetic.

    It is. At sixty four flops per element the eight extra passes maximum fusion makes over the
    cone cost several times what the write and the three reads cost, so the plan that keeps the
    tip is both the rule's answer and the search's.
    """
    graph = expensive_cone()
    rule = materialise_reused(graph)
    best = exhaustive(graph)
    return {
        "rule": round(rule.cost, 2),
        "fully_fused": round(materialise_outputs_only(graph).cost, 2),
        "best": round(best.cost, 2),
        "regret": round(rule.cost / best.cost, 4),
        "matches": abs(rule.cost - best.cost) < 1e-9,
    }


def the_rule_loses_on_a_cheap_cone() -> dict:
    """And whether it is still right when the cone barely computes anything.

    It is not. At one flop per element the cone is pure traffic, and recomputing it three times
    costs less arithmetic than the single tensor the rule writes and reads back costs bytes. The
    rule keeps the tip anyway, because it looks at the reader count and not at what the value
    cost to produce.
    """
    graph = cheap_cone()
    rule = materialise_reused(graph)
    best = exhaustive(graph)
    return {
        "rule": round(rule.cost, 2),
        "fully_fused": round(materialise_outputs_only(graph).cost, 2),
        "best": round(best.cost, 2),
        "regret": round(rule.cost / best.cost, 4),
        "rule_materialised": sorted(rule.materialised),
        "best_materialised": sorted(best.materialised),
    }


def intensity_sweep(
    intensities: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
) -> list[dict]:
    """Where the answer flips, as the cone is given more arithmetic to do.

    Between eight and sixteen flops per element, which is where eight extra passes over the cone
    start costing more than the two tensors the rule moves. Nothing about the graph changes
    across this sweep. The right plan changes because the machine did.
    """
    if not intensities:
        raise ConfigError("there is nothing to sweep")
    rows = []
    for intensity in intensities:
        graph = cone_dag(intensity=intensity)
        rule = materialise_reused(graph)
        best = exhaustive(graph)
        rows.append(
            {
                "intensity": intensity,
                "rule": round(rule.cost, 2),
                "best": round(best.cost, 2),
                "regret": round(rule.cost / best.cost, 4),
                "rule_is_best": abs(rule.cost - best.cost) < 1e-9,
            }
        )
    return rows


def the_crossover_is_a_property_of_the_machine() -> dict:
    """Which side of the crossover each end of the sweep lands on."""
    rows = {row["intensity"]: row for row in intensity_sweep()}
    flipped = [intensity for intensity, row in sorted(rows.items()) if row["rule_is_best"]]
    return {
        "at_one": rows[1]["rule_is_best"],
        "at_sixty_four": rows[64]["rule_is_best"],
        "flips_at": min(flipped) if flipped else 0,
        "worst_regret": max(row["regret"] for row in rows.values()),
    }


def memory_weight_sweep(
    weights: Sequence[float] = (0.0, 0.1, 1.0, 10.0, 100.0),
    graph: Dag | None = None,
) -> list[dict]:
    """How the best plan moves as memory gets expensive relative to arithmetic.

    The same crossover approached from the other side. Making a byte dearer has the same effect
    on the decision as making a flop cheaper, which is worth saying because only one of the two
    is something a compiler can see from the graph.
    """
    if not weights:
        raise ConfigError("there is nothing to sweep")
    target = graph if graph is not None else cone_dag(intensity=8)
    rows = []
    for weight in weights:
        best = exhaustive(target, memory_weight=weight)
        rows.append(
            {
                "memory_weight": weight,
                "materialised": len(best.materialised),
                "duplication": round(duplication_factor(target, sorted(best.materialised)), 3),
                "cost": round(best.cost, 2),
            }
        )
    return rows


def free_memory_means_no_duplication() -> dict:
    """What the search picks when traffic is not charged.

    A plan that computes everything once, and not the plan that writes everything. With no
    charge for a write the cost is flops alone, and the flops bottom out as soon as nothing is
    duplicated, which happens at five values rather than nine. Everything past that is free and
    pointless, and the search takes the first of the tied plans rather than the largest.
    """
    graph = cone_dag(intensity=8)
    best = exhaustive(graph, memory_weight=0.0)
    everything = len(graph.interior) + len(graph.required)
    return {
        "materialised": len(best.materialised),
        "everything": everything,
        "duplication": round(duplication_factor(graph, sorted(best.materialised)), 4),
        "flops": best.flops,
        "unfused_flops": materialise_everything(graph).flops,
    }


def expensive_memory_means_recompute() -> dict:
    """And what it picks when traffic dominates.

    Less, and past a point nothing beyond what it has to. A write it can avoid is worth more
    than the arithmetic it costs to avoid it, so the search trades the cone for flops.
    """
    graph = cone_dag(intensity=8)
    cheap = exhaustive(graph, memory_weight=0.1)
    dear = exhaustive(graph, memory_weight=100.0)
    return {
        "at_cheap_memory": len(cheap.materialised),
        "at_dear_memory": len(dear.materialised),
        "fell": len(dear.materialised) < len(cheap.materialised),
        "duplication_at_dear_memory": round(
            duplication_factor(graph, sorted(dear.materialised)), 3
        ),
    }


def compare_graphs(memory_weight: float = MEMORY_WEIGHT) -> list[dict]:
    """The rule against the search on three shapes.

    Right on the chain, right on the cone that does work, wrong on the cone that does not. That
    is about the right report for a heuristic: correct on what it was designed around and wrong
    in a direction somebody has to know about before they trust it.
    """
    rows = []
    for label, graph in (
        ("chain", chain_dag()),
        ("expensive cone", expensive_cone()),
        ("cheap cone", cheap_cone()),
    ):
        rule = materialise_reused(graph, memory_weight)
        best = exhaustive(graph, memory_weight)
        rows.append(
            {
                "graph": label,
                "rule": round(rule.cost, 2),
                "best": round(best.cost, 2),
                "regret": round(rule.cost / best.cost, 4),
                "matches": abs(rule.cost - best.cost) < 1e-9,
            }
        )
    return rows


def search_size(graph: Dag | None = None) -> dict:
    """How many plans the search has to look at.

    Two to the interior count. The default cone has thirty two and a cone twenty deep has two
    million, which is where a rule stops being a convenience and starts being the only option.
    """
    target = graph if graph is not None else cheap_cone()
    return {
        "interior": len(target.interior),
        "plans": 2 ** len(target.interior),
        "at_depth_twenty": 2 ** len(cone_dag(depth=20).interior),
    }


def a_large_graph_is_refused() -> bool:
    """Whether the search refuses a graph it cannot enumerate rather than trying."""
    try:
        exhaustive(cone_dag(depth=30), limit=16)
    except ScheduleError:
        return True
    return False


def materialising_an_input_is_refused() -> bool:
    """Whether asking to write a value that is already in memory is caught."""
    try:
        cost_of(chain_dag(), (0,))
    except ScheduleError:
        return True
    return False


def a_forward_reference_is_refused() -> bool:
    """Whether a producer list that reads a later value is refused at construction.

    Every cost here walks producers backwards and trusts the ordering, so a graph that is not
    topologically sorted would not fail, it would give a wrong number quietly. Refusing it where
    the graph is built is the only place the check is cheap.
    """
    try:
        Dag(producers=((1,), ()), sizes=(1, 1), flops=(1, 1), outputs=(0,))
    except ConfigError:
        return True
    return False


@dataclass
class SearchReport:
    """One graph's numbers, packaged."""

    graph: str
    best: FusionPlan = field(default_factory=FusionPlan)
    rule_cost: float = 0.0
    plans: int = 0

    @property
    def regret(self) -> float:
        """What the rule gives up against the search."""
        if self.best.cost == 0:
            return 1.0
        return self.rule_cost / self.best.cost

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "graph": self.graph,
            "best_cost": round(self.best.cost, 2),
            "rule_cost": round(self.rule_cost, 2),
            "regret": round(self.regret, 4),
            "plans": self.plans,
        }


def report_for(graph: Dag, label: str = "") -> SearchReport:
    """Assemble the summary for one graph."""
    return SearchReport(
        graph=label,
        best=exhaustive(graph),
        rule_cost=materialise_reused(graph).cost,
        plans=2 ** len(graph.interior),
    )
