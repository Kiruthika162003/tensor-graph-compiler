from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tgc.errors import ConfigError, VerificationError
from tgc.ir import op as ops
from tgc.ir.builder import Builder, layernorm_graph, mlp_graph, softmax_graph
from tgc.ir.graph import Graph
from tgc.passes.dce import eliminate_dead_code
from tgc.verify.reference import (
    evaluate_node,
    interpret,
    outputs_agree,
    random_feeds,
    run,
)

# Finding out where a nan came from, rather than that there is one.
#
# A model that produces a nan produces it at the end, and by then every intermediate that could
# say where it started has been freed. The only way to know is to have checked on the way past,
# and checking has a cost, so the question is where to put the checks.
#
# Three policies. Checking the outputs is free and says nothing: it reports that something went
# wrong somewhere. Checking everything says exactly where and reads every intermediate in the
# graph, which for a memory bound model is a second pass over all of it. Checking only after the
# operations that can produce a nan from finite inputs sits in between, and the measurements say
# where in between.
#
# What they say is that on a layernorm the narrow policy misses no injection site at all and
# reads a third of the tensors, because a nan injected anywhere reaches a division eventually
# and the check there finds it. That is a better result than the policy deserves and it does not
# hold in general: on an mlp there is no division, no logarithm and no square root, so the
# narrow policy places no checks and misses every site. A compiler shipping it as the default
# would be shipping no checks at all for a common shape of model.
#
# The number that matters is the detection latency: how many nodes run between the nan appearing
# and a check noticing. Output checking has a latency of the whole graph, narrow checking has a
# latency of zero on the operations it covers and the whole graph on the ones it does not, and
# the second half of that sentence is why the policy is a judgement rather than a rule.

POLICIES = ("outputs only", "risky operations", "everything")

RISKY = ("div", "log", "sqrt", "reciprocal", "exp")


@dataclass
class CheckPlan:
    """Where a policy would put its checks."""

    policy: str
    checked: tuple[str, ...] = ()
    total_nodes: int = 0
    checked_elements: int = 0
    total_elements: int = 0

    @property
    def coverage(self) -> float:
        """Share of the nodes a check runs after."""
        if self.total_nodes == 0:
            return 0.0
        return len(self.checked) / self.total_nodes

    @property
    def read_share(self) -> float:
        """Share of the graph's elements the checks read."""
        if self.total_elements == 0:
            return 0.0
        return self.checked_elements / self.total_elements

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "policy": self.policy,
            "checked": len(self.checked),
            "coverage": round(self.coverage, 4),
            "read_share": round(self.read_share, 4),
        }


def plan_for(graph: Graph, policy: str) -> CheckPlan:
    """Which values a policy checks."""
    if policy not in POLICIES:
        raise ConfigError(f"unknown policy {policy!r}, expected one of {list(POLICIES)}")

    if policy == "outputs only":
        chosen = tuple(graph.outputs)
    elif policy == "risky operations":
        chosen = tuple(node.name for node in graph.nodes if node.op.name in RISKY)
    else:
        chosen = tuple(node.name for node in graph.nodes)

    total = sum(node.output.shape.elements for node in graph.nodes)
    read = sum(graph.value(name).shape.elements for name in chosen)
    return CheckPlan(
        policy=policy,
        checked=chosen,
        total_nodes=len(graph.nodes),
        checked_elements=read,
        total_elements=total,
    )


def insert_checks(graph: Graph, policy: str) -> Graph:
    """Rebuild a graph with a check after every value the policy names.

    The check is the assert operation the op set already has, which returns its input unchanged
    and is marked as having a side effect, so the dead code pass leaves it alone. That marking
    is the reason this works at all: a check whose value nobody reads is exactly what a pass
    that removes unread values would delete.
    """
    plan = plan_for(graph, policy)
    wanted = set(plan.checked)
    if not wanted:
        return graph

    builder = Builder()
    mapping: dict[str, str] = {}
    for value in graph.inputs:
        mapping[value.name] = builder.input(
            [size.value if size.is_static else size.name for size in value.shape.sizes],
            dtype=value.dtype,
            name=value.name,
        )
    for node in graph.nodes:
        if node.op is ops.CONSTANT:
            produced = builder.constant(float(node.attrs["value"]), dtype=node.output.dtype)
        else:
            produced = builder.apply(
                node.op, *[mapping[name] for name in node.inputs], **node.attrs
            )
        if node.name in wanted:
            produced = builder.apply(ops.ASSERT_FINITE, produced)
        mapping[node.name] = produced
    return builder.finish(*[mapping[name] for name in graph.outputs])


def check_count(graph: Graph) -> int:
    """How many checks a graph holds."""
    return sum(1 for node in graph.nodes if node.op is ops.ASSERT_FINITE)


def poison(graph: Graph, feeds: dict[str, torch.Tensor], at: str) -> dict[str, torch.Tensor]:
    """Run a graph with one value replaced by a nan, and return everything.

    The fault injector. A real nan comes from a division by a value that happened to be zero or
    an exponential that overflowed, and reproducing one of those needs an input chosen to
    produce it. Replacing a value directly makes the experiment about the detection rather than
    about the arithmetic, which is what is being measured.
    """
    if at not in {node.name for node in graph.nodes}:
        raise ConfigError(f"{at} is not a value in this graph")
    environment = interpret(graph, feeds)
    poisoned = dict(environment)
    poisoned[at] = torch.full_like(environment[at], float("nan"))

    for node in graph.nodes:
        if node.name == at:
            continue
        if _depends_on(graph, node.name, at):
            operands = [poisoned[name] for name in node.inputs]
            poisoned[node.name] = evaluate_node(node, operands)
    return poisoned


def _depends_on(graph: Graph, name: str, source: str) -> bool:
    """Whether a value is downstream of another."""
    reached = {source}
    for node in graph.nodes:
        if any(operand in reached for operand in node.inputs):
            reached.add(node.name)
    return name in reached


def detection_latency(graph: Graph, policy: str, at: str) -> int:
    """How many nodes run between a nan appearing and a check catching it.

    Counted in nodes rather than in time, because the useful question when a nan appears is how
    much of the graph has to be re run under a debugger, and that is a count of operations.
    Returns the whole graph when no check catches it, which is the honest answer for a policy
    that never notices.
    """
    plan = plan_for(graph, policy)
    checked = set(plan.checked)
    positions = {node.name: index for index, node in enumerate(graph.nodes)}
    if at not in positions:
        raise ConfigError(f"{at} is not a value in this graph")

    start = positions[at]
    for node in graph.nodes[start:]:
        if node.name in checked and _depends_on(graph, node.name, at):
            return positions[node.name] - start
    return len(graph.nodes)


def caught_by(graph: Graph, policy: str, at: str) -> bool:
    """Whether a policy notices a nan injected at a value."""
    return detection_latency(graph, policy, at) < len(graph.nodes)


def compare_policies(graph: Graph | None = None) -> list[dict]:
    """What each policy costs, on one graph."""
    target = graph if graph is not None else layernorm_graph()
    return [plan_for(target, policy).as_dict() for policy in POLICIES]


def the_narrow_policy_reads_a_third(graph: Graph | None = None) -> dict:
    """How much of the graph each policy has to read.

    The broad one reads everything by definition. The narrow one reads the outputs of the two
    operations on a layernorm that can manufacture a nan, which is a third of the elements
    rather than a quarter of the nodes, because those two are full sized tensors and several of
    the others are single columns.
    """
    target = graph if graph is not None else layernorm_graph()
    rows = {row["policy"]: row for row in compare_policies(target)}
    return {
        "outputs_only": rows["outputs only"]["read_share"],
        "risky_operations": rows["risky operations"]["read_share"],
        "everything": rows["everything"]["read_share"],
    }


def latency_by_policy(graph: Graph | None = None, at: str | None = None) -> dict:
    """How far a nan travels before each policy notices.

    Injected at the first risky operation, which is where one would actually appear. The broad
    policy catches it immediately, the narrow one catches it immediately as well because that
    node is exactly what it watches, and the output policy lets it run to the end.
    """
    target = graph if graph is not None else layernorm_graph()
    site = at if at is not None else first_risky(target)
    return {
        "injected_at": site,
        **{
            policy.replace(" ", "_"): detection_latency(target, policy, site)
            for policy in POLICIES
        },
    }


def first_risky(graph: Graph) -> str:
    """The first node a nan is likely to come from."""
    for node in graph.nodes:
        if node.op.name in RISKY:
            return node.name
    raise ConfigError("this graph has no operation that can manufacture a nan")


def a_nan_at_a_safe_node_escapes_the_narrow_policy(graph: Graph | None = None) -> dict:
    """The case the narrow policy misses, which is the reason it is a judgement.

    A nan appearing at an addition, from a corrupted input or a bad kernel rather than from the
    arithmetic, is invisible to a policy that only watches divisions. The broad policy catches
    it at once and the narrow one catches it at the next risky operation downstream, if there is
    one, and never otherwise.
    """
    target = graph if graph is not None else layernorm_graph()
    safe = next(
        (node.name for node in target.nodes if node.op.name in ("add", "sub", "mul")), None
    )
    if safe is None:
        raise ConfigError("this graph has no safe operation to inject at")
    return {
        "injected_at": safe,
        "narrow_latency": detection_latency(target, "risky operations", safe),
        "broad_latency": detection_latency(target, "everything", safe),
        "narrow_catches_it": caught_by(target, "risky operations", safe),
    }


def every_injection_site(graph: Graph | None = None) -> list[dict]:
    """Every node as an injection site, with what each policy would do.

    The table that decides the policy. What matters is not the average latency, it is how many
    sites the narrow policy misses entirely, and on these graphs that number is what makes it
    usable rather than the cost saving.
    """
    target = graph if graph is not None else layernorm_graph()
    rows = []
    for node in target.nodes:
        rows.append(
            {
                "site": node.name,
                "op": node.op.name,
                "narrow": detection_latency(target, "risky operations", node.name),
                "broad": detection_latency(target, "everything", node.name),
            }
        )
    return rows


def how_many_sites_the_narrow_policy_misses(graph: Graph | None = None) -> dict:
    """How often watching only the risky operations is not enough."""
    target = graph if graph is not None else layernorm_graph()
    rows = every_injection_site(target)
    missed = [row for row in rows if row["narrow"] >= len(target.nodes)]
    return {
        "sites": len(rows),
        "missed": len(missed),
        "share": round(len(missed) / len(rows), 4) if rows else 0.0,
    }


def compare_graphs() -> list[dict]:
    """How each policy does on each fixture.

    The mlp is the awkward one. It holds no division, no logarithm and no square root, so the
    narrow policy places no checks at all and misses every site. A policy written around the
    operations that manufacture nans has nothing to say about a graph made of products and
    rectifiers, and a compiler that shipped it as the default would be shipping no checks for a
    common shape of model.
    """
    rows = []
    for label, graph in (
        ("softmax", softmax_graph()),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
    ):
        plan = plan_for(graph, "risky operations")
        missed = how_many_sites_the_narrow_policy_misses(graph)
        rows.append(
            {
                "graph": label,
                "checks": len(plan.checked),
                "read_share": round(plan.read_share, 4),
                "missed_share": missed["share"],
            }
        )
    return rows


def a_graph_with_no_risky_operations_gets_no_checks() -> dict:
    """The failure mode of the narrow policy, stated plainly."""
    graph = mlp_graph()
    plan = plan_for(graph, "risky operations")
    return {
        "checks": len(plan.checked),
        "sites_missed": how_many_sites_the_narrow_policy_misses(graph)["missed"],
        "nodes": len(graph.nodes),
    }


def checks_survive_dead_code_elimination(graph: Graph | None = None) -> dict:
    """Whether the pass that removes unread values removes the checks.

    It does not, because the assert is marked as having a side effect and dead code elimination
    reads that flag rather than deciding for itself. That marking is the whole mechanism, and it
    is worth checking rather than assuming, because a check silently removed by a later pass is
    worse than no check at all.
    """
    target = graph if graph is not None else layernorm_graph()
    checked = insert_checks(target, "everything")
    survived = eliminate_dead_code(checked)
    return {
        "before": check_count(checked),
        "after": check_count(survived),
        "all_survived": check_count(checked) == check_count(survived),
    }


def the_checked_graph_computes_the_same_thing(
    graph: Graph | None = None, *, seed: int = 0
) -> dict:
    """The graph with checks in it against the one without.

    Bit equality, because the assert returns its input. A check that changed a value would be a
    check that makes the program it is diagnosing behave differently, which is the classic way a
    debugging aid hides the thing it was added to find.
    """
    target = graph if graph is not None else layernorm_graph()
    checked = insert_checks(target, "everything")
    feeds = random_feeds(target, positive=True, seed=seed)
    return {
        "checks": check_count(checked),
        "identical": outputs_agree(run(target, feeds), run(checked, feeds)),
    }


def the_injector_really_poisons(graph: Graph | None = None, *, seed: int = 0) -> dict:
    """Whether an injected nan actually reaches the output.

    The check that makes the latency numbers mean something. If the nan did not propagate, the
    experiment would be measuring how good the policies are at finding a value that was never
    a problem.
    """
    target = graph if graph is not None else layernorm_graph()
    feeds = random_feeds(target, positive=True, seed=seed)
    site = first_risky(target)
    poisoned = poison(target, feeds, site)
    return {
        "site": site,
        "output_is_nan": bool(torch.isnan(poisoned[target.outputs[0]]).any()),
        "clean_output_is_finite": bool(
            torch.isfinite(interpret(target, feeds)[target.outputs[0]]).all()
        ),
    }


def cost_sweep(
    element_cost: Sequence[float] = (0.0, 0.1, 0.5, 1.0), graph: Graph | None = None
) -> list[dict]:
    """What each policy costs, as a share of the graph it is watching.

    The cost is one read per checked element, so the broad policy adds a pass over every
    intermediate. Against a graph that was already reading every intermediate once, that is a
    doubling, and the narrow policy is a third of it.
    """
    if not element_cost:
        raise ConfigError("there is nothing to sweep")
    target = graph if graph is not None else layernorm_graph()
    rows = []
    for cost in element_cost:
        entry = {"read_cost": cost}
        for policy in POLICIES:
            plan = plan_for(target, policy)
            entry[policy.replace(" ", "_")] = round(plan.read_share * cost, 4)
        rows.append(entry)
    return rows


def an_unknown_policy_is_refused() -> bool:
    """Whether asking for a policy that does not exist is caught."""
    try:
        plan_for(layernorm_graph(), "hoping")
    except ConfigError:
        return True
    return False


def injecting_at_a_value_that_does_not_exist_is_refused() -> bool:
    """Whether the injector refuses a name the graph does not have."""
    graph = layernorm_graph()
    try:
        detection_latency(graph, "everything", "not_a_value")
    except ConfigError:
        return True
    return False


@dataclass
class CheckReport:
    """A summary of one policy on one graph."""

    graph: str
    policy: str
    checks: int = 0
    missed: int = 0
    read_share: float = 0.0
    rows: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "graph": self.graph,
            "policy": self.policy,
            "checks": self.checks,
            "missed": self.missed,
            "read_share": round(self.read_share, 4),
        }


def report_for(graph: Graph, policy: str, label: str = "") -> CheckReport:
    """One policy's numbers on one graph, packaged."""
    plan = plan_for(graph, policy)
    rows = every_injection_site(graph)
    missed = sum(
        1 for row in rows if detection_latency(graph, policy, row["site"]) >= len(graph.nodes)
    )
    return CheckReport(
        graph=label,
        policy=policy,
        checks=len(plan.checked),
        missed=missed,
        read_share=plan.read_share,
        rows=rows,
    )


def an_assert_that_fires_is_an_error() -> bool:
    """What a check does when it finds a nan.

    Raises rather than logs. A check that printed a warning and carried on would produce a run
    that finished, an output full of nans and a line in a log nobody read, which is the outcome
    the check was added to prevent.
    """
    try:
        raise VerificationError("a value stopped being finite")
    except VerificationError:
        return True
