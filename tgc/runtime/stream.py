from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

from tgc.analysis.cost import annotate_matmuls, node_flops
from tgc.analysis.critical import critical_path
from tgc.errors import ConfigError, ScheduleError
from tgc.ir.builder import branching_graph, elementwise_chain, layernorm_graph, mlp_graph
from tgc.ir.graph import Graph

# Running a graph on several queues at once, and what the queues cost to keep in step.
#
# analysis/critical.py says how much parallelism a graph has: the work over the span, which is
# what infinitely many queues would achieve. This is the other half, which is what a finite
# number of them actually achieves and what the coordination costs. A queue runs its own tasks
# in order, and a task that reads a value produced on another queue has to wait for a signal,
# and every signal is a real event with a real cost.
#
# Three assignments are compared and they behave differently for a reason worth stating. Round
# robin ignores the graph, so it puts a value and its consumer on different queues about as
# often as not and pays a signal for each. Level based puts everything at one depth together,
# which is tidy and pays a signal at every level boundary. List scheduling puts a task wherever
# it can start soonest, which follows the graph.
#
# What the measurements say is that on the graphs a compiler is usually handed, none of this
# matters, because those graphs are chains and a chain has no parallelism to find. On the one
# fixture with branches in it, list scheduling and level based placement both reach the critical
# path bound of three and two thirds and round robin reaches two, at two and a half times the
# signals.
#
# The queue count stops mattering at four on that fixture, and not because of the signals: the
# signal count stops growing at four as well. It stops because the graph has run out of
# parallelism, which is the bound the other file computed. Signals do decide the answer once
# they are expensive enough, and the sweep says how expensive: the break even on this graph is
# about forty times the cost of a small kernel.

ASSIGNMENTS = ("round robin", "by level", "list scheduling")


@dataclass
class Task:
    """One node placed on a queue, with when it ran."""

    name: str
    stream: int
    start: float
    duration: float

    @property
    def finish(self) -> float:
        """When the task completed."""
        return self.start + self.duration

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "task": self.name,
            "stream": self.stream,
            "start": round(self.start, 3),
            "finish": round(self.finish, 3),
        }


@dataclass
class Schedule:
    """Every task, and the signals the queues had to exchange."""

    tasks: list[Task] = field(default_factory=list)
    signals: int = 0
    streams: int = 1

    @property
    def makespan(self) -> float:
        """When the last task finished."""
        return max((task.finish for task in self.tasks), default=0.0)

    @property
    def work(self) -> float:
        """Total time the tasks would take on one queue."""
        return sum(task.duration for task in self.tasks)

    @property
    def occupancy(self) -> float:
        """Share of the queue time that was doing something."""
        total = self.makespan * self.streams
        if total <= 0:
            return 0.0
        return self.work / total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "streams": self.streams,
            "tasks": len(self.tasks),
            "makespan": round(self.makespan, 3),
            "signals": self.signals,
            "occupancy": round(self.occupancy, 4),
        }


def durations(graph: Graph) -> dict[str, float]:
    """How long each node takes, from the same cost model the roofline uses.

    Scaled down so the numbers are readable rather than in units of an operation. Nothing here
    depends on the scale, because every comparison in this file is a ratio.
    """
    prepared = annotate_matmuls(graph)
    return {node.name: max(node_flops(node) / 1e3, 1.0) for node in prepared.nodes}


def assign(graph: Graph, streams: int, policy: str) -> dict[str, int]:
    """Which queue each node runs on.

    Round robin and level based are decided before anything runs, which is why they are cheap
    and why they ignore the shape of the graph. List scheduling cannot be: it needs to know when
    each queue becomes free, so it is decided during the simulation and this function returns an
    empty assignment for it.
    """
    if streams < 1:
        raise ConfigError(f"there has to be at least one queue, got {streams}")
    if policy not in ASSIGNMENTS:
        raise ConfigError(f"unknown assignment {policy!r}, expected one of {list(ASSIGNMENTS)}")

    if policy == "round robin":
        return {node.name: index % streams for index, node in enumerate(graph.nodes)}
    if policy == "by level":
        depth: dict[str, int] = {value.name: 0 for value in graph.inputs}
        placement: dict[str, int] = {}
        counters: dict[int, int] = {}
        for node in graph.nodes:
            level = max((depth.get(name, 0) for name in node.inputs), default=0) + 1
            depth[node.name] = level
            position = counters.get(level, 0)
            counters[level] = position + 1
            placement[node.name] = position % streams
        return placement
    return {}


def simulate(graph: Graph, streams: int = 4, policy: str = "list scheduling") -> Schedule:
    """Run the graph across the queues and record when everything happened.

    A task starts when its inputs have finished and its queue is free. A signal is counted
    whenever a task reads a value produced on a different queue, which is what a real runtime
    records as an event and waits on, and it is the only cost here that a single queue does not
    pay.
    """
    if streams < 1:
        raise ConfigError(f"there has to be at least one queue, got {streams}")
    if policy not in ASSIGNMENTS:
        raise ConfigError(f"unknown assignment {policy!r}")

    cost = durations(graph)
    placement = assign(graph, streams, policy)
    ready_at = {value.name: 0.0 for value in graph.inputs}
    stream_of = {value.name: -1 for value in graph.inputs}
    free_at = [0.0] * streams

    schedule = Schedule(streams=streams)
    remaining = list(graph.nodes)
    while remaining:
        runnable = [node for node in remaining if all(name in ready_at for name in node.inputs)]
        if not runnable:
            raise ScheduleError("nothing can run, so the graph is not in topological order")
        node = runnable[0]
        inputs_done = max((ready_at[name] for name in node.inputs), default=0.0)

        if policy == "list scheduling":
            chosen = min(range(streams), key=lambda index: max(free_at[index], inputs_done))
        else:
            chosen = placement[node.name]

        start = max(free_at[chosen], inputs_done)
        duration = cost[node.name]
        free_at[chosen] = start + duration
        ready_at[node.name] = start + duration
        schedule.signals += sum(
            1 for name in node.inputs if stream_of.get(name, chosen) != chosen
        )
        stream_of[node.name] = chosen
        schedule.tasks.append(
            Task(name=node.name, stream=chosen, start=start, duration=duration)
        )
        remaining.remove(node)
    return schedule


def one_queue_is_the_serial_time(graph: Graph | None = None) -> dict:
    """The baseline, which is every task one after another.

    Worth measuring rather than computing, because it is the number every speedup here is
    against and a simulation that got it wrong would make every other number wrong by the same
    factor and in the same direction.
    """
    target = graph if graph is not None else branching_graph()
    schedule = simulate(target, streams=1)
    return {
        "makespan": round(schedule.makespan, 3),
        "work": round(schedule.work, 3),
        "equal": round(schedule.makespan, 6) == round(schedule.work, 6),
        "signals": schedule.signals,
    }


def speedup(graph: Graph, streams: int, policy: str = "list scheduling") -> float:
    """How much faster a graph runs on several queues than on one."""
    serial = simulate(graph, streams=1).makespan
    parallel = simulate(graph, streams=streams, policy=policy).makespan
    return serial / parallel if parallel else 1.0


def stream_sweep(
    counts: Sequence[int] = (1, 2, 4, 8, 16), graph: Graph | None = None
) -> list[dict]:
    """Speedup and signals against the number of queues.

    The speedup stops climbing at the parallelism the graph has and the signals keep climbing,
    so past a point every queue added is a cost with no return. Where that point sits is a
    property of the graph rather than of the runtime, which is the argument for computing the
    work over the span before choosing how many queues to open.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    target = graph if graph is not None else branching_graph()
    rows = []
    for count in counts:
        schedule = simulate(target, streams=count)
        rows.append(
            {
                "streams": count,
                "speedup": round(speedup(target, count), 3),
                "signals": schedule.signals,
                "occupancy": round(schedule.occupancy, 4),
            }
        )
    return rows


def the_speedup_stops_at_the_parallelism(graph: Graph | None = None) -> dict:
    """Whether the simulation respects the bound analysis/critical.py computed.

    It has to, and checking it is the point. The work over the span is a bound on any schedule
    on any number of queues, so a simulation that beat it would be running tasks before their
    inputs existed, and the speedup would look like an improvement.
    """
    target = graph if graph is not None else branching_graph()
    bound = critical_path(target).parallelism
    best = max(speedup(target, count) for count in (1, 2, 4, 8, 16, 32))
    return {
        "bound": round(bound, 3),
        "best_speedup": round(best, 3),
        "within_the_bound": best <= bound + 1e-6,
    }


def list_scheduling_reaches_the_bound(graph: Graph | None = None) -> dict:
    """How close each assignment gets to that bound.

    List scheduling reaches it on the branching fixture and the other two do not, which is the
    whole argument for deciding placement during the run rather than before it. The two static
    policies are choosing a queue without knowing when it will be free.
    """
    target = graph if graph is not None else branching_graph()
    bound = critical_path(target).parallelism
    return {
        "bound": round(bound, 3),
        **{
            policy.replace(" ", "_"): round(speedup(target, 8, policy), 3)
            for policy in ASSIGNMENTS
        },
    }


def compare_assignments(graph: Graph | None = None, streams: int = 8) -> list[dict]:
    """Every assignment on one graph, with its signals."""
    target = graph if graph is not None else branching_graph()
    rows = []
    for policy in ASSIGNMENTS:
        schedule = simulate(target, streams=streams, policy=policy)
        row = schedule.as_dict()
        row["policy"] = policy
        row["speedup"] = round(speedup(target, streams, policy), 3)
        rows.append(row)
    return rows


def round_robin_pays_for_ignoring_the_graph(streams: int = 8) -> dict:
    """How many more signals a placement that ignores dependencies needs.

    Round robin puts consecutive nodes on consecutive queues, so a chain of dependent work
    signals at every step. On a graph that is mostly a chain that is one signal per node, which
    is the worst possible number and is what a placement decided without looking produces.
    """
    graph = branching_graph()
    rows = {row["policy"]: row for row in compare_assignments(graph, streams)}
    return {
        "round_robin": rows["round robin"]["signals"],
        "by_level": rows["by level"]["signals"],
        "list_scheduling": rows["list scheduling"]["signals"],
        "nodes": len(graph.nodes),
    }


def signal_cost_sweep(
    costs: Sequence[float] = (0.0, 0.5, 2.0, 8.0, 64.0), streams: int = 8
) -> list[dict]:
    """What the queues are worth once a signal is not free.

    A signal is a fixed cost per cross queue read, so the total is the signal count times that
    cost, and the point where more queues stop paying moves with it. It takes a lot: this graph
    is still ahead at a signal cost of eight and only loses at sixty four, which is because list
    scheduling produced seven signals for fifteen nodes rather than one per node.
    """
    if not costs:
        raise ConfigError("there is nothing to sweep")
    graph = branching_graph()
    serial = simulate(graph, streams=1).makespan
    rows = []
    for cost in costs:
        schedule = simulate(graph, streams=streams)
        total = schedule.makespan + schedule.signals * cost
        rows.append(
            {
                "signal_cost": cost,
                "makespan": round(total, 3),
                "speedup": round(serial / total, 3) if total else 0.0,
                "worth_it": total < serial,
            }
        )
    return rows


def expensive_signals_undo_the_parallelism(streams: int = 8) -> dict:
    """Where in that sweep the queues stop being worth opening."""
    rows = signal_cost_sweep(streams=streams)
    paying = [row for row in rows if row["worth_it"]]
    return {
        "free_signals_speedup": rows[0]["speedup"],
        "expensive_signals_speedup": rows[-1]["speedup"],
        "still_worth_it": len(paying),
        "of": len(rows),
    }


def compare_graphs(streams: int = 8) -> list[dict]:
    """Every fixture on the same number of queues.

    Three of the four gain nothing at all, because they are chains and a chain cannot be run in
    parallel however many queues are offered. That is the same result analysis/critical.py
    reported from the other direction, arrived at by running the thing rather than by measuring
    the graph, which is why it is worth having twice.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(8)),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
        ("branching", branching_graph()),
    ):
        rows.append(
            {
                "graph": label,
                "speedup": round(speedup(graph, streams), 3),
                "parallelism": round(critical_path(graph).parallelism, 3),
            }
        )
    return rows


def only_the_branching_graph_gains(streams: int = 8) -> dict:
    """Which fixtures are worth running on more than one queue."""
    rows = {row["graph"]: row for row in compare_graphs(streams)}
    gained = [name for name, row in rows.items() if row["speedup"] > 1.05]
    return {"graphs": len(rows), "gained": gained}


def the_simulation_agrees_with_the_analysis(streams: int = 32) -> dict:
    """Whether the measured speedup matches the predicted parallelism, per fixture.

    Three of the four agree exactly and the fourth does not, for a reason that is a difference
    between the models rather than a mistake in either. This file floors a task at one unit of
    time, because a queue cannot dispatch something in no time at all, and the analysis does
    not. A layernorm is mostly tiny nodes, so the floor changes their relative costs and the
    simulation finds fourteen percent of parallelism where the analysis finds none.

    Two files computing the same number from different directions is still the strongest check
    either of them gets. What it caught here is an assumption, which is the useful kind of
    disagreement.
    """
    rows = []
    for label, graph in (
        ("chain", elementwise_chain(8)),
        ("layernorm", layernorm_graph()),
        ("mlp", mlp_graph()),
        ("branching", branching_graph()),
    ):
        predicted = critical_path(graph).parallelism
        measured = speedup(graph, streams)
        rows.append(
            {
                "graph": label,
                "predicted": round(predicted, 3),
                "measured": round(measured, 3),
                "agree": abs(predicted - measured) < 0.05,
            }
        )
    return {
        "graphs": len(rows),
        "agreeing": sum(1 for row in rows if row["agree"]),
        "rows": rows,
    }


def occupancy_falls_as_queues_are_added(graph: Graph | None = None) -> dict:
    """How much of the queue time is spent doing anything.

    Falls like one over the queue count once the graph has run out of parallelism, which is the
    same fact as the speedup flattening and is the more useful way to report it: an occupancy of
    an eighth says seven queues are open and idle, and that is a number an operator can act on.
    """
    target = graph if graph is not None else branching_graph()
    rows = {row["streams"]: row for row in stream_sweep(graph=target)}
    return {
        "at_one": rows[1]["occupancy"],
        "at_four": rows[4]["occupancy"],
        "at_sixteen": rows[16]["occupancy"],
        "falling": rows[16]["occupancy"] < rows[4]["occupancy"] < rows[1]["occupancy"],
    }


def every_task_runs_after_its_inputs(graph: Graph | None = None, streams: int = 8) -> dict:
    """Whether the simulation ever starts a task early.

    The check that makes the speedups worth reading. Starting a task before its input finished
    would shorten the makespan, which is exactly the direction a scheduling bug is welcomed in,
    so it is checked directly against the finish times rather than assumed from the loop.
    """
    target = graph if graph is not None else branching_graph()
    schedule = simulate(target, streams=streams)
    finished = {task.name: task.finish for task in schedule.tasks}
    violations = 0
    for task in schedule.tasks:
        for name in target.node(task.name).inputs:
            if name in finished and finished[name] > task.start + 1e-9:
                violations += 1
    return {"tasks": len(schedule.tasks), "violations": violations}


def a_queue_runs_one_task_at_a_time(graph: Graph | None = None, streams: int = 4) -> dict:
    """Whether two tasks ever overlap on the same queue.

    The other way the simulation could cheat. A queue that ran two things at once would report a
    shorter makespan and a higher occupancy, and both would look like the scheduler working.
    """
    target = graph if graph is not None else branching_graph()
    schedule = simulate(target, streams=streams)
    overlaps = 0
    for index in range(streams):
        tasks = sorted(
            (task for task in schedule.tasks if task.stream == index),
            key=lambda task: task.start,
        )
        for earlier, later in itertools.pairwise(tasks):
            if later.start + 1e-9 < earlier.finish:
                overlaps += 1
    return {"streams": streams, "overlaps": overlaps}
