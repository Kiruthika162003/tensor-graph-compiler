from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tgc.analysis.liveness import Interval, compute_intervals, peak_bytes
from tgc.codegen.emit import arena_elements, emit
from tgc.errors import ConfigError, PassError
from tgc.ir.graph import Graph
from tgc.memory.planner import Allocation, Plan, plan_largest_first, validate_plan
from tgc.schedule.order import depth_first_order
from tgc.verify.reference import outputs_agree, random_feeds, run, to_torch

# Letting an operation write over the buffer it read.
#
# The allocator already reuses storage between values whose lifetimes do not overlap. This is
# the stronger version: a value and the one computed from it do overlap, at exactly one
# instant, and an elementwise operation that reads element i and writes element i can be given
# the same buffer for both without ever reading something it has already overwritten.
#
# Five conditions, and skipping any of them produces wrong numbers rather than an error. The
# operation must be elementwise. It must not read the donor twice, because the second read
# happens after the first write. The donor must have exactly one use, or the other reader gets
# the overwritten version. The shapes must match exactly, since a broadcast writes more
# elements than it read. And the donor must not be a graph input, because the caller owns that
# memory and will be surprised to find it changed.


@dataclass
class Donation:
    """One value whose buffer is handed to the node reading it."""

    donor: str
    receiver: str

    def as_dict(self) -> dict[str, str]:
        """Flat mapping for logging."""
        return {"donor": self.donor, "receiver": self.receiver}


@dataclass
class InplaceReport:
    """Which buffers a graph can donate."""

    donations: list[Donation] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Buffers eliminated."""
        return len(self.donations)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "donations": self.count,
            "refused": len(self.refused),
            "reasons": sorted(set(self.refused.values())),
        }


def can_donate(graph: Graph, donor: str, receiver: str) -> tuple[bool, str]:
    """Whether one value's buffer can be written over by the node reading it.

    Returns the reason as well as the answer, because every refusal here corresponds to a bug
    somebody would otherwise introduce, and having them named makes the report readable.
    """
    node = graph.producer_of(receiver)
    if node is None:
        return False, "the receiver is a graph input"
    if donor not in node.inputs:
        return False, "the receiver does not read the donor"
    if not node.op.is_elementwise:
        return False, "the receiver is not elementwise"
    if not node.op.reads_input_once:
        return False, "the receiver reads its input more than once"
    if node.inputs.count(donor) > 1:
        return False, "the donor appears twice in the same node"
    if graph.producer_of(donor) is None:
        return False, "the donor is a graph input and the caller owns it"
    if donor in graph.outputs:
        return False, "the donor is a graph output"
    if graph.use_counts().get(donor, 0) != 1:
        return False, "the donor has another reader"
    if graph.value(donor).shape != node.output.shape:
        return False, "the shapes differ, so the write covers more than the read"
    if graph.value(donor).dtype is not node.output.dtype:
        return False, "the types differ, so the elements are different widths"
    return True, ""


def find_donations(graph: Graph) -> InplaceReport:
    """Every buffer that can be written over, and why the rest cannot."""
    report = InplaceReport()
    taken: set[str] = set()

    for node in graph.nodes:
        for donor in node.inputs:
            if donor in taken:
                continue
            allowed, reason = can_donate(graph, donor, node.name)
            if allowed:
                report.donations.append(Donation(donor=donor, receiver=node.name))
                taken.add(donor)
                break
            if donor not in report.refused:
                report.refused[donor] = reason
    return report


def donation_chains(graph: Graph) -> dict[str, str]:
    """Every donated value mapped to the head of the chain it belongs to.

    A donates to b and b donates to c means all three share one buffer, so all three belong to
    one chain named after a. Following the chain to its head is what stops c being placed
    wherever b used to live, since b no longer lives there.
    """
    parent: dict[str, str] = {}
    for donation in find_donations(graph).donations:
        parent[donation.receiver] = donation.donor

    heads: dict[str, str] = {}
    for name in graph.value_names:
        root = name
        seen = {name}
        while root in parent:
            root = parent[root]
            if root in seen:
                raise PassError(f"the donation chain through {name} is circular")
            seen.add(root)
        heads[name] = root
    return heads


def merged_intervals(graph: Graph) -> list[Interval]:
    """Liveness with every donation chain treated as one value.

    The correction the execution check forced, and the reason it is worth writing checks that
    run rather than checks that assert. Donation does not merely rename a buffer, it extends
    the donor's lifetime to cover the receiver's, so a plan computed from the undonated
    intervals is not valid after the offsets are collapsed onto each other. On a graph two
    branches wide the allocator had given two values the same offset because their lifetimes
    did not overlap, donation extended one of them across the other, and the answer was wrong
    while every per pair condition still held.

    Merging first and planning second gets it right by construction: a chain is one value,
    alive from the first write to the last read, and the allocator has never heard of
    donation.
    """
    order = depth_first_order(graph)
    intervals = compute_intervals(graph, order)
    heads = donation_chains(graph)

    grouped: dict[str, Interval] = {}
    for interval in intervals:
        head = heads.get(interval.name, interval.name)
        existing = grouped.get(head)
        if existing is None:
            grouped[head] = Interval(
                name=head, start=interval.start, end=interval.end, size=interval.size
            )
            continue
        grouped[head] = Interval(
            name=head,
            start=min(existing.start, interval.start),
            end=max(existing.end, interval.end),
            size=max(existing.size, interval.size),
        )
    return list(grouped.values())


def aliased_plan(graph: Graph) -> Plan:
    """A buffer plan where every donation chain occupies one slot."""
    grouped = merged_intervals(graph)
    plan = plan_largest_first(grouped)
    validate_plan(grouped, plan)

    heads = donation_chains(graph)
    placements = plan.by_name()
    allocations = []
    for name in sorted(graph.value_names):
        head = heads.get(name, name)
        allocation = placements[head]
        allocations.append(
            Allocation(name=name, offset=allocation.offset, size=allocation.size)
        )
    return Plan(allocations=allocations, strategy="largest first with donation")


def donation_saving(graph: Graph) -> dict:
    """Peak memory with donation and without, and the answer is the opposite of the guess.

    A chain has seven donation opportunities and saves nothing, because the allocator had
    already reached the floor and there was nothing left to take. A graph four branches wide
    has the same seven and saves a quarter of the arena, going below the peak of
    simultaneously live bytes, which is only possible because donation changes what
    simultaneously live means.
    """
    order = depth_first_order(graph)
    intervals = compute_intervals(graph, order)
    plain = plan_largest_first(intervals)
    validate_plan(intervals, plain)

    aliased = aliased_plan(graph)
    return {
        "donations": len(find_donations(graph).donations),
        "arena_plain": plain.arena_bytes,
        "arena_donated": aliased.arena_bytes,
        "peak_bytes": peak_bytes(intervals),
        "saved": plain.arena_bytes - aliased.arena_bytes,
    }


def refusal_reasons(graph: Graph) -> list[str]:
    """The distinct reasons donation was declined on a graph."""
    return sorted(set(find_donations(graph).refused.values()))


def compare_graphs(graphs: dict[str, Graph]) -> list[dict]:
    """Donation opportunities across several graphs."""
    if not graphs:
        raise ConfigError("there is nothing to compare")
    rows = []
    for name, graph in graphs.items():
        row = donation_saving(graph)
        row["graph"] = name
        row["elementwise_nodes"] = sum(1 for node in graph.nodes if node.op.is_elementwise)
        rows.append(row)
    return rows


def check_donation(graph: Graph, donor: str, receiver: str) -> None:
    """Raise with the reason if a donation is not allowed."""
    allowed, reason = can_donate(graph, donor, receiver)
    if not allowed:
        raise PassError(f"cannot donate {donor} to {receiver}: {reason}")


def donated_matches_reference(graph: Graph, seed: int = 0) -> bool:
    """Generate code against the donated plan and check it against the interpreter.

    The only check worth having here. A donated plan places two values that the liveness
    analysis says overlap into the same bytes, so validate_plan rejects it by construction and
    cannot be the safety argument. Running it is.
    """

    order = depth_first_order(graph)
    module = emit(graph, order, aliased_plan(graph))
    namespace: dict = {}
    exec(compile(module.source, "<tgc-inplace>", "exec"), namespace)

    feeds = random_feeds(graph, seed=seed, positive=True)
    arena = torch.zeros(arena_elements(module), dtype=to_torch(graph.inputs[0].dtype))
    produced = namespace["compiled"](arena, feeds)
    return outputs_agree(produced, run(graph, feeds))


def donation_is_safe(graphs: dict[str, Graph], seeds: int = 4) -> list[dict]:
    """Run every graph under its donated plan on several inputs."""
    if not graphs:
        raise ConfigError("there is nothing to check")
    if seeds < 1:
        raise ConfigError(f"the seed count must be positive, got {seeds}")
    rows = []
    for name, graph in graphs.items():
        rows.append(
            {
                "graph": name,
                "donations": len(find_donations(graph).donations),
                "matches": all(
                    donated_matches_reference(graph, seed=seed) for seed in range(seeds)
                ),
            }
        )
    return rows
