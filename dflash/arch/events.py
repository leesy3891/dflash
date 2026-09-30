"""The diagnostic record schema, and the JSONL writer behind it.

Every event carries the same identifying fields so that a phase, a layer, a
kernel and a routing decision recorded in different passes can be joined
afterwards without guessing. The batch branch is expected to write the same
schema with ``batch_size > 1`` and a populated ``request``, which is what lets
its B=1 point be compared against this branch's numbers rather than merely
placed beside them.

Identity
--------
``run_id``       one process invocation
``request_id``   one prompt within it (always ``0`` on this branch)
``step``         verify step index, or AR decode token index
``phase``        the named phase this event belongs to
``parent_event`` the enclosing event's id, so inclusive and exclusive time can
                 be separated without re-deriving the nesting
``model_role``   ``target`` | ``draft``
``layer``        decoder layer index within that role
``module``       dotted module path, as the taxonomy found it
``device``       ``cuda:N``
``stream``       CUDA stream id
``shape``        the shape that sets this event's cost
``backend``      which implementation actually ran

Timing
------
``duration_s`` is exclusive: a parent's own time, with its children's removed.
``inclusive_s`` is the wall span including children. They are stored in
separate fields and never added together, because summing a parent and its
children double-counts, and summing concurrent GPU kernels produces a number
larger than the wall time it supposedly explains.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict

SCHEMA = "arch-main-events/1"

# Why a device had no kernel running. Only assigned where there is evidence.
GAP_STRUCTURAL_SHARD_WAIT = "structural_shard_wait"
GAP_TARGET_DRAFT_DEPENDENCY = "target_draft_dependency"
GAP_HOST_LAUNCH = "host_launch_or_sync"
GAP_TRANSFER = "transfer"
GAP_PROFILER = "profiler"
GAP_UNKNOWN = "unknown"

# What limited an event. "undetermined" is a real answer and the default.
BOTTLENECK_BANDWIDTH = "bandwidth"
BOTTLENECK_COMPUTE = "compute"
BOTTLENECK_HOST_LAUNCH = "host_launch"
BOTTLENECK_TRANSFER = "transfer"
BOTTLENECK_DEPENDENCY = "dependency"
BOTTLENECK_CAPACITY = "capacity"
BOTTLENECK_MIXED = "mixed"
BOTTLENECK_UNDETERMINED = "undetermined"


@dataclass
class Event:
    """One measured thing. Fields left ``None`` were not measured, not zero."""

    event_id: str
    run_id: str
    kind: str
    name: str
    request_id: int = 0
    step: int | None = None
    phase: str | None = None
    parent_event: str | None = None
    model_role: str | None = None
    layer: int | None = None
    mixer: str | None = None
    ffn: str | None = None
    module: str | None = None
    device: str | None = None
    stream: int | None = None
    shape: dict | None = None
    backend: str | None = None
    duration_s: float | None = None
    inclusive_s: float | None = None
    bytes: dict | None = None
    gap_class: str | None = None
    bottleneck: str | None = None
    confidence: str | None = None
    evidence: dict | None = None
    na_reason: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None and v != {}}


class EventLog:
    """Append-only JSONL sink, safe to hold open across a whole sweep."""

    def __init__(self, path: str | os.PathLike, run_id: str | None = None) -> None:
        self.path = str(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._stack: list[str] = []
        self.count = 0

    def emit(self, kind: str, name: str, **fields) -> str:
        event = Event(
            event_id=uuid.uuid4().hex[:16],
            run_id=self.run_id,
            kind=kind,
            name=name,
            parent_event=fields.pop("parent_event", self._stack[-1] if self._stack else None),
            **fields,
        )
        line = json.dumps(event.as_dict(), default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self.count += 1
        return event.event_id

    def header(self, manifest: dict) -> None:
        """The run manifest, written as the first line of its own log."""
        with self._lock:
            self._handle.write(
                json.dumps(
                    {"kind": "manifest", "schema": SCHEMA,
                     "run_id": self.run_id, **manifest},
                    default=str,
                ) + "\n"
            )

    def scope(self, kind: str, name: str, **fields):
        return _Scope(self, kind, name, fields)

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        self.flush()
        self._handle.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _Scope:
    """A timed, nestable region. Records inclusive and exclusive time apart.

    Host-side timing only -- no CUDA synchronize, so it does not serialise the
    stream it is measuring. The GPU-side truth comes from the Nsight pass;
    these spans are the structure that pass is correlated against.
    """

    def __init__(self, log: EventLog, kind: str, name: str, fields: dict) -> None:
        self.log = log
        self.kind = kind
        self.name = name
        self.fields = fields
        self.event_id = uuid.uuid4().hex[:16]
        self._start = 0.0
        self._child_time = 0.0

    def __enter__(self) -> "_Scope":
        self._start = time.perf_counter()
        self.log._stack.append(self.event_id)
        return self

    def __exit__(self, *exc) -> None:
        inclusive = time.perf_counter() - self._start
        self.log._stack.pop()
        parent = self.log._stack[-1] if self.log._stack else None
        fields = dict(self.fields)
        fields.setdefault("parent_event", parent)
        self.log.emit(
            self.kind,
            self.name,
            duration_s=inclusive - self._child_time,
            inclusive_s=inclusive,
            **{k: v for k, v in fields.items() if k != "event_id"},
        )


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall time covered by at least one interval.

    Concurrent kernels on different devices overlap, so their durations sum to
    more than the elapsed time. The union is the figure that can legitimately
    be compared against a wall clock; the sum is not, and the two are reported
    as different fields.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for lo, hi in ordered[1:]:
        if lo > end:
            total += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    return total + (end - start)


def residual(wall_s: float, attributed_s: float) -> dict:
    """Wall time no event claims. Reported, never distributed."""
    return {
        "wall_s": wall_s,
        "attributed_s": attributed_s,
        "unattributed_s": max(wall_s - attributed_s, 0.0),
        "unattributed_fraction": (
            max(wall_s - attributed_s, 0.0) / wall_s if wall_s > 0 else None
        ),
    }
