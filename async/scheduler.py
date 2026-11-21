"""Tick-driven scheduler for graphics/compute queues.

Time is discrete: one scheduling batch is one tick, and there is no sleeping or
wall-clock dependency.  A command is a mapping with exactly the public fields
``{"id": str, "queue": str, "deps": list[str]}``.  Queues are FIFO lists, and
``deps`` names commands that must finish before the command may start.

The package is intentionally named ``async`` (a Python keyword); import it with
``importlib.import_module("async.scheduler")``.  Timing assertions use a
tolerance of one tick, so a boundary observed at tick N may be compared with
tick N +/- 1.
"""

from __future__ import annotations

import heapq  # noqa: F401  (stdlib surface kept for tick ordering helpers)
from collections import defaultdict, deque


class AsyncError(Exception):
    """Raised when an async scheduling contract is violated."""


def _is_int(value) -> bool:
    """True for plain ints (booleans are rejected despite being int subclasses)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require_cmd_shape(cmd: dict) -> tuple[str, str, list]:
    """Validate one command mapping and return ``(id, queue, deps)``."""
    if not isinstance(cmd, dict):
        raise AsyncError("command must be a mapping")
    cid = cmd.get("id")
    queue = cmd.get("queue")
    deps = cmd.get("deps")
    if not isinstance(cid, str) or not cid.strip():
        raise AsyncError("missing or blank command id")
    if not isinstance(queue, str) or not queue.strip():
        raise AsyncError("missing or blank queue name")
    if not isinstance(deps, list):
        raise AsyncError("deps must be a list")
    return cid, queue, deps


def _validate_graph(cmds: list[dict]) -> None:
    """Validate shape, unique ids and known dependencies of a command graph."""
    if not isinstance(cmds, list):
        raise AsyncError("cmds must be a list")
    ids: set[str] = set()
    for cmd in cmds:
        cid, _queue, _deps = _require_cmd_shape(cmd)
        if cid in ids:
            raise AsyncError(f"duplicate-id:{cid}")
        ids.add(cid)
    for cmd in cmds:
        for dep in cmd["deps"]:
            if dep not in ids:
                raise AsyncError(f"unknown-dependency:{cmd['id']}:{dep}")


def submit(queue: list[dict], cmd: dict) -> dict:
    """Append one command to the tail of a FIFO queue.

    ``queue`` is the mutable queue list and ``cmd`` is a command mapping.  The
    command is appended without reordering earlier entries; this is the only
    place where submission order is established.

    Contract:
      * A duplicate ``id`` already present in ``queue`` raises ``AsyncError``
        and leaves the queue unchanged.
      * A missing or blank ``id``/``queue``, or a non-list ``deps``, raises
        ``AsyncError`` without mutating the queue.
      * Returns the stored command mapping.
      * Queue bookkeeping happens immediately; command completion happens on a
        later explicit tick, never by sleeping.
    """
    if not isinstance(queue, list):
        raise AsyncError("queue must be a list")
    cid, _queue, _deps = _require_cmd_shape(cmd)
    for existing in queue:
        if isinstance(existing, dict) and existing.get("id") == cid:
            raise AsyncError(f"duplicate-id:{cid}")
    queue.append(cmd)
    return cmd


def insert_fence(queues: list[str], semaphore: int) -> dict:
    """Insert a cross-queue ordering fence and return its descriptor.

    ``queues`` must contain exactly two distinct queue names in the order
    ``[before, after]``.  The fence is intentionally cross-queue: no command
    committed to ``after`` may start until every command already committed to
    ``before`` has completed.  A completion observed at a tick boundary may be
    acted on at the next tick (one-tick tolerance).

    ``semaphore`` is the initial number of permits on the fence (``>= 0``).
    Each completed command in ``before`` releases one permit; a command entering
    ``after`` acquires one.  With zero permits the fence starts closed.

    Returns ``{"before": before, "after": after, "semaphore": semaphore,
    "cross_queue": True, "ordering": "before_then_after"}``.  Wrong queue count,
    duplicate queue names, or a negative permit count raises ``AsyncError``.
    """
    if not isinstance(queues, list) or len(queues) != 2:
        raise AsyncError("fence requires exactly two queue names [before, after]")
    before, after = queues
    for name in (before, after):
        if not isinstance(name, str) or not name.strip():
            raise AsyncError("fence queue names must be non-blank strings")
    if before == after:
        raise AsyncError("fence queues must be distinct")
    if not _is_int(semaphore) or semaphore < 0:
        raise AsyncError("semaphore permit count must be a non-negative int")
    return {
        "before": before,
        "after": after,
        "semaphore": semaphore,
        "cross_queue": True,
        "ordering": "before_then_after",
    }


def _schedule_levels(
    cmds: list[dict], concurrency: int, satisfied: set[str] | None = None
) -> list[list[str]]:
    """Core tick batching for an already validated command set.

    ``satisfied`` names deps already completed by earlier scheduling windows;
    they count as done before the first tick.
    """
    remaining = list(cmds)
    done: set[str] = set(satisfied or ())
    batches: list[list[str]] = []
    while remaining:
        # Re-scan every tick: dependency completion, not a fixed slice size,
        # decides readiness.  The list comprehension keeps submission order.
        ready = [c for c in remaining if all(d in done for d in c["deps"])]
        if not ready:
            # Unreachable after the cycle check above, but never spin.
            raise AsyncError("dependency cycle detected")
        tick = ready[:concurrency]
        batches.append([c["id"] for c in tick])
        done.update(c["id"] for c in tick)
        tick_ids = {c["id"] for c in tick}
        remaining = [c for c in remaining if c["id"] not in tick_ids]
    return batches


def schedule(cmds: list[dict], concurrency: int) -> list[list[str]]:
    """Batch commands by dependency level, respecting a concurrency cap.

    Every inner list is one dispatch tick and may run in parallel.  A command is
    eligible for the next tick only when all of its dependencies appeared in an
    earlier batch.  Batch width is at most ``concurrency``; if more commands are
    ready, they are split over successive ticks.  Submission order is preserved
    within a batch and when choosing among ready commands.

    The concurrency cap is checked first: ``concurrency <= 0`` raises
    ``AsyncError`` even when ``cmds`` is empty.  An otherwise empty command list
    returns ``[]``.  Duplicate ids, unknown dependencies, and dependency cycles
    raise ``AsyncError`` rather than hanging.  All tick boundaries follow the
    one-tick tolerance.
    """
    if not _is_int(concurrency) or concurrency <= 0:
        raise AsyncError("concurrency must be a positive int")
    if cmds == []:
        return []
    _validate_graph(cmds)
    if detect_deadlock(cmds):
        raise AsyncError("dependency cycle detected")
    return _schedule_levels(cmds, concurrency)


def detect_deadlock(cmds: list[dict]) -> list[str]:
    """Return the ids participating in the first dependency cycle.

    Dependencies are directed edges ``id -> dep``.  Acyclic input returns
    ``[]``.  A cycle returns each id on that cycle exactly once, deterministically
    starting from the earliest-submitted id and following ``deps`` order.  The
    function must terminate; it never waits for a tick to discover the cycle.

    Duplicate ids, unknown dependencies, or malformed commands raise
    ``AsyncError``.
    """
    _validate_graph(cmds)
    deps_of = {c["id"]: list(c["deps"]) for c in cmds}

    # Kahn peeling: whatever remains after zero-in-degree peeling is part of
    # some cycle (terminates in O(V+E), never waits on a tick).
    indeg = {cid: 0 for cid in deps_of}
    dependents: dict[str, list[str]] = defaultdict(list)
    for cmd in cmds:
        for dep in cmd["deps"]:
            indeg[cmd["id"]] += 1
            dependents[dep].append(cmd["id"])
    peel = deque(cid for cid, degree in indeg.items() if degree == 0)
    peeled: set[str] = set()
    while peel:
        node = peel.popleft()
        peeled.add(node)
        for dependent in dependents[node]:
            indeg[dependent] -= 1
            if indeg[dependent] == 0:
                peel.append(dependent)

    cyclic_in_order = [c["id"] for c in cmds if c["id"] not in peeled]
    if not cyclic_in_order:
        return []

    start = cyclic_in_order[0]  # earliest-submitted id on any cycle
    cyclic_set = set(cyclic_in_order)
    cycle = [start]
    node = start
    while True:
        nxt = next(dep for dep in deps_of[node] if dep in cyclic_set)
        if nxt == start:
            break
        cycle.append(nxt)
        node = nxt
    return cycle


def wait_semaphore(signals: list[int], value: int) -> int:
    """Fold a tick-ordered signal sequence into a semaphore count.

    ``value`` is the initial non-negative permit count.  Each positive signal
    releases that many permits; each negative signal waits for that many
    permits.  A wait is processed at one explicit tick and may not drive the
    count below zero: an underflow raises ``AsyncError`` immediately.  A zero
    signal is a no-op.

    Returns the final count, which is always ``>= 0``.  A negative initial
    ``value`` raises ``AsyncError``.  No sleeping or retry loop is allowed.
    """
    if not _is_int(value) or value < 0:
        raise AsyncError("initial semaphore value must be a non-negative int")
    count = value
    for signal in signals:
        if not _is_int(signal):
            raise AsyncError("semaphore signal must be an int")
        count += signal
        if count < 0:
            raise AsyncError("semaphore wait underflowed below zero")
    return count


def overlap_ratio(timeline: list[dict]) -> float:
    """Return the graphics/compute overlap ratio over a tick timeline.

    Each segment is ``{"kind": "gfx" | "compute", "start": int, "end": int}``
    and covers the half-open tick interval ``[start, end)`` with ``end > start``.
    Let ``G`` and ``C`` be the sets of distinct ticks covered by graphics and
    compute segments.  The result is ``len(G & C) / len(G | C)``, a float in
    ``[0.0, 1.0]``; an empty timeline returns ``0.0``.

    Segments are unioned, so repeated or adjacent segments do not double-count.
    Malformed segments or unknown kinds raise ``AsyncError``.  Callers compare
    the result with an absolute tolerance of one tick when using non-integral
    intervals.
    """
    if not isinstance(timeline, list):
        raise AsyncError("timeline must be a list")
    gfx_ticks: set[int] = set()
    compute_ticks: set[int] = set()
    for segment in timeline:
        if not isinstance(segment, dict):
            raise AsyncError("timeline segment must be a mapping")
        kind = segment.get("kind")
        start = segment.get("start")
        end = segment.get("end")
        if kind not in ("gfx", "compute"):
            raise AsyncError(f"unknown segment kind: {kind!r}")
        if not (_is_int(start) and _is_int(end)):
            raise AsyncError("segment start and end must be ints")
        if end <= start:
            raise AsyncError("segment must satisfy end > start")
        covered = set(range(start, end))
        if kind == "gfx":
            gfx_ticks |= covered
        else:
            compute_ticks |= covered
    union = gfx_ticks | compute_ticks
    if not union:
        return 0.0
    return len(gfx_ticks & compute_ticks) / len(union)


def validate(cmds: list[dict], max_inflight: int) -> list[str]:
    """Return deterministic issue strings; an empty list means valid input.

    ``max_inflight`` is the maximum number of commands allowed to be ready in
    one dependency level.  The issue strings are exactly:

      * ``duplicate-id:<id>`` for an id that occurs more than once;
      * ``unknown-dependency:<id>:<dep>`` when ``dep`` is not a command id;
      * ``over-concurrency:<ready>/<max_inflight>`` when a dependency level has
        more ready commands than the cap;
      * ``max-inflight:<max_inflight>`` when the cap is not positive.

    Issues are emitted in command order and, for each command, in dependency
    order.  Validation is read-only; it inspects one readiness level per tick
    and never sleeps.
    """
    issues: list[str] = []
    seen: set[str] = set()
    levels_input: list[tuple[str, list[str]]] = []
    for cmd in cmds:
        cid, _queue, deps = _require_cmd_shape(cmd)
        if cid in seen:
            issues.append(f"duplicate-id:{cid}")
        else:
            seen.add(cid)
        levels_input.append((cid, deps))

    # Unknown dependencies: command order, then each command's dep order.
    for cid, deps in levels_input:
        for dep in deps:
            if dep not in seen:
                issues.append(f"unknown-dependency:{cid}:{dep}")

    cap_ok = _is_int(max_inflight) and max_inflight > 0
    if not cap_ok:
        issues.append(f"max-inflight:{max_inflight}")
        return issues

    # Level widths only make sense on a well-formed, acyclic graph.
    structural_bad = any(
        issue.startswith(("duplicate-id:", "unknown-dependency:")) for issue in issues
    )
    if structural_bad:
        return issues
    if detect_deadlock(cmds):
        return issues

    level: dict[str, int] = {}
    for cid, deps in levels_input:
        level[cid] = 0 if not deps else 1 + max(level[dep] for dep in deps)
    width: dict[int, int] = defaultdict(int)
    for cid, _deps in levels_input:
        width[level[cid]] += 1
    for lv in sorted(width):
        if width[lv] > max_inflight:
            issues.append(f"over-concurrency:{width[lv]}/{max_inflight}")
    return issues


def schedule_batch(batches: list[list[dict]], config: dict) -> dict:
    """Atomically commit ordered command batches into a queue registry.

    ``batches`` is a list of dependency-ready groups, executed one tick apart;
    each group is a list of command mappings.  ``config`` is
    ``{"queues": {name: list[command]}, "concurrency": positive int}``.  Every
    batch is an independent scheduling window: commands from an earlier batch
    must never be resubmitted or leak into a later one.  Ids must therefore be
    unique across *all* batches, and every dependency must exist in the same
    batch or an earlier batch; forward dependencies raise ``AsyncError``.
    Within a batch, concurrency is capped as in :func:`schedule`.

    Validation and scheduling are completed for every batch before any queue is
    mutated.  If anything is invalid, raise ``AsyncError`` and leave
    ``config["queues"]`` (including pre-existing entries) byte-for-byte
    unchanged: the whole call is all-or-nothing.

    On success, commands are appended to their named queues in batch order and
    submit order, and the result is::

        {"committed": True,
         "order": [ids in dispatch order],
         "batches": [[ids per dispatch tick]],
         "ticks": number of dispatch ticks,
         "queues": config["queues"]}

    A missing queue name, ``concurrency <= 0``, duplicate id, unknown
    dependency, or cycle raises ``AsyncError``.  Tick hand-offs use the shared
    one-tick tolerance.
    """
    if not isinstance(config, dict):
        raise AsyncError("config must be a mapping")
    if "queues" not in config or "concurrency" not in config:
        raise AsyncError("config requires 'queues' and 'concurrency'")
    queues = config["queues"]
    concurrency = config["concurrency"]
    if not isinstance(queues, dict):
        raise AsyncError("config['queues'] must be a registry mapping")
    if not _is_int(concurrency) or concurrency <= 0:
        raise AsyncError("concurrency must be a positive int")
    if not isinstance(batches, list):
        raise AsyncError("batches must be a list")

    # Phase 1: validate everything. No queue is touched in this phase, so any
    # failure leaves the registry byte-for-byte untouched.
    known_ids: set[str] = set()
    prior_ids: set[str] = set()
    for batch in batches:
        if not isinstance(batch, list):
            raise AsyncError("each batch must be a list of commands")
        batch_ids: set[str] = set()
        for cmd in batch:
            cid, queue_name, _deps = _require_cmd_shape(cmd)
            if cid in known_ids:
                raise AsyncError(f"duplicate-id:{cid}")
            if queue_name not in queues:
                raise AsyncError(f"unknown-queue:{queue_name}")
            known_ids.add(cid)
            batch_ids.add(cid)
        for cmd in batch:
            for dep in cmd["deps"]:
                # A dep that is neither in this batch nor an earlier one is
                # either unknown or a forward reference: both are rejected.
                if dep not in prior_ids and dep not in batch_ids:
                    raise AsyncError(
                        f"unknown-dependency:{cmd['id']}:{dep}"
                    )
        # Cycle check over intra-batch edges only: deps satisfied by earlier
        # batches are already completed and cannot participate in a cycle.
        indeg = {cid: 0 for cid in batch_ids}
        dependents: dict[str, list[str]] = defaultdict(list)
        for cmd in batch:
            for dep in cmd["deps"]:
                if dep in batch_ids:
                    indeg[cmd["id"]] += 1
                    dependents[dep].append(cmd["id"])
        peel = deque(cid for cid, degree in indeg.items() if degree == 0)
        reachable = 0
        while peel:
            node = peel.popleft()
            reachable += 1
            for dependent in dependents[node]:
                indeg[dependent] -= 1
                if indeg[dependent] == 0:
                    peel.append(dependent)
        if reachable != len(batch_ids):
            raise AsyncError("dependency cycle detected within batch")
        prior_ids |= batch_ids

    # Phase 2: plan every dispatch tick. Validation happened in phase 1 (and
    # deliberately tolerates deps satisfied by earlier batches), so use the
    # core batcher directly instead of re-running the closed-graph checks.
    dispatch_batches: list[list[str]] = []
    committed_ids: set[str] = set()
    for batch in batches:
        dispatch_batches.extend(
            _schedule_levels(batch, concurrency, satisfied=committed_ids)
        )
        committed_ids.update(cmd["id"] for cmd in batch)

    # Phase 3: commit. Append in batch order and submit order so batches can
    # never interleave or leak into one another's queue windows.
    order = [cid for tick in dispatch_batches for cid in tick]
    for batch in batches:
        for cmd in batch:
            queues[cmd["queue"]].append(cmd)

    return {
        "committed": True,
        "order": order,
        "batches": dispatch_batches,
        "ticks": len(dispatch_batches),
        "queues": queues,
    }
