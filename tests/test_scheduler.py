"""Red test suite for the tick-based async scheduler.

The package directory is named ``async``, which is a Python keyword, so the
module is loaded through :mod:`importlib` rather than an ``import async``
statement.  Every test exercises a public stub and fails until the contract in
``async/scheduler.py`` is implemented.
"""

import importlib

import pytest

scheduler = importlib.import_module("async.scheduler")
AsyncError = scheduler.AsyncError


def cmd(cid, queue="gpu", deps=()):
    """Build a command document without touching scheduler state."""
    return {"id": cid, "queue": queue, "deps": list(deps)}


# ---------------------------------------------------------------------------
# One test per seeded defect.
# ---------------------------------------------------------------------------


def test_defect_01_dependencies_override_submission_order():
    """A later-submitted prerequisite still runs before its dependent."""
    cmds = [cmd("b", deps=["a"]), cmd("a")]
    assert scheduler.schedule(cmds, 2) == [["a"], ["b"]]


def test_defect_02_fence_waits_across_different_queues():
    """A fence between two queue names must be cross-queue, not local-only."""
    fence = scheduler.insert_fence(["graphics", "compute"], 1)
    assert fence == {
        "before": "graphics",
        "after": "compute",
        "semaphore": 1,
        "cross_queue": True,
        "ordering": "before_then_after",
    }


def test_defect_03_batches_follow_dependencies_not_fixed_size():
    """Dependency levels, not a fixed batch size, decide the batches."""
    cmds = [
        cmd("a"),
        cmd("b", deps=["a"]),
        cmd("c", deps=["a"]),
        cmd("d", deps=["b", "c"]),
    ]
    assert scheduler.schedule(cmds, 2) == [["a"], ["b", "c"], ["d"]]


def test_defect_04_cycle_is_reported_instead_of_deadlocking():
    """A dependency cycle is returned; it must not wait forever."""
    cmds = [
        cmd("a", deps=["c"]),
        cmd("b", deps=["a"]),
        cmd("c", deps=["b"]),
    ]
    cycle = scheduler.detect_deadlock(cmds)
    assert set(cycle) == {"a", "b", "c"}
    assert len(cycle) == 3


def test_defect_05_semaphore_wait_never_goes_negative():
    """Waiting twice with one permit underflows and raises AsyncError."""
    with pytest.raises(AsyncError):
        scheduler.wait_semaphore([-1, -1], 1)


def test_defect_06_overlap_uses_tick_time_not_command_count():
    """Two segments can still overlap for only five of fifteen ticks."""
    timeline = [
        {"kind": "gfx", "start": 0, "end": 10},
        {"kind": "compute", "start": 5, "end": 15},
    ]
    # Overlap [5, 10) is five ticks; the union [0, 15) is fifteen ticks.
    assert scheduler.overlap_ratio(timeline) == pytest.approx(1 / 3, abs=1 / 15)


def test_defect_07_unknown_dependency_is_reported():
    """An unknown prerequisite is an issue, never silently skipped."""
    issues = scheduler.validate([cmd("a", deps=["ghost"])], 4)
    assert "unknown-dependency:a:ghost" in issues


def test_defect_08_inflight_cap_is_validated():
    """Three simultaneously-ready commands exceed a cap of two."""
    issues = scheduler.validate([cmd("a"), cmd("b"), cmd("c")], 2)
    assert "over-concurrency:3/2" in issues


def test_defect_09_batches_do_not_leak_and_roll_back():
    """A cross-batch duplicate aborts the call without queue contamination."""
    queues = {"gpu": []}
    config = {"queues": queues, "concurrency": 2}
    batches = [
        [cmd("a"), cmd("b", deps=["a"])],
        [cmd("a")],
    ]
    with pytest.raises(AsyncError):
        scheduler.schedule_batch(batches, config)
    assert queues == {"gpu": []}


def test_defect_10_zero_concurrency_does_not_spin():
    """concurrency=0 is rejected instead of looping with no progress."""
    with pytest.raises(AsyncError):
        scheduler.schedule([cmd("a")], 0)


# ---------------------------------------------------------------------------
# Combination and boundary tests.
# ---------------------------------------------------------------------------


def test_combination_no_dependencies_finish_in_one_batch():
    """Independent commands fit in a single batch when concurrency allows."""
    cmds = [cmd("a"), cmd("b"), cmd("c")]
    assert scheduler.schedule(cmds, 3) == [["a", "b", "c"]]


def test_combination_concurrency_one_is_serial():
    """A concurrency cap of one produces one command per tick."""
    cmds = [cmd("a"), cmd("b"), cmd("c")]
    assert scheduler.schedule(cmds, 1) == [["a"], ["b"], ["c"]]


def test_combination_empty_commands_returns_empty():
    """There is no work, so there are no batches and no tick."""
    assert scheduler.schedule([], 2) == []


def test_combination_batch_zero_concurrency_raises_and_rolls_back():
    """A zero cap is rejected at both scheduling entry points."""
    with pytest.raises(AsyncError):
        scheduler.schedule([], 0)

    queues = {"gpu": []}
    with pytest.raises(AsyncError):
        scheduler.schedule_batch(
            [[cmd("a")]],
            {"queues": queues, "concurrency": 0},
        )
    assert queues == {"gpu": []}


def test_combination_batch_success_preserves_order():
    """Successful batch commits append in batch/submit order."""
    queues = {"gpu": []}
    scheduler.submit(queues["gpu"], cmd("seed"))

    result = scheduler.schedule_batch(
        [[cmd("a")], [cmd("b")]],
        {"queues": queues, "concurrency": 2},
    )

    assert result["committed"] is True
    assert result["order"] == ["a", "b"]
    assert result["batches"] == [["a"], ["b"]]
    assert result["ticks"] == 2
    assert [item["id"] for item in queues["gpu"]] == ["seed", "a", "b"]
