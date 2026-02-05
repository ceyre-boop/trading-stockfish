import json
from pathlib import Path

from engine.cognitive_timeline import append_cycle_to_timeline, build_cycle_snapshot

SNAP_INPUT = dict(
    tips={"a": 1},
    priors={"a": 1},
    belief_shift={"a": {"delta_expected_R": 0.1}},
    drift={"a": 1},
    stability_report={"E1": {"stability": 0.6}, "E2": {"stability": 0.8}},
    archetypes={"ARCH_1": {"members": ["E1", "E2"]}},
    archetype_performance={
        "ARCH_1": {"mean_expected_R": 1.2},
        "ARCH_2": {"mean_expected_R": 0.5},
    },
    archetype_evolution={
        "ARCH_1": {"delta_expected_R": 0.05},
        "ARCH_2": {"delta_expected_R": -0.1},
    },
)


def test_snapshot_construction():
    snap = build_cycle_snapshot(1, "2024-01-01T00:00:00Z", **SNAP_INPUT)
    assert abs(snap.avg_stability - 0.7) < 1e-9
    assert snap.dominant_archetype == "ARCH_1"
    assert "trajectory" in snap.commentary


def test_timeline_append(tmp_path):
    path = tmp_path / "timeline.json"
    snap1 = build_cycle_snapshot(2, "2024-01-02T00:00:00Z", **SNAP_INPUT)
    snap2 = build_cycle_snapshot(1, "2024-01-01T00:00:00Z", **SNAP_INPUT)
    append_cycle_to_timeline(path, snap1)
    timeline = append_cycle_to_timeline(path, snap2)
    assert [c.cycle_index for c in timeline.cycles] == [1, 2]


def test_timeline_persistence(tmp_path):
    path = tmp_path / "timeline.json"
    snap1 = build_cycle_snapshot(1, "2024-01-01T00:00:00Z", **SNAP_INPUT)
    append_cycle_to_timeline(path, snap1)
    snap2 = build_cycle_snapshot(2, "2024-01-02T00:00:00Z", **SNAP_INPUT)
    append_cycle_to_timeline(path, snap2)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert len(loaded.get("cycles", [])) == 2


def test_commentary_deterministic():
    snap1 = build_cycle_snapshot(1, "2024-01-01T00:00:00Z", **SNAP_INPUT)
    snap2 = build_cycle_snapshot(1, "2024-01-01T00:00:00Z", **SNAP_INPUT)
    assert snap1.commentary == snap2.commentary


def test_artifact_sorted(tmp_path):
    path = tmp_path / "timeline.json"
    snap1 = build_cycle_snapshot(2, "2024-01-02T00:00:00Z", **SNAP_INPUT)
    snap2 = build_cycle_snapshot(1, "2024-01-01T00:00:00Z", **SNAP_INPUT)
    append_cycle_to_timeline(path, snap1)
    append_cycle_to_timeline(path, snap2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = [c.get("cycle_index") for c in payload.get("cycles", [])]
    assert indices == sorted(indices)
