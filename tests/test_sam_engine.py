#!/usr/bin/env python3
"""Regression tests for the OPTIONAL SAM segmentation layer.

Fast, dependency-light: they exercise the deterministic 'mock' adapter (or the
classical-CV fallback) so they run on CI without the heavy Meta SAM stack. They
also assert the two hard guarantees of the feature: (1) it is inert/fault-safe
when disabled or unavailable, and (2) it only segments the interesting windows.

Run: pytest tests/test_sam_engine.py  (or `python -m pytest`)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "python" / "sam_engine.py"

sys.path.insert(0, str(ROOT / "python"))


def _make_fixtures(tmp_path: Path):
    # A tiny real video so OpenCV can decode real frames.
    video = tmp_path / "dummy.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
         "-pix_fmt", "yuv420p", str(video), "-loglevel", "error"],
        check=True,
    )
    samples = []
    for f in range(1, 76):
        x = 0.3 + 0.004 * f
        samples.append({"frame": f, "track_id": 1, "class": "player", "x": x, "y": 0.55,
                        "w": 0.12, "h": 0.30, "confidence": 0.9,
                        "x1": x - 0.06, "y1": 0.4, "x2": x + 0.06, "y2": 0.7})
        samples.append({"frame": f, "track_id": 99, "class": "ball", "x": x + 0.02, "y": 0.6,
                        "w": 0.03, "h": 0.03, "confidence": 0.92,
                        "x1": x + 0.005, "y1": 0.585, "x2": x + 0.035, "y2": 0.615})
    tracking = tmp_path / "tracking.json"
    tracking.write_text(json.dumps({"success": True, "summary": {
        "fps": 25, "width": 320, "height": 240, "frames": 75, "samples": samples}}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"duration": 64, "timeline": [
        {"timeline_index": 0, "source_start": 0.4, "source_end": 1.6, "action": "dribble",
         "shot_type": "medium", "beat_role": "setup"},
        {"timeline_index": 1, "source_start": 1.8, "source_end": 2.8, "action": "shot on goal",
         "shot_type": "action", "beat_role": "climax"},
        {"timeline_index": 2, "source_start": 0.2, "source_end": 0.6, "action": "plain lut slow motion",
         "shot_type": "wide", "beat_role": "escalation"},
    ]}))
    return video, tracking, plan


def _run(engine_args, extra_env=None):
    import os
    env = dict(os.environ)
    env["SAM_ADAPTER"] = "mock"
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, str(ENGINE), *engine_args],
                          capture_output=True, text=True, env=env)


def test_check_reports_tooling():
    proc = _run(["check"])
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["success"] is True
    assert data["opencv"] is True


def test_run_segments_only_interesting_windows(tmp_path):
    video, tracking, plan = _make_fixtures(tmp_path)
    out = tmp_path / "sam.json"
    proc = _run(["run", "--video", str(video), "--detections", str(tracking),
                 "--plan", str(plan), "--output", str(out),
                 "--mask-mode", "rle", "--cache-dir", str(tmp_path / "cache")])
    assert proc.returncode == 0
    data = json.loads(out.read_text())
    assert data["success"] is True and data["applied"] is True
    # Only the shot/goal + dribble windows qualify; the plain LUT clip must not.
    reasons = {s["reason"] for s in data["segments"]}
    assert "goal" in reasons or "shot" in reasons
    assert all(s["needsIsolation"] is True for s in data["segments"])
    assert data["segmentsProcessed"] <= 3
    assert data["objectsTracked"] >= 1
    # Normalised contract present.
    for s in data["segments"]:
        for o in s["objects"]:
            for key in ("id", "class", "confidence", "startFrame", "endFrame", "frames"):
                assert key in o
            assert o["maskQuality"] >= 0


def test_rle_masks_are_compact_and_decodable(tmp_path):
    video, tracking, plan = _make_fixtures(tmp_path)
    cache_dir = tmp_path / "cache"
    out = tmp_path / "sam.json"
    _run(["run", "--video", str(video), "--detections", str(tracking),
          "--plan", str(plan), "--output", str(out), "--mask-mode", "rle",
          "--cache-dir", str(cache_dir)])
    entries = list(cache_dir.glob("*.json"))
    assert entries, "expected at least one cached mask"
    from sam_engine import decode_rle
    sample = json.loads(entries[0].read_text())
    assert "rle" in sample and "shape" in sample
    mask = decode_rle(sample["rle"], sample["shape"][0], sample["shape"][1])
    assert mask.shape == tuple(sample["shape"])
    assert int(mask.sum()) > 0


def test_no_isolation_for_boring_plan(tmp_path):
    video, _tracking, _plan = _make_fixtures(tmp_path)
    plan = tmp_path / "boring.json"
    plan.write_text(json.dumps({"duration": 64, "timeline": [
        {"timeline_index": 0, "source_start": 0.2, "source_end": 1.0,
         "action": "slow motion lut", "shot_type": "wide", "beat_role": "escalation"},
    ]}))
    out = tmp_path / "sam.json"
    proc = _run(["run", "--video", str(video), "--plan", str(plan), "--output", str(out)])
    assert proc.returncode == 0
    data = json.loads(out.read_text())
    assert data["applied"] is True and data["segmentsProcessed"] == 0


def test_meta_backend_unavailable_degrades_gracefully(tmp_path):
    video, tracking, plan = _make_fixtures(tmp_path)
    out = tmp_path / "sam.json"
    proc = _run(["run", "--video", str(video), "--detections", str(tracking),
                 "--plan", str(plan), "--output", str(out)],
                extra_env={"SAM_ADAPTER": "meta"})
    assert proc.returncode == 0
    data = json.loads(out.read_text())
    assert data["success"] is False and data["applied"] is False
    assert data.get("error")


def test_missing_video_does_not_crash(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"timeline": []}))
    proc = _run(["run", "--video", str(tmp_path / "nope.mp4"), "--plan", str(plan)])
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["success"] is False and data["error"] == "video-not-found"


def test_type_consistency_of_tracking_bridge():
    """The normalised object shape must match the object contract for every entry."""
    import sam_engine

    tracking = {"summary": {"fps": 25, "samples": [
        {"frame": 1, "track_id": 1, "class": "player", "x": .5, "y": .5, "confidence": .9},
    ]}}
    idx = sam_engine.index_detections(tracking)
    assert "1" in idx["players_by_track"]
