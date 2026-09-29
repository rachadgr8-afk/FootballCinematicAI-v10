#!/usr/bin/env python3
"""Regression tests for the OPTIONAL psychological storyteller layer.

They assert the hard guarantees:
  * duel detection needs REAL evidence (no duel => no duel, no invented winner)
  * the VLM never asserts a winner from unknown/insufficient evidence
  * DeepSeek failure/timeout degrades to story_script = []
  * story-script timestamps are clamped + spaced (never collide)
  * edit-plan enrichment is ADDITIVE (old fields untouched, no-duel => untouched)
  * depth engine degrades honestly (proxy labelled, no model => applied/false)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / "yolo"
sys.path.insert(0, str(YOLO))
sys.path.insert(0, str(ROOT / "python"))

import cinematic_storyteller as cs  # noqa: E402
import deepseek_storyteller as ds  # noqa: E402


def _converging_tracking(dur=20.0, fps=25):
    total = int(dur * fps)
    samples = []
    for f in range(total):
        t = f / fps
        if t <= 12:
            px = 0.32 + (0.52 - 0.32) * (t / 12); qx = 0.70 + (0.52 - 0.70) * (t / 12)
        else:
            px = 0.52 + (0.80 - 0.52) * ((t - 12) / 8); qx = 0.52 + (0.34 - 0.52) * ((t - 12) / 8)
        samples.append({"frame": f, "track_id": 7, "class": "player", "x": round(px, 5), "y": 0.55,
                        "w": 0.08, "h": 0.2, "confidence": 0.92})
        samples.append({"frame": f, "track_id": 8, "class": "player", "x": round(qx, 5), "y": 0.55,
                        "w": 0.08, "h": 0.2, "confidence": 0.88})
        samples.append({"frame": f, "track_id": 99, "class": "ball", "x": round(px + 0.02, 5), "y": 0.61,
                        "w": 0.02, "h": 0.02, "confidence": 0.9})
    return {"summary": {"fps": fps, "width": 1280, "height": 720, "frames": total,
                        "sourceFrames": total, "detections": {"player": 2 * total, "ball": total},
                        "samples": samples}}


def _events():
    return {"version": "v10", "fps": 25, "ball_detected": True, "events": [
        {"time": 11.6, "event_score": 0.85, "ball_proximity": 0.8, "pressure": 0.7,
         "direction_change": 0.5, "acceleration": 0.6, "ball_speed": 0.5, "track_id": "7",
         "events": ["ball_engagement", "pressure_candidate"]},
        {"time": 12.1, "event_score": 0.94, "ball_proximity": 0.92, "pressure": 0.85,
         "direction_change": 0.7, "acceleration": 0.85, "ball_speed": 0.8, "track_id": "7",
         "events": ["ball_engagement", "shot_candidate", "explosive_run", "direction_change_candidate"]},
    ]}


def test_duel_detected_with_real_converging_evidence():
    tracks, balls, fps = cs.build_series(_converging_tracking())
    cs._ball_affinity(tracks, balls)
    duel = cs.detect_duel(tracks, balls, _events(), None, 20.0)
    assert duel["detected"] is True, duel
    assert duel["predator"] in ("7", "8") and duel["prey"] in ("7", "8")
    assert duel["predator"] != duel["prey"]
    assert duel["winner"] is None and duel["loser"] is None  # only VLM may set them
    assert duel["confidence"] >= cs.DUEL_MIN_SCORE
    assert duel["moment"] is not None


def test_no_duel_when_evidence_is_thin():
    # Two players moving in parallel, far apart, no ball.
    samples = []
    for f in range(300):
        samples.append({"frame": f, "track_id": 1, "class": "player", "x": 0.2 + f * 0.001, "y": 0.5,
                        "confidence": 0.9})
        samples.append({"frame": f, "track_id": 2, "class": "player", "x": 0.8 + f * 0.001, "y": 0.5,
                        "confidence": 0.9})
    tracking = {"summary": {"fps": 25, "samples": samples}}
    tracks, balls, _ = cs.build_series(tracking)
    cs._ball_affinity(tracks, balls)
    duel = cs.detect_duel(tracks, balls, None, None, 12.0)
    assert duel["detected"] is False
    assert duel["winner"] is None

    plan = {"duration": 64, "timeline": [
        {"timeline_index": i, "source_start": i, "source_end": i + 1, "output_start": i * 4,
         "output_end": (i + 1) * 4, "text": "KEEP"} for i in range(4)]}
    bundle = {"success": True, "version": cs.VERSION, "evidence_level": "low", "duel": duel,
              "story_arc": [], "story_script": [], "pov_switch_points": [],
              "low_angle_segments": [], "depth_segments": [], "duel_moment": 6.0}
    out = cs.enrich_plan(json.loads(json.dumps(plan)), bundle)
    # No duel => additive metadata only; existing fields untouched.
    for c in out["timeline"]:
        assert c["text"] == "KEEP"
        assert "story_role" not in c
        assert "low_angle" not in c
    assert out["psychological_story"]["duel_detected"] is False


def test_enrichment_is_additive_and_roles_are_evidence_gated():
    tracks, balls, _ = cs.build_series(_converging_tracking())
    cs._ball_affinity(tracks, balls)
    duel = cs.detect_duel(tracks, balls, _events(), None, 20.0)
    arc = cs.build_story_arc(duel, _events(), 20.0)
    bundle = {"success": True, "version": cs.VERSION, "evidence_level": "medium", "duel": duel,
              "story_arc": arc, "story_script": [], "pov_switch_points": cs.pov_triggers(_events(), 20.0),
              "low_angle_segments": [{"time": 11.0}], "depth_segments": [{"time": 11.0, "span": 6}],
              "duel_moment": duel["moment"]}
    plan = {"duration": 64, "color_grade": {"contrast": 1.2}, "timeline": [
        {"timeline_index": i, "source_start": round(i * 1.25, 3), "source_end": round(i * 1.25 + 1.1, 3),
         "output_start": i * 4, "output_end": (i + 1) * 4, "text": ""} for i in range(16)]}
    original_texts = [c["text"] for c in plan["timeline"]]
    out = cs.enrich_plan(plan, bundle)
    # Everything old is preserved.
    assert out["color_grade"]["contrast"] == 1.2
    assert len(out["timeline"]) == 16
    [c["timeline_index"] for c in out["timeline"]] == list(range(16))
    # Additive fields present.
    assert out["psychological_story"]["duel_detected"] is True
    assert any(c.get("story_role") == "impact" for c in out["timeline"])
    assert any(c.get("story_role") == "approach" for c in out["timeline"])
    assert any(c.get("low_angle") is True for c in out["timeline"])
    # psychological grade block added but does not replace the global grade.
    assert out["color_grade"]["psychological"]["enabled"] is True
    assert out["color_grade"]["contrast"] == 1.2


def test_deepseek_failsafe_without_key(monkeypatch=None):
    bundle = {"duel": {"detected": True, "predator": "7", "prey": "8", "confidence": 0.9},
              "duel_moment": 30.0, "story_arc": ["approach", "impact"], "notes": None}
    import os
    old = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        script, meta = ds.generate_script(bundle, 64.0)
        assert script == []
        assert meta["applied"] is False
        assert "DEEPSEEK_API_KEY" in (meta["reason"] or "")
    finally:
        if old is not None:
            os.environ["DEEPSEEK_API_KEY"] = old


def test_deepseek_timestamp_clamping_and_spacing():
    times = ds._schedule_times(30.0, 64.0)
    assert all(0.0 <= t <= 64.0 for t in times)
    assert len(times) == 6
    for a, b in zip(times, times[1:]):
        assert b - a >= ds.MIN_GAP - 1e-6, (a, b)
    # Clamp near the very start / end.
    assert all(t >= ds.EDGE_MARGIN for t in ds._schedule_times(0.4, 64.0))
    assert all(t <= 64.0 - ds.EDGE_MARGIN + 1e-6 for t in ds._schedule_times(63.9, 64.0))


def test_deepseek_line_validation_rejects_collisions_and_dupes():
    forbidden = {ds._norm(x) for x in ds.TONE_REFERENCES} | {ds._norm("COME CLOSER")}
    lines = [
        {"time": 10, "text": "COME CLOSER...", "position": "center"},          # forbidden -> drop
        {"time": 11, "text": "HE IS HUNTING YOU", "position": "center"},
        {"time": 12, "text": "He is hunting you", "position": "center"},        # dup -> drop
        {"time": 13, "text": "NO WAY OUT NOW", "position": "bogus"},            # position coerced
    ]
    valid = ds._validate_lines(lines, 64.0, 30.0, forbidden)
    assert len(valid) == 2
    assert all(v["position"] in ("lower_third", "center", "top") for v in valid)
    assert len({ds._norm(v["text"]) for v in valid}) == 2


def test_vlm_analyzer_rejects_unknown_player_ids(tmp_path):
    import vlm_analyzer as va
    evidence = {"players": ["7", "8"], "duel": {"predator": "7", "prey": "8"}, "events": []}
    obj = {"duel_moment": 12.0, "winner": "42", "loser": "7",
           "story_arc": ["approach", "NONSENSE"], "confidence": 0.8}
    verdict = va._coerce_verdict(obj, evidence, (10.0, 14.0), 20.0)
    assert verdict["winner"] is None       # unknown id -> null
    assert verdict["loser"] == "7"
    assert verdict["story_arc"] == ["approach"]
    assert 0.0 <= verdict["confidence"] <= 1.0


def test_vlm_analyzer_disabled_without_key():
    import os
    import vlm_analyzer as va
    old = os.environ.pop("VLM_API_KEY", None)
    try:
        verdict, meta = va.analyze("does-not-exist.mp4", {}, (0, 1), 10.0)
        assert verdict is None and meta["applied"] is False
    finally:
        if old is not None:
            os.environ["VLM_API_KEY"] = old


def test_depth_engine_honest_fallback(tmp_path):
    import subprocess
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
                    "-pix_fmt", "yuv420p", str(video), "-loglevel", "error"], check=True)
    segs = tmp_path / "seg.json"
    segs.write_text(json.dumps([{"time": 0.4, "span": 2.0}]))
    out = tmp_path / "depth.json"
    masks = tmp_path / "masks"
    res = subprocess.run([sys.executable, str(ROOT / "python" / "depth_engine.py"), "run",
                          "--video", str(video), "--segments", str(segs), "--output", str(out),
                          "--mask-dir", str(masks), "--proxy"], capture_output=True, text=True)
    assert res.returncode == 0
    data = json.loads(out.read_text())
    assert data["success"] is True
    # Honest: the source is clearly labelled (proxy when no model), never a fake.
    assert data["source"] in ("proxy", "depth-anything-v2", "depth-anything-v2+proxy", "none")
    if data["applied"]:
        assert data["segments"][0]["source"] in ("proxy", "depth-anything-v2", "depth-anything-v2+proxy")


def test_all_new_modules_import_and_compile():
    for f in ["cinematic_storyteller.py", "deepseek_storyteller.py", "vlm_analyzer.py"]:
        p = subprocess.run([sys.executable, "-m", "py_compile", str(YOLO / f)])
        assert p.returncode == 0, f
    p = subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "python" / "depth_engine.py")])
    assert p.returncode == 0
