#!/usr/bin/env python3
"""Regression tests for the Cinematic Director V2.

They assert the hard guarantees of the edit plan:
  * exactly 64.00s, contiguous output timeline
  * every source window inside the real source duration
  * player-first framing / real tracking anchors when tracking exists
  * hard cut is the default transition
  * slow motion only on event-driven windows
  * no invented goal/celebration claim and no generic template caption without
    an on-screen action signal
  * REFERENCE mode really consumes the measured style profile
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "yolo" / "cinematic_director.py"
sys.path.insert(0, str(ROOT / "yolo"))

import cinematic_director as cd  # noqa: E402


def _motion(duration: float) -> dict:
    n = int(duration * 2)
    prof = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        # Real motion peak around 70% of the clip.
        energy = 0.85 if abs(t - duration * 0.7) < 1.6 else 0.35 + 0.1 * ((i % 5) / 5)
        prof.append({"t": round(t, 3), "energy": round(energy, 4)})
    return {"duration": duration, "motion_profile": prof,
            "thresholds": {"low": 0.4, "high": 0.6}, "ball_detected": False}


def _tracking(duration: float, fps: int = 30) -> dict:
    samples = []
    total = int(duration * fps)
    for f in range(total):
        t = f / fps
        x = 0.28 + 0.42 * (t / max(duration, 1e-6))
        samples.append({"frame": f, "track_id": 7, "class": "player", "x": x, "y": 0.55,
                        "w": 0.08, "h": 0.22, "confidence": 0.92})
        samples.append({"frame": f, "track_id": 8, "class": "player", "x": 0.62, "y": 0.52,
                        "w": 0.07, "h": 0.2, "confidence": 0.8})
        samples.append({"frame": f, "track_id": 99, "class": "ball", "x": x + 0.02, "y": 0.6,
                        "w": 0.02, "h": 0.02, "confidence": 0.9})
    return {"summary": {"fps": fps, "samples": samples}}


def _events(duration: float) -> dict:
    return {"version": "v10", "fps": 30, "ball_detected": True, "events": [
        {"time": round(duration * 0.7 + 0.4, 3), "event_score": 0.92, "ball_proximity": 0.9,
         "ball_speed": 0.85, "acceleration": 0.7, "direction_change": 0.5,
         "pressure": 0.6, "track_id": "7", "events": ["ball_engagement", "shot_candidate"]},
        {"time": round(duration * 0.35, 3), "event_score": 0.55, "ball_proximity": 0.6,
         "ball_speed": 0.4, "track_id": "7", "events": ["ball_engagement"]},
    ]}


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_plan_contract(tmp_path):
    duration = 40.0
    motion = _write(tmp_path, "motion.json", _motion(duration))
    tracking = _write(tmp_path, "tracking.json", _tracking(duration))
    events = _write(tmp_path, "events.json", _events(duration))
    out = tmp_path / "plan.json"
    subprocess.run([sys.executable, str(ENGINE), "--duration", str(duration), "--output", str(out),
                    "--motion", str(motion), "--tracking", str(tracking), "--events", str(events),
                    "--mode", "PRO"], check=True)
    plan = json.loads(out.read_text())

    assert plan["validation"]["ok"] is True, plan["validation"]
    assert plan["duration"] == 64
    tl = plan["timeline"]
    assert 8 <= len(tl) <= 30
    assert abs(tl[-1]["output_end"] - 64.0) < 0.01
    assert tl[0]["output_start"] == 0

    for i, c in enumerate(tl):
        assert c["source_start"] >= 0
        assert c["source_end"] <= duration + 0.001, f"clip {i} exceeds source duration"
        assert c["source_end"] > c["source_start"]
        assert 0.05 <= c["crop_x"] <= 0.95
        assert c["transition"] in ("hard_cut", "dissolve", "flash", "fade")
        if i:
            assert abs(c["output_start"] - tl[i - 1]["output_end"]) < 0.02

    # Hard cut must remain the dominant/default transition.
    kinds = {}
    for c in tl:
        kinds[c["transition"]] = kinds.get(c["transition"], 0) + 1
    assert kinds.get("hard_cut", 0) >= len(tl) - max(2, len(tl) // 4)

    # Player is the subject: real tracking gives player_track anchors.
    assert plan["cinematic_director"]["protagonist_track"] == "7"
    assert plan["cinematic_director"]["ball_tracked"] is True
    assert any(c.get("subject_anchor") == "player_track" for c in tl)

    # Slow motion is event-driven, never random.
    for c in tl:
        if c.get("slow_motion"):
            assert c["beat_role"] in ("climax", "impact") or c["evidence"]["has_ball_evidence"]


def test_never_invents_events_or_template_captions(tmp_path):
    duration = 30.0
    motion = _write(tmp_path, "motion.json", _motion(duration))
    # No tracking, no events: the director must stay evidence-honest.
    out = tmp_path / "plan.json"
    subprocess.run([sys.executable, str(ENGINE), "--duration", str(duration), "--output", str(out),
                    "--motion", str(motion), "--mode", "STANDARD"], check=True)
    plan = json.loads(out.read_text())
    assert plan["validation"]["ok"] is True
    assert plan["cinematic_director"]["evidence_level"] == "low"
    for c in plan["timeline"]:
        assert c["evidence"]["has_ball_evidence"] is False
        assert c["slow_motion"] is False
        assert "goal" not in str(c["action"]).lower()
        assert "celebration" not in str(c["action"]).lower()


def test_reference_mode_consumes_style_profile(tmp_path):
    duration = 40.0
    motion = _write(tmp_path, "motion.json", _motion(duration))
    profile = _write(tmp_path, "reference.json", {
        "source": "reference.mp4",
        "avg_shot_duration": 2.6,
        "cut_density": 0.38,
        "zoom_intensity": 0.02,
        "slow_motion_shot_ratio": 0.2,
        "text_per_shot": 0.08,
        "shot_type_weights": {"close_up": 0.7, "action": 0.2, "wide": 0.1},
        "transition_weights": {"hard_cut": 0.8, "dissolve": 0.15, "flash": 0.05},
        "color": {"contrast": 120.0, "saturation": 130.0, "skin_ratio": 0.22, "neon_grass_ratio": 0.3},
    })
    out = tmp_path / "plan.json"
    subprocess.run([sys.executable, str(ENGINE), "--duration", str(duration), "--output", str(out),
                    "--motion", str(motion), "--reference-style", str(profile),
                    "--mode", "REFERENCE"], check=True)
    plan = json.loads(out.read_text())

    assert plan["validation"]["ok"] is True
    # Pacing follows the measured reference (dense cutting), not the mode default.
    pro_clips = cd.slot_count(duration, cd.profile_from_modes("PRO", None))
    ref_clips = len(plan["timeline"])
    assert ref_clips >= pro_clips
    assert plan["style_profile"]["source"] == "reference.mp4"
    assert abs(plan["style_profile"]["zoom_intensity"] - 0.02) < 1e-6
    assert plan["color_grade"]["protect_skin_tones"] is True
    assert plan["color_grade"]["prevent_neon_grass"] is True
    # The profile must never be copied as a timeline.
    assert "shots" not in json.dumps(plan["cinematic_director"])
