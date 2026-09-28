#!/usr/bin/env python3
"""Regression tests for the Reference Style Analyzer.

They build a tiny real video (distinct colour blocks = real hard cuts) and assert
the profile exposes the measured style parameters the Cinematic Director needs.
Local-only: no network, no model download.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "yolo" / "reference_style_analyzer.py"

sys.path.insert(0, str(ROOT / "yolo"))


def _make_video(tmp_path: Path) -> Path:
    video = tmp_path / "reference.mp4"
    # Three visually distinct 2s blocks -> at least two real hard cuts.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=darkgreen:s=320x568:r=30:d=2",
         "-f", "lavfi", "-i", "color=c=white:s=320x568:r=30:d=2",
         "-f", "lavfi", "-i", "color=c=black:s=320x568:r=30:d=2",
         "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
         "-map", "[v]", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    return video


def test_profile_keys_and_types(tmp_path):
    video = _make_video(tmp_path)
    out = tmp_path / "profile.json"
    subprocess.run([sys.executable, str(ENGINE), "--input", str(video), "--output", str(out)], check=True)
    profile = json.loads(out.read_text())

    for key in ("shot_count", "avg_shot_duration", "cut_density", "zoom_intensity",
                "shot_type_weights", "camera_weights", "transition_weights",
                "slow_motion_shot_ratio", "text_per_shot", "subject_shot_ratio",
                "hero_structure", "ending_structure", "shots", "cuts", "color", "audio"):
        assert key in profile, f"missing profile key: {key}"

    assert profile["shot_count"] >= 2
    assert profile["avg_shot_duration"] > 0
    assert profile["cut_density"] > 0
    assert isinstance(profile["shot_type_weights"], dict)
    assert set(profile["color"]).issuperset({"contrast", "saturation", "skin_ratio", "neon_grass_ratio"})
    assert profile["duration"] > 5


def test_style_only_never_a_sequence(tmp_path):
    """The profile must carry style parameters, not a copyable timeline."""
    video = _make_video(tmp_path)
    out = tmp_path / "profile.json"
    subprocess.run([sys.executable, str(ENGINE), "--input", str(video), "--output", str(out)], check=True)
    text = out.read_text()
    # No per-frame plan and no reference frames are embedded.
    assert "source_start" not in text and "output_start" not in text
    assert len(text) < 400000


def test_missing_file_fails_safely(tmp_path):
    out = tmp_path / "profile.json"
    proc = subprocess.run(
        [sys.executable, str(ENGINE), "--input", str(tmp_path / "nope.mp4"), "--output", str(out)],
        capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout.strip().split("\n")[-1])
    assert payload["success"] is False
    assert payload.get("error")
