#!/usr/bin/env python3
"""
Tests for the standalone CinematicEngine (video_engine.py).

Covers the two required public methods plus their error contracts:
  * import of `from video_engine import CinematicEngine`
  * CinematicEngine.slow_motion_optical_flow(input, output, slow_factor, target_segment)
  * CinematicEngine.add_cinematic_lut(input, output, intensity)

The engine depends only on OpenCV + NumPy. If OpenCV is unavailable (e.g. a bare
CI box without the pinned wheel) the whole module is skipped rather than failed,
because the production venv always ships opencv-python-headless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2", reason="OpenCV is required for video_engine tests")
import numpy as np  # noqa: E402

from video_engine import CinematicEngine  # noqa: E402


def _make_clip(path: Path, seconds: float = 4.0, fps: int = 30, size=(320, 180)) -> None:
    """Synthesise a small MP4 with cv2 only (no ffmpeg dependency in the test)."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "test could not create a synthetic input clip"
    total = int(seconds * fps)
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # A moving block so optical flow has real motion to estimate.
        cx = int((i / total) * (w - 40)) + 20
        cv2.rectangle(frame, (cx - 15, h // 2 - 15), (cx + 15, h // 2 + 15), (40, 200, 245), -1)
        cv2.circle(frame, (w // 2, h // 2), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened(), f"output not readable: {path}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
        return fps, w, h, frames
    finally:
        cap.release()


@pytest.fixture()
def input_clip(tmp_path):
    src = tmp_path / "input.mp4"
    _make_clip(src, seconds=4.0, fps=30)
    return src


def test_import_and_instantiation():
    engine = CinematicEngine()
    assert engine is not None
    assert hasattr(engine, "slow_motion_optical_flow")
    assert hasattr(engine, "add_cinematic_lut")


def test_slow_motion_optical_flow_creates_valid_mp4(tmp_path, input_clip):
    engine = CinematicEngine()
    out = tmp_path / "slow_output.mp4"
    result = engine.slow_motion_optical_flow(
        str(input_clip), str(out), slow_factor=0.25, target_segment=(1.0, 2.0)
    )
    assert result["success"] is True
    assert result["num_interpolated"] == 3  # 1/0.25 - 1
    assert out.exists() and out.stat().st_size > 0
    fps, w, h, frames = _probe(out)
    assert fps > 0 and w == 320 and h == 180
    # Slow motion adds frames inside the segment only, so output > source.
    assert frames > int(4.0 * 30)


def test_add_cinematic_lut_creates_valid_mp4(tmp_path, input_clip):
    engine = CinematicEngine()
    out = tmp_path / "cinematic_output.mp4"
    result = engine.add_cinematic_lut(str(input_clip), str(out), intensity=0.7)
    assert result["success"] is True
    assert abs(result["intensity"] - 0.7) < 1e-9
    assert out.exists() and out.stat().st_size > 0
    fps, w, h, frames = _probe(out)
    assert fps > 0 and w == 320 and h == 180
    assert frames == int(4.0 * 30)


def test_missing_input_raises(tmp_path):
    engine = CinematicEngine()
    with pytest.raises(FileNotFoundError):
        engine.slow_motion_optical_flow(
            str(tmp_path / "nope.mp4"), str(tmp_path / "o.mp4")
        )
    with pytest.raises(FileNotFoundError):
        engine.add_cinematic_lut(str(tmp_path / "nope.mp4"), str(tmp_path / "o.mp4"))


def test_invalid_slow_factor_raises(tmp_path, input_clip):
    engine = CinematicEngine()
    for bad in (0.0, -1.0, 2.0):
        with pytest.raises(ValueError):
            engine.slow_motion_optical_flow(
                str(input_clip), str(tmp_path / "o.mp4"), slow_factor=bad
            )


def test_invalid_segment_raises(tmp_path, input_clip):
    engine = CinematicEngine()
    with pytest.raises(ValueError):
        engine.slow_motion_optical_flow(
            str(input_clip), str(tmp_path / "o.mp4"), target_segment=(3.0, 1.0)
        )
    with pytest.raises(ValueError):
        engine.slow_motion_optical_flow(
            str(input_clip), str(tmp_path / "o.mp4"), target_segment=None
        )


def test_invalid_intensity_raises(tmp_path, input_clip):
    engine = CinematicEngine()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            engine.add_cinematic_lut(
                str(input_clip), str(tmp_path / "o.mp4"), intensity=bad
            )
