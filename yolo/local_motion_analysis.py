#!/usr/bin/env python3
"""
LOCAL MOTION ANALYSIS — no external API, no API keys.

Replaces the Gemini "analyze-video" call with a pure local OpenCV pass.

The script samples the real uploaded footage, measures the frame-to-frame
motion energy (cv2.absdiff mean), and returns:

    - motion_profile : normalised per-sample energy curve (0..1)
    - action_moments : high-motion samples (sprints / fast action / crowd)
    - slow_moments   : low-motion samples (dribble / setup / build-up / replay)
    - peaks          : local motion maxima (candidate impact shots)

It is intentionally cheap: frames are downscaled to a small width and sampled
at a low rate, so a full match runs in a few seconds on a 512MB CPU instance.

Usage:
    python3 local_motion_analysis.py --video match.mp4
    python3 local_motion_analysis.py --video match.mp4 --sample-fps 4 --output motion.json

Exit code 0 even when OpenCV is unavailable: it then emits a valid "uniform"
profile so the pipeline can still build an honest source-only edit plan.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _percentile(values, pct):
    """Small dependency-free percentile (no numpy required)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _moving_average(values, window=3):
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def uniform_profile(duration, reason):
    """Honest fallback: no invention, just evenly spread sampling points."""
    duration = max(0.1, float(duration or 0.1))
    samples = 48
    step = duration / samples
    profile = [{"t": round(i * step, 3), "energy": 0.5} for i in range(samples)]
    return {
        "success": True,
        "model": "fallback-uniform",
        "fallback_reason": reason,
        "duration": round(duration, 3),
        "fps": 25.0,
        "sampled_fps": round(samples / duration, 3),
        "samples": samples,
        "motion_profile": profile,
        "slow_moments": [],
        "action_moments": [],
        "peaks": [],
        "thresholds": {"low": 0.4, "high": 0.6},
        "mean_energy": 0.5,
        "max_energy": 0.5,
    }


def analyse(video_path, sample_fps=4.0, max_seconds=300.0, max_width=160):
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return uniform_profile(max_seconds, f"opencv-unavailable: {exc}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return uniform_profile(max_seconds, "video-open-failed")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.5 or fps > 240:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0

    # Bound the work on very long sources.
    if duration <= 0 or duration > max_seconds:
        duration = min(max_seconds, duration if duration > 0 else max_seconds)

    sample_fps = max(0.5, min(12.0, float(sample_fps)))
    step = max(1, int(round(fps / sample_fps)))
    max_frames = int(duration * fps)

    energies = []
    times = []

    prev_gray = None
    idx = 0
    started = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                # Downscale for speed + noise reduction.
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / float(w)
                    frame = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                if prev_gray is not None and prev_gray.shape == gray.shape:
                    diff = cv2.absdiff(gray, prev_gray)
                    energy = float(diff.mean()) / 255.0
                    energies.append(energy)
                    times.append(idx / fps)
                prev_gray = gray

            idx += 1
            if max_frames > 0 and idx >= max_frames:
                break
            # Hard safety stop (never let a corrupt file hang the render).
            if time.time() - started > 180:
                break
    finally:
        cap.release()

    if len(energies) < 3:
        return uniform_profile(duration, "too-few-samples")

    smoothed = _moving_average(energies, 3)
    peak_energy = max(smoothed) or 1.0
    normalised = [min(1.0, max(0.0, e / peak_energy)) for e in smoothed]

    low_thr = _percentile(normalised, 30)
    high_thr = _percentile(normalised, 75)
    mean_energy = sum(normalised) / len(normalised)

    profile = [
        {"t": round(times[i], 3), "energy": round(normalised[i], 5)}
        for i in range(len(normalised))
    ]

    slow_moments = [profile[i]["t"] for i in range(len(normalised)) if normalised[i] <= low_thr]
    action_moments = [
        {"t": profile[i]["t"], "energy": round(normalised[i], 5)}
        for i in range(len(normalised))
        if normalised[i] >= high_thr
    ]

    # Local maxima -> candidate impact shots.
    peaks = []
    for i in range(1, len(normalised) - 1):
        if normalised[i] >= normalised[i - 1] and normalised[i] >= normalised[i + 1] and normalised[i] >= high_thr * 0.8:
            peaks.append({"t": profile[i]["t"], "energy": round(normalised[i], 5)})

    return {
        "success": True,
        "model": "local-opencv-motion",
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "sampled_fps": round(len(normalised) / max(0.1, duration), 3),
        "samples": len(normalised),
        "motion_profile": profile,
        "slow_moments": [round(float(t), 3) for t in slow_moments],
        "action_moments": action_moments,
        "peaks": peaks,
        "thresholds": {"low": round(low_thr, 5), "high": round(high_thr, 5)},
        "mean_energy": round(mean_energy, 5),
        "max_energy": 1.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Local OpenCV motion analysis (no API keys)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--sample-fps", type=float, default=4.0)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--max-width", type=int, default=160)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try:
        result = analyse(args.video, args.sample_fps, args.max_seconds, args.max_width)
    except Exception as exc:  # never crash the pipeline
        result = uniform_profile(args.max_seconds, f"unhandled: {exc}")

    text = json.dumps(result, ensure_ascii=False)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
