#!/usr/bin/env python3
"""Style QC — compare a rendered 64s master against a ReferenceStyleProfile.

This is the QC step of the pipeline:

  Video Analysis -> Football Events -> Player/Subject Tracking
      -> Hero Moment Detection -> Cinematic Director -> Edit Plan V2
      -> Smart Reframing -> Speed Ramps -> Subject Isolation -> Color Grade
      -> Sound Design -> FFmpeg -> **QC**

It measures the SAME style parameters on the produced master (via
reference_style_analyzer) and reports the distance to the reference profile, so
"close to the reference" is a measured fact rather than an impression. It also
verifies the hard output contract (1080x1920 / 30fps / 64.00s / yuv420p / aac).

Usage:
  python3 yolo/style_qc.py --video out.mp4 --profile docs/reference_style_profile.json --output qc.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_style_analyzer import analyse_video  # noqa: E402

TARGET = {"width": 1080, "height": 1920, "fps": 30.0, "duration": 64.0}


def probe(path: str) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-400:])
    return json.loads(p.stdout)


def relative_error(actual: float, target: float) -> float:
    if target == 0:
        return 0.0 if actual == 0 else 1.0
    return abs(actual - target) / abs(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    report: dict = {"success": False, "video": a.video, "checks": {}, "comparison": {}}
    try:
        meta = probe(a.video)
        vs = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
        aud = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), None)
        duration = float(meta.get("format", {}).get("duration") or 0)
        fps_num, fps_den = (vs.get("r_frame_rate", "30/1").split("/") + ["1"])[:2] if vs else ("30", "1")
        fps = float(fps_num) / max(1.0, float(fps_den))

        report["checks"] = {
            "width": int(vs.get("width", 0)) if vs else 0,
            "height": int(vs.get("height", 0)) if vs else 0,
            "fps": round(fps, 3),
            "duration": round(duration, 3),
            "video_codec": vs.get("codec_name") if vs else None,
            "pix_fmt": vs.get("pix_fmt") if vs else None,
            "audio_codec": aud.get("codec_name") if aud else None,
            "width_ok": bool(vs and int(vs.get("width", 0)) == TARGET["width"]),
            "height_ok": bool(vs and int(vs.get("height", 0)) == TARGET["height"]),
            "fps_ok": abs(fps - TARGET["fps"]) < 0.05,
            "duration_ok": abs(duration - TARGET["duration"]) < 0.15,
            "pix_fmt_ok": bool(vs and vs.get("pix_fmt") == "yuv420p"),
            "h264_ok": bool(vs and vs.get("codec_name") == "h264"),
            "aac_ok": bool(aud and aud.get("codec_name") == "aac"),
        }
        report["checks"]["contract_ok"] = all(report["checks"][k] for k in
                                              ("width_ok", "height_ok", "fps_ok", "duration_ok", "pix_fmt_ok", "h264_ok"))

        target = json.loads(Path(a.profile).read_text(encoding="utf8"))
        actual = analyse_video(a.video)

        pairs = {
            "avg_shot_duration": (actual["avg_shot_duration"], target.get("avg_shot_duration")),
            "cut_density": (actual["cut_density"], target.get("cut_density")),
            "zoom_intensity": (actual["zoom_intensity"], target.get("zoom_intensity")),
            "slow_motion_shot_ratio": (actual["slow_motion_shot_ratio"], target.get("slow_motion_shot_ratio")),
            "subject_shot_ratio": (actual["subject_shot_ratio"], target.get("subject_shot_ratio")),
            "text_per_shot": (actual["text_per_shot"], target.get("text_per_shot")),
            "hard_cut_ratio": (actual["hard_cut_ratio"], target.get("hard_cut_ratio", target.get("transition_weights", {}).get("hard_cut"))),
        }
        comparison = {}
        for key, (act, tgt) in pairs.items():
            if tgt is None:
                continue
            err = relative_error(float(act), float(tgt))
            comparison[key] = {
                "actual": round(float(act), 5),
                "reference": round(float(tgt), 5),
                "relative_error": round(err, 4),
                "match": err <= 0.35,
            }
        color_actual = actual.get("color", {})
        color_target = target.get("color", {})
        comparison["color_contrast"] = {
            "actual": round(float(color_actual.get("contrast", 0)), 3),
            "reference": round(float(color_target.get("contrast", 0)), 3),
            "relative_error": round(relative_error(float(color_actual.get("contrast", 0)), float(color_target.get("contrast", 1) or 1)), 4),
        }
        comparison["color_saturation"] = {
            "actual": round(float(color_actual.get("saturation", 0)), 3),
            "reference": round(float(color_target.get("saturation", 0)), 3),
            "relative_error": round(relative_error(float(color_actual.get("saturation", 0)), float(color_target.get("saturation", 1) or 1)), 4),
        }
        report["comparison"] = comparison
        matched = [k for k, v in comparison.items() if v.get("match") is True]
        report["style_match_score"] = round(len(matched) / max(1, len([k for k, v in comparison.items() if "match" in v])), 3)
        report["matched_parameters"] = sorted(matched)
        report["actual_profile_summary"] = {
            "shot_count": actual["shot_count"],
            "avg_shot_duration": actual["avg_shot_duration"],
            "cut_density": actual["cut_density"],
            "subject_shot_ratio": actual["subject_shot_ratio"],
            "transition_weights": actual["transition_weights"],
            "color": color_actual,
        }
        report["audio"] = actual.get("audio", {})
        report["success"] = bool(report["checks"]["contract_ok"])
    except Exception as e:
        report["error"] = str(e)

    Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: report[k] for k in ("success", "style_match_score", "checks") if k in report}, ensure_ascii=False))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
