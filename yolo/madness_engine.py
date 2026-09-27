#!/usr/bin/env python3
"""MADNESS ENGINE v5 — evidence-gated cinematic intensity.

This module does not invent football events. It only upgrades verified/high-confidence
timeline moments into increasingly aggressive visual treatments. Level 4/5 are
cinematic simulations built from the real source footage, not claims of literal POV
or generated reality.
"""
import argparse, json, math, re
from pathlib import Path

LEVELS = {
    1: "cinematic",
    2: "exceptional",
    3: "crazy",
    4: "hallucination",
    5: "absolute_madness",
}

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))

def overlap_score(events, start, end):
    hits = [e for e in events if float(e.get("time", -1)) >= start - 0.35 and float(e.get("time", -1)) <= end + 0.35]
    if not hits:
        return 0.0, []
    return max(float(e.get("event_score", 0)) for e in hits), hits

def infer_intensity(clip, score, hits):
    text = " ".join([
        str(clip.get("action", "")),
        str(clip.get("text", "")),
        str(clip.get("shot_type", "")),
        str(clip.get("beat_role", "")),
    ]).lower()
    labels = []
    for h in hits:
        labels.extend([str(x).lower() for x in h.get("events", [])])
    joined = " ".join(labels)
    duel = any(k in text or k in joined for k in [
        "duel","drib","nutmeg","feint","direction_change","pressure","tackle","fall","1v1"
    ])
    impact = any(k in text or k in joined for k in [
        "shot","strike","goal","finish","impact","explosive","ball_speed"
    ])
    reaction = any(k in text for k in ["reaction","celebr","crowd","fall"])
    importance = clamp(float(clip.get("importance", 5)) / 10)
    evidence = clamp(score)
    raw = .38 * evidence + .28 * importance + .16 * (1 if duel else 0) + .10 * (1 if impact else 0) + .08 * (1 if reaction else 0)
    return clamp(raw), duel, impact, reaction

def choose_level(clip, score, hits):
    intensity, duel, impact, reaction = infer_intensity(clip, score, hits)
    role = str(clip.get("beat_role", "")).lower()
    importance = float(clip.get("importance", 5))
    # Level 5 is intentionally rare: climax + strong evidence + duel/impact.
    if role == "climax" and importance >= 8 and score >= .72 and (duel or impact):
        return 5, intensity
    if intensity >= .72 and score >= .62 and (duel or impact):
        return 4, intensity
    if intensity >= .55 and score >= .50:
        return 3, intensity
    if intensity >= .40:
        return 2, intensity
    return 1, intensity

def effect_for(level):
    return {
        1: {"name":"cinematic_pulse","transition":"hard_cut"},
        2: {"name":"impact_punch","transition":"directional_blur"},
        3: {"name":"selective_color_freeze","transition":"hard_cut",
            "description":"desaturated frame with a colored focal window, micro-freeze and heartbeat-like visual tension"},
        4: {"name":"defender_nightmare","transition":"directional_blur",
            "description":"Dutch-angle simulation, temporal echo and chromatic visual instability"},
        5: {"name":"absolute_madness","transition":"hard_cut",
            "description":"dark breath-in, POV-like crop, silence hit, violent punch-in and psychological typography"},
    }[level]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    event_data = json.loads(Path(args.events).read_text(encoding="utf-8"))
    timeline_data = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    events = event_data.get("events", []) if isinstance(event_data, dict) else []
    timeline = timeline_data.get("timeline", timeline_data) if isinstance(timeline_data, dict) else timeline_data

    if not isinstance(timeline, list):
        raise SystemExit("timeline must be an array")

    out = []
    for idx, clip in enumerate(timeline):
        start = float(clip.get("source_start", 0))
        end = float(clip.get("source_end", start + 0.1))
        score, hits = overlap_score(events, start, end)
        level, intensity = choose_level(clip, score, hits)
        effect = effect_for(level)
        c = dict(clip)
        c["madness"] = {
            "enabled": level >= 3,
            "level": level,
            "name": LEVELS[level],
            "intensity": round(intensity, 4),
            "evidence_score": round(score, 4),
            "effect": effect["name"],
            "evidence_events": sorted(set(
                str(x) for h in hits for x in h.get("events", [])
            ))[:8],
            "reason": "evidence-gated cinematic escalation",
            "safety": "real_footage_only",
        }
        out.append(c)

    counts = {str(i): sum(1 for c in out if c["madness"]["level"] == i) for i in range(1, 6)}
    # Never allow a montage to become all madness. Level 5 is max one shot.
    level5 = [i for i,c in enumerate(out) if c["madness"]["level"] == 5]
    if len(level5) > 1:
        for i in level5[1:]:
            out[i]["madness"]["level"] = 4
            out[i]["madness"]["name"] = LEVELS[4]
            out[i]["madness"]["effect"] = effect_for(4)["name"]
            out[i]["madness"]["enabled"] = True
    result = {
        "success": True,
        "version": "MADNESS-5",
        "policy": "evidence_gated_real_footage_only",
        "levels": LEVELS,
        "counts": {str(i): sum(1 for c in out if c["madness"]["level"] == i) for i in range(1, 6)},
        "timeline": out,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "version": "MADNESS-5", "counts": result["counts"]}))

if __name__ == "__main__":
    main()
