#!/usr/bin/env python3
"""Cinematic Director V2 — style-constrained, event-driven edit planning.

Pipeline position:

  Video Analysis -> Football Events -> Player/Subject Tracking
      -> Hero Moment Detection -> **Cinematic Director** -> Edit Plan V2
      -> Smart Reframing -> Speed Ramps -> Subject Isolation -> Color Grade
      -> Sound Design -> FFmpeg -> QC

The director never invents football actions, never copies a reference video's
timeline and never emits fixed timestamps. It takes:

  * the real source duration + a local motion profile (always available), and
  * optionally, real event evidence (yolo/event_engine.py) and real player/ball
    tracking (yolo/track_football.py),

and produces an Edit Plan V2 whose every decision is driven by the *measured*
style parameters of a ReferenceStyleProfile (or the mode defaults of
STANDARD / PRO / REFERENCE).

Hard guarantees enforced at the end (fail-safe, never throws):
  1. exactly 64.00s of output
  2. every source_start/source_end inside the real source duration
  3. player-first framing: the subject anchor is the tracked player/ball centre
     when tracking exists, else the motion centroid
  4. hard cut is the DEFAULT transition; dissolves/flashes are rationed by the
     reference profile
  5. slow motion only on event-driven windows (never randomly)
  6. no invented goal / celebration / player event
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TARGET_DURATION = 64.0

# ---------------------------------------------------------------------------
# Cinematic modes. STANDARD is the conservative default; PRO adds stronger
# dynamics; REFERENCE is driven by a measured ReferenceStyleProfile.
# ---------------------------------------------------------------------------
MODE_DEFAULTS = {
    "STANDARD": {
        "avg_shot_duration": 4.0,
        "cut_density": 0.25,
        "zoom_intensity": 0.010,
        "slow_motion_shot_ratio": 0.12,
        "text_per_shot": 0.25,
        "dissolve_ratio": 0.06,
        "flash_ratio": 0.04,
        "subject_bias": 0.60,
        "contrast": 1.16,
        "saturation": 1.02,
        "grain": 0.03,
    },
    "PRO": {
        "avg_shot_duration": 3.2,
        "cut_density": 0.31,
        "zoom_intensity": 0.016,
        "slow_motion_shot_ratio": 0.22,
        "text_per_shot": 0.20,
        "dissolve_ratio": 0.10,
        "flash_ratio": 0.05,
        "subject_bias": 0.72,
        "contrast": 1.22,
        "saturation": 1.06,
        "grain": 0.035,
    },
    "REFERENCE": {
        # Placeholder: overwritten by the measured ReferenceStyleProfile.
        "avg_shot_duration": 2.9,
        "cut_density": 0.34,
        "zoom_intensity": 0.014,
        "slow_motion_shot_ratio": 0.15,
        "text_per_shot": 0.10,
        "dissolve_ratio": 0.08,
        "flash_ratio": 0.04,
        "subject_bias": 0.85,
        "contrast": 1.20,
        "saturation": 1.04,
        "grain": 0.035,
    },
}

# Editorial captions that are *observations about the image*, never factual
# claims about the match. Templates like "WATCH THIS" are only used when the
# window really carries a strong, verified on-screen action signal.
CAPTION_GENERIC = ["THE TOUCH", "THE MOMENT", "ONE MORE STEP", "CLOSE", "THE BUILD UP", "NOW"]
CAPTION_ACTION = ["WATCH THIS", "HERE IT COMES", "RIGHT NOW", "THAT MOVE"]
CAPTION_HERO = ["THE MOMENT", "THIS IS IT", "NO WAY BACK"]
CAPTION_REACTION = ["THE FACE SAYS IT ALL", "AND THEN...", "LOOK AT HIM"]
CAPTION_OUTRO = ["LIVE THE GOAL", "ONE MORE TIME", "THE STORY ENDS HERE"]


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def profile_from_modes(mode: str, reference: dict | None) -> dict:
    mode = (mode or "STANDARD").upper()
    base = dict(MODE_DEFAULTS.get(mode, MODE_DEFAULTS["STANDARD"]))
    if mode == "REFERENCE":
        base = dict(MODE_DEFAULTS["REFERENCE"])
    if mode == "REFERENCE" and isinstance(reference, dict) and reference:
        def pick(key, fallback):
            v = reference.get(key)
            return float(v) if isinstance(v, (int, float)) and v else fallback

        shot_type_weights = reference.get("shot_type_weights") or {}
        trans = reference.get("transition_weights") or {}
        color = reference.get("color") or {}
        base.update({
            "avg_shot_duration": clamp(pick("avg_shot_duration", base["avg_shot_duration"]), 0.9, 9.0),
            "cut_density": clamp(pick("cut_density", base["cut_density"]), 0.08, 1.2),
            "zoom_intensity": clamp(pick("zoom_intensity", base["zoom_intensity"]), 0.0, 0.06),
            "slow_motion_shot_ratio": clamp(pick("slow_motion_shot_ratio", base["slow_motion_shot_ratio"]), 0.0, 0.5),
            "text_per_shot": clamp(pick("text_per_shot", base["text_per_shot"]), 0.0, 0.5),
            "subject_bias": clamp(0.45 + 0.5 * float(shot_type_weights.get("close_up", 0.3) or 0.3), 0.45, 0.95),
            "dissolve_ratio": clamp(float(trans.get("dissolve", 0.0) or 0.0), 0.0, 0.4),
            "flash_ratio": clamp(float(trans.get("flash", 0.0) or 0.0), 0.0, 0.15),
            "contrast": clamp(1.0 + (float(color.get("contrast", 100.0) or 100.0) - 100.0) / 300.0, 1.02, 1.38),
            "saturation": clamp(1.0 + (float(color.get("saturation", 120.0) or 120.0) - 120.0) / 420.0, 0.9, 1.18),
        })
    base["mode"] = mode
    return base


def slot_count(duration: float, style: dict) -> int:
    """Number of shots implied by the measured cut density (adaptive, bounded)."""
    by_density = int(round(duration * float(style["cut_density"])))
    by_duration = int(round(duration / max(0.7, float(style["avg_shot_duration"]))))
    n = int(round((by_density + 2 * by_duration) / 3.0))
    return max(8, min(30, n))


def build_motion_profile(duration: float, analysis: dict | None) -> tuple[list, float, float]:
    prof = []
    if isinstance(analysis, dict):
        prof = [p for p in (analysis.get("motion_profile") or [])
                if isinstance(p, dict) and isinstance(p.get("t"), (int, float)) and isinstance(p.get("energy"), (int, float))]
    if not prof:
        n = max(8, int(duration * 2))
        prof = [{"t": round(duration * (i + 0.5) / n, 3), "energy": 0.5} for i in range(n)]
    high = float((analysis or {}).get("thresholds", {}).get("high", 0.6) or 0.6)
    low = float((analysis or {}).get("thresholds", {}).get("low", 0.4) or 0.4)
    return prof, high, low


def energy_at(prof: list, t: float) -> float:
    best, best_d = prof[0]["energy"], abs(prof[0]["t"] - t)
    for p in prof:
        d = abs(p["t"] - t)
        if d < best_d:
            best, best_d = p["energy"], d
    return float(best)


def norm_energy(prof: list) -> list:
    """Normalise energies to 0..1. A genuinely FLAT profile (e.g. no real motion
    analysis available) keeps its absolute level instead of collapsing to 0, so a
    source without a measurable motion peak is never misread as "no energy at
    all" (which would wrongly flatten the whole edit)."""
    vals = [float(p["energy"]) for p in prof]
    lo, hi = min(vals), max(vals)
    if hi - lo < 0.05:
        return [{"t": float(p["t"]), "energy": clamp(float(p["energy"]), 0.0, 1.0)} for p in prof]
    span = max(1e-6, hi - lo)
    return [{"t": float(p["t"]), "energy": clamp((float(p["energy"]) - lo) / span)} for p in prof]


def load_events(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except Exception:
        return None


def load_tracking(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except Exception:
        return None


def build_track_index(tracking: dict | None, fps_hint: float) -> dict:
    """Per-frame subject anchors + per-track summaries (player as the subject).

    Uses real YOLO samples when present. Falls back to an empty index, in which
    case the director keeps the motion centroid as the subject anchor.
    """
    index = {"frames": {}, "tracks": {}, "fps": float(tracking.get("summary", {}).get("fps", fps_hint)) if tracking else fps_hint,
             "has_ball": False}
    if not isinstance(tracking, dict):
        return index
    samples = (tracking.get("summary") or {}).get("samples") or []
    ball_names = {"ball", "football", "soccer_ball", "sports ball", "sports_ball"}
    for s in samples:
        try:
            f = int(s.get("frame"))
            cls = str(s.get("class", "")).lower()
            x = float(s.get("x", 0.5))
            y = float(s.get("y", 0.5))
            conf = float(s.get("confidence", 0.5) or 0.5)
        except Exception:
            continue
        is_ball = cls in ball_names
        if is_ball:
            index["has_ball"] = True
        tid = str(s.get("track_id"))
        entry = index["frames"].setdefault(f, {"players": [], "ball": None})
        if is_ball:
            if entry["ball"] is None or conf > entry["ball"]["conf"]:
                entry["ball"] = {"x": x, "y": y, "conf": conf}
        else:
            entry["players"].append({"x": x, "y": y, "conf": conf, "track_id": tid})
        if not is_ball:
            t = index["tracks"].setdefault(tid, {"samples": [], "ball_dist": []})
            t["samples"].append((f, x, y, conf))
    for tid, t in index["tracks"].items():
        t["samples"].sort()
        t["size"] = len(t["samples"])
        t["first"] = t["samples"][0][0] if t["samples"] else 0
        t["last"] = t["samples"][-1][0] if t["samples"] else 0
        # Total path length = real movement; used to pick the protagonist.
        t["travel"] = 0.0
        for i in range(1, len(t["samples"])):
            _, x0, y0, _ = t["samples"][i - 1]
            _, x1, y1, _ = t["samples"][i]
            t["travel"] += math.hypot(x1 - x0, y1 - y0)
        t["ball_affinity"] = 0.0
    for f, entry in index["frames"].items():
        if not entry["ball"]:
            continue
        for p in entry["players"]:
            tid = p["track_id"]
            t = index["tracks"].get(tid)
            if t is not None:
                t["ball_affinity"] += max(0.0, 1.0 - math.hypot(p["x"] - entry["ball"]["x"], p["y"] - entry["ball"]["y"]) * 3.0)
    return index


def anchor_for(index: dict, f: int, protagonist: str | None) -> dict | None:
    """Subject anchor (crop focus) for a frame, with a bounded backward search."""
    frames = index.get("frames") or {}
    for step in range(0, 8):
        entry = frames.get(f - step)
        if not entry:
            continue
        players = entry["players"]
        chosen = None
        if protagonist:
            for p in players:
                if p["track_id"] == protagonist:
                    chosen = p
                    break
        if chosen is None and players:
            chosen = max(players, key=lambda p: p["conf"])
        if chosen is not None:
            anchor = {"x": chosen["x"], "y": chosen["y"], "source": "player_track",
                      "track_id": chosen.get("track_id"), "confidence": round(float(chosen["conf"]), 3)}
            if entry["ball"] is not None:
                anchor["ball_x"] = entry["ball"]["x"]
                anchor["ball_y"] = entry["ball"]["y"]
            return anchor
        if entry["ball"] is not None:
            return {"x": entry["ball"]["x"], "y": entry["ball"]["y"], "source": "ball_track",
                    "track_id": None, "confidence": round(float(entry["ball"]["conf"]), 3)}
    return None


def pick_protagonist(index: dict) -> str | None:
    """The camera protagonist = the player who moves the most with the ball."""
    best, best_score = None, -1.0
    for tid, t in index.get("tracks", {}).items():
        if t.get("size", 0) < 4:
            continue
        score = t.get("travel", 0.0) * 2.0 + t.get("ball_affinity", 0.0) * 0.9 + t.get("size", 0) * 0.002
        if score > best_score:
            best, best_score = tid, score
    return best


def window_energy(index: dict, protagonist: str | None, f: int, span: int = 4) -> tuple[float, str, bool]:
    """Player/ball engagement in a window: (score 0..1, dominant kind, has evidence)."""
    ball_hits, total = 0, 0
    min_dist = 1.0
    for f2 in range(f - span, f + span + 1):
        entry = (index.get("frames") or {}).get(f2)
        if not entry:
            continue
        total += 1
        target = None
        for p in entry["players"]:
            if protagonist and p["track_id"] == protagonist:
                target = p
                break
        if target is None and entry["players"]:
            target = max(entry["players"], key=lambda p: p["conf"])
        if entry["ball"] is not None:
            ball_hits += 1
            if target is not None:
                min_dist = min(min_dist, math.hypot(target["x"] - entry["ball"]["x"], target["y"] - entry["ball"]["y"]))
    if total == 0:
        return 0.0, "none", False
    proximity = 1.0 - clamp(min_dist)
    score = clamp(0.62 * (ball_hits / total) + 0.38 * proximity)
    return score, "ball_engagement" if ball_hits else "motion", ball_hits > 0


def event_windows(events: dict | None, duration: float) -> list:
    out = []
    if not isinstance(events, dict):
        return out
    for e in events.get("events", []) or []:
        try:
            t = float(e.get("time", 0.0))
        except Exception:
            continue
        if not (0.0 <= t < duration):
            continue
        out.append({
            "t": t,
            "score": clamp(e.get("event_score", 0.0) or 0.0),
            "ball_proximity": clamp(e.get("ball_proximity", 0.0) or 0.0),
            "pressure": clamp(e.get("pressure", 0.0) or 0.0),
            "direction_change": clamp(e.get("direction_change", 0.0) or 0.0),
            "ball_speed": clamp(e.get("ball_speed", 0.0) or 0.0),
            "acceleration": clamp(e.get("acceleration", 0.0) or 0.0),
            "labels": list(e.get("events", []) or []),
            "track_id": e.get("track_id"),
        })
    out.sort(key=lambda x: x["t"])
    return out


def nearest_event(windows: list, t: float, tol: float) -> dict | None:
    best, best_d = None, tol
    for w in windows:
        d = abs(w["t"] - t)
        if d <= best_d:
            best, best_d = w, d
    return best


def hero_moment(prof: list, windows: list, duration: float, low_band: float = 0.45, high_band: float = 0.55) -> dict:
    """Hero moment = the window that is BOTH visually strong and event-backed.

    Selection is bounded to the second half of the reel so the hero moment can
    play as the climax/ending rather than being spent in the hook.
    """
    lo_t, hi_t = duration * low_band, duration * high_band
    best, best_score = None, -1.0
    for p in prof:
        if not (lo_t <= p["t"] <= hi_t):
            continue
        ev = nearest_event(windows, p["t"], 1.2)
        ev_score = (0.55 * ev["score"] + 0.25 * ev["ball_proximity"] + 0.20 * ev["ball_speed"]) if ev else 0.0
        score = 0.45 * p["energy"] + 0.55 * ev_score
        if score > best_score:
            best, best_score = p, score
    if best is None:
        # No window in the band: take the global maximum, still bounded.
        best = max(prof, key=lambda p: p["energy"]) if prof else {"t": duration * 0.75, "energy": 0.5}
        best_score = 0.8 * float(best.get("energy", 0.5))
    return {"t": float(best["t"]), "energy": float(best.get("energy", 0.5)),
            "score": round(min(1.0, best_score), 4), "event_backed": bool(nearest_event(windows, best["t"], 1.2))}


def beat_roles(n: int, hero_index: int, duration: float) -> list:
    roles = []
    for i in range(n):
        rel = i / max(1, n - 1)
        if i == 0:
            roles.append("hook")
        elif i == n - 1:
            roles.append("outro")
        elif i == hero_index:
            roles.append("climax")
        elif i == hero_index - 1:
            roles.append("reaction")
        elif i == hero_index - 2:
            roles.append("impact")
        elif rel < 0.30:
            roles.append("setup")
        elif rel < 0.62:
            roles.append("escalation")
        elif rel < 0.85:
            roles.append("escalation")
        else:
            roles.append("reaction")
    return roles


def shot_type_for(role: str, energy: float, subject_evidence: bool, style: dict, style_weights: dict) -> str:
    """Player-first framing. Reference weights are a *tie-breaker*, never a
    replacement for the real evidence about the shot."""
    close_bias = float(style_weights.get("close_up", 0.0) or 0.0)
    if role == "hook":
        return "wide" if energy < 0.45 else "action"
    if role in ("climax", "impact"):
        return "extreme_close_up" if (subject_evidence or close_bias > 0.5) and style["subject_bias"] > 0.7 else "action"
    if role == "reaction":
        return "close_up"
    if role == "outro":
        return "close_up"
    # Player-first: when the reference style is close-up heavy, the subject shot
    # is the DEFAULT even before YOLO tracking confirms a specific player.
    if subject_evidence or close_bias > 0.35:
        return "close_up"
    if energy > 0.72:
        return "action"
    if energy < 0.35:
        return "medium"
    return "medium"


def build_timeline(prof, windows, index, style, style_weights, duration, hero, protagonist):
    n = slot_count(duration, style)
    hero_index = int(round(clamp(hero["t"] / max(1e-6, duration), 0.0, 1.0) * (n - 1)))
    hero_index = max(2, min(n - 2, hero_index))
    roles = beat_roles(n, hero_index, duration)

    # Source windows are drawn from the real video, hero window reserved.
    hero_t = clamp(hero["t"], 0.2, max(0.4, duration - 1.4))
    base_avg = duration / n
    clips = []
    for i in range(n):
        role = roles[i]
        if i == hero_index:
            centre = hero_t
        else:
            # Even spatial coverage of the source, then pulled toward motion
            # peaks so cuts follow the action instead of splitting blindly.
            centre = duration * (i + 0.5) / n
            if prof:
                near = min(prof, key=lambda p: abs(p["t"] - centre))
                centre = centre + 0.35 * (float(near["t"]) - centre)
        centre = clamp(centre, 0.2, max(0.3, duration - 0.6))
        energy = energy_at(prof, centre)
        ev = nearest_event(windows, centre, 1.0)
        ev_score = (0.5 * ev["score"] + 0.3 * ev["ball_proximity"] + 0.2 * ev["ball_speed"]) if ev else 0.0
        eng, kind, has_ev = window_energy(index, protagonist, int(round(centre * (index.get("fps") or 30.0))))
        subject_evidence = has_ev or eng > 0.35

        shot_type = shot_type_for(role, max(energy, ev_score), subject_evidence, style, style_weights)

        # Adaptive duration from the measured distribution: the hero gets the
        # longest hold, the hook stays short, everything else follows the style.
        dur_scale = {"hook": 0.75, "climax": 1.45, "reaction": 1.05, "outro": 1.35, "impact": 0.95}.get(role, 1.0)
        # Short-shot-heavy references also get short non-hero shots.
        if float(style["avg_shot_duration"]) < 2.2 and role not in ("climax", "outro"):
            dur_scale *= 0.85
        out_dur = max(0.55, base_avg * dur_scale)

        # ---- speed ramps: event-driven only ------------------------------
        slow_motion = False
        speed = 1.0
        if role in ("climax", "impact") and (has_ev or ev_score > 0.45) and eng > 0.30:
            slow_motion = True
            speed = 0.6 if role == "climax" else 0.75
        elif subject_evidence and energy >= 0.78 and i % max(2, int(1.0 / max(0.05, float(style["slow_motion_shot_ratio"])) or 3)) == 0:
            slow_motion = True
            speed = 0.8
        elif energy < 0.30:
            speed = 1.15

        # ---- smart reframing: real tracking anchor, eased ---------------
        f_centre = int(round(centre * (index.get("fps") or 30.0)))
        anchor = anchor_for(index, f_centre, protagonist)
        if anchor:
            crop_x, crop_y = clamp(anchor["x"], 0.12, 0.88), clamp(anchor["y"], 0.15, 0.85)
            anchor_source = anchor["source"]
        else:
            # Motion fallback: keep the frame centre but bias vertically to the
            # subject band (players occupy the lower half of a pitch shot).
            crop_x, crop_y = 0.5, 0.56
            anchor_source = "motion_fallback"

        zoom_intensity = float(style["zoom_intensity"])
        # tight = how subject-centric the style is. It sets a framing FLOOR so a
        # low-energy window is never rendered as a flat full-pitch wide shot.
        tight = clamp(0.5 + (float(style["subject_bias"]) - 0.45) * 0.95, 0.5, 1.0)
        if shot_type in ("extreme_close_up", "eye_close_up", "detail"):
            zoom_start, zoom_end = 1.08 + 0.06 * tight, 1.08 + 0.06 * tight + max(0.03, zoom_intensity * 1.6)
        elif shot_type in ("close_up", "reaction"):
            zoom_start, zoom_end = 1.04 + 0.05 * tight, 1.04 + 0.05 * tight + max(0.04, zoom_intensity * 2.0)
        elif shot_type == "action":
            zoom_start, zoom_end = 1.02 + 0.03 * tight, 1.02 + 0.03 * tight + max(0.05, zoom_intensity * 2.4)
        else:
            zoom_start, zoom_end = 1.0 + 0.02 * tight, 1.0 + 0.02 * tight + max(0.03, zoom_intensity * 1.6)
        if role == "outro":
            # Ending is intentional: an eased pull-out instead of a punch-in.
            zoom_start, zoom_end = max(1.04, zoom_end), 1.02 + 0.02 * tight
        if energy < 0.25 and role not in ("climax", "reaction"):
            # Calm window: keep the subject framing, just remove the push.
            zoom_end = zoom_start + 0.01

        # ---- captions: rare, meaningful, never a fake factual claim -----
        text = ""
        text_budget = float(style["text_per_shot"])
        wants_text = (i % max(2, int(round(1.0 / max(0.05, text_budget)))) == 0)
        if role == "hook" and energy > 0.6:
            text = CAPTION_GENERIC[0]
        elif role in ("climax", "impact") and subject_evidence:
            text = CAPTION_HERO[0] if role == "climax" else CAPTION_ACTION[0]
        elif role == "reaction":
            text = CAPTION_REACTION[0]
        elif role == "outro":
            text = CAPTION_OUTRO[0]
        elif wants_text and energy > 0.45:
            text = CAPTION_GENERIC[(i // 2) % len(CAPTION_GENERIC)]
        # NEVER template a factual claim (goal / score / name).

        clip = {
            "timeline_index": 0,  # renumbered after the total duration is fixed
            "source_start": round(centre, 3),
            "source_end": round(min(duration, centre + out_dur * max(0.6, speed)), 3),
            "output_start": 0.0,
            "output_end": round(out_dur, 3),
            "action": f"{role} @ {centre:.1f}s (energy {energy:.2f}{', ball-tracked' if has_ev else ''})",
            "importance": int(round(2 + 8 * max(energy, ev_score, hero["score"] if i == hero_index else 0.0))),
            "speed": round(speed, 3),
            "zoom_start": round(zoom_start, 3),
            "zoom_end": round(zoom_end, 3),
            "crop_x": round(crop_x, 4),
            "crop_y": round(crop_y, 4),
            "subject_anchor": anchor_source,
            "slow_motion": slow_motion,
            "subject_isolation": bool(subject_evidence and role in ("climax", "impact", "reaction")),
            "isolation_reason": role if subject_evidence else "",
            "transition": "hard_cut",
            "text": text,
            "shot_type": shot_type,
            "beat_role": role,
            "veo_needed": False,
            "veo_prompt": "",
            "evidence": {"event_score": round(ev_score, 3), "ball_window_score": round(eng, 3),
                         "has_ball_evidence": bool(has_ev), "energy": round(energy, 3)},
            "_weight": out_dur,
        }
        clips.append(clip)

    # ---- exact 64.00s allocation (source interval stays the authority) -----
    total_weight = sum(c["_weight"] for c in clips) or 1.0
    t = 0.0
    for i, c in enumerate(clips):
        share = TARGET_DURATION * c.pop("_weight") / total_weight
        c["output_start"] = round(t, 3)
        t = TARGET_DURATION if i == len(clips) - 1 else t + share
        c["output_end"] = round(t, 3)
        c["timeline_index"] = i
    clips[-1]["output_end"] = TARGET_DURATION

    # ---- transition rationing: hard cut is the default --------------------
    dissolve_budget = int(round(len(clips) * float(style["dissolve_ratio"])))
    flash_budget = int(round(len(clips) * float(style["flash_ratio"])))
    for c in clips:
        if c["beat_role"] == "climax" and c["evidence"]["has_ball_evidence"] and flash_budget > 0:
            c["transition"] = "flash"
            flash_budget -= 1
    for i in range(1, len(clips)):
        if dissolve_budget > 0 and clips[i - 1]["beat_role"] != clips[i]["beat_role"] and clips[i]["beat_role"] in ("reaction", "outro"):
            clips[i]["transition"] = "fade" if clips[i]["beat_role"] == "outro" else "dissolve"
            dissolve_budget -= 1

    return clips


def build_plan(duration: float, analysis: dict | None, mode: str, reference: dict | None,
               events: dict | None, tracking: dict | None, subject_name: str) -> dict:
    style = profile_from_modes(mode, reference)
    prof_raw, high_thr, low_thr = build_motion_profile(duration, analysis)
    prof = norm_energy(prof_raw)
    windows = event_windows(events, duration)
    index = build_track_index(tracking, 30.0)
    protagonist = pick_protagonist(index) if index.get("frames") else None
    hero = hero_moment(prof, windows, duration)
    style_weights = (reference or {}).get("shot_type_weights", {}) if isinstance(reference, dict) else {}

    timeline = build_timeline(prof, windows, index, style, style_weights, duration, hero, protagonist)

    # Sound design cues follow the real evidence, never a fixed arrangement.
    impacts = sorted([c["timeline_index"] for c in timeline
                      if c["beat_role"] in ("impact", "climax") and c["evidence"]["has_ball_evidence"]])
    riser_target = next((c for c in timeline if c["beat_role"] == "climax"), None)
    riser_times = []
    if riser_target:
        riser_times = [round(max(0.0, riser_target["output_start"] - 1.2), 2)]

    color = {
        "contrast": round(float(style["contrast"]), 3),
        "saturation": round(float(style["saturation"]), 3),
        "highlights": 0.03,
        "shadows": -0.02,
        "grain": round(float(style["grain"]), 3),
        "protect_skin_tones": True,
        "cooler_shadows": True,
        "warmer_highlights": True,
        "prevent_neon_grass": True,
    }
    if isinstance(reference, dict):
        rc = reference.get("color") or {}
        if isinstance(rc.get("skin_ratio"), (int, float)):
            color["reference_skin_ratio"] = rc["skin_ratio"]
        if isinstance(rc.get("neon_grass_ratio"), (int, float)):
            color["reference_neon_grass_ratio"] = rc["neon_grass_ratio"]

    plan = {
        "duration": TARGET_DURATION,
        "aspect_ratio": "9:16",
        "subject": {"name": subject_name or "Main player",
                    "confidence": 0.85 if protagonist else (0.6 if index.get("frames") else 0.45),
                    "track_id": protagonist},
        "timeline": timeline,
        "music": {
            "style": "Emotional cinematic football / dark pulse",
            "bpm": 126 if mode != "STANDARD" else 118,
            "energy_curve": [0.72, 0.55, 0.62, 0.74, 0.86, 1.0, 0.8, 0.34],
        },
        "color_grade": color,
        "sound_design": {
            "impact_clips": impacts,
            "riser_starts": riser_times,
            "silence_before_ms": 220 if float(hero["score"]) > 0.7 else 120,
            "impact_source": "ball_engagement_evidence" if impacts else "none",
        },
        "style_profile": {
            "name": f"{mode} CINEMATIC",
            "mode": mode,
            "average_shot_duration": round(TARGET_DURATION / max(1, len(timeline)), 3),
            "zoom_intensity": round(float(style["zoom_intensity"]), 4),
            "transition_frequency": round(1.0 - float(style["dissolve_ratio"]) - float(style["flash_ratio"]), 3),
            "slow_motion_frequency": round(float(style["slow_motion_shot_ratio"]), 3),
            "text_frequency": round(float(style["text_per_shot"]), 3),
            "caption_style": "small white uppercase, lower third, no opaque box",
            "visual_language": "player-first, event-driven hard-cut montage with eased dynamic reframing",
            "source": (reference or {}).get("source") if isinstance(reference, dict) else "mode-defaults",
        },
        "style_name": f"{mode} CINEMATIC",
        "cinematic_director": {
            "version": "v11",
            "mode": mode,
            "evidence_level": "high" if (windows and index.get("has_ball")) else "medium" if (windows or index.get("frames")) else "low",
            "hero_moment": hero,
            "protagonist_track": protagonist,
            "ball_tracked": bool(index.get("has_ball")),
            "player_tracks": len([t for t in (index.get("tracks") or {}).values() if t.get("size", 0) >= 4]),
            "subject_isolation_clips": [c["timeline_index"] for c in timeline if c["subject_isolation"]],
            "speed_ramp_clips": [c["timeline_index"] for c in timeline if c["slow_motion"]],
            "transition_mix": {k: sum(1 for c in timeline if c["transition"] == k)
                               for k in {c["transition"] for c in timeline}},
            "rules": {
                "player_is_subject": True,
                "no_static_crop_zoom": True,
                "no_full_frame_blur_default": True,
                "sam_optional_with_yolo_then_motion_fallback": True,
                "slow_motion_event_driven": True,
                "cuts_follow_the_action": True,
                "no_invented_goal_or_player_event": True,
                "reference_is_style_only_not_a_template": True,
            },
        },
    }
    return plan


def validate(plan: dict, duration: float) -> dict:
    tl = plan["timeline"]
    problems = []
    if len(tl) < 6:
        problems.append("timeline_too_short")
    if abs(float(tl[-1]["output_end"]) - TARGET_DURATION) > 0.02:
        problems.append("output_not_64s")
    for c in tl:
        if c["source_start"] < 0 or c["source_end"] > duration + 0.001 or c["source_end"] <= c["source_start"]:
            problems.append(f"clip_{c['timeline_index']}_invalid_source_window")
    for c in tl:
        if not (0.0 <= c["crop_x"] <= 1.0 and 0.0 <= c["crop_y"] <= 1.0):
            problems.append(f"clip_{c['timeline_index']}_invalid_crop")
        if c["transition"] not in ("hard_cut", "dissolve", "flash", "fade", "match_cut", "directional_blur"):
            problems.append(f"clip_{c['timeline_index']}_invalid_transition")
    for i in range(1, len(tl)):
        if abs(tl[i]["output_start"] - tl[i - 1]["output_end"]) > 0.02:
            problems.append("output_timeline_not_contiguous")
            break
    plan["validation"] = {"ok": not problems, "problems": problems}
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--motion", default=None, help="local motion analysis JSON (optional)")
    ap.add_argument("--events", default=None, help="yolo/event_engine.py output JSON (optional)")
    ap.add_argument("--tracking", default=None, help="yolo/track_football.py output JSON (optional)")
    ap.add_argument("--reference-style", default=None, help="ReferenceStyleProfile JSON (optional)")
    ap.add_argument("--mode", default="STANDARD", choices=["STANDARD", "PRO", "REFERENCE"])
    ap.add_argument("--subject", default="Main player")
    a = ap.parse_args()

    try:
        motion = json.loads(Path(a.motion).read_text(encoding="utf8")) if a.motion and Path(a.motion).exists() else None
        reference = json.loads(Path(a.reference_style).read_text(encoding="utf8")) if a.reference_style and Path(a.reference_style).exists() else None
        duration = max(1.0, float(a.duration))
        plan = build_plan(duration, motion, a.mode, reference, load_events(a.events), load_tracking(a.tracking), a.subject)
        plan = validate(plan, duration)
        Path(a.output).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({
            "success": True,
            "plan": a.output,
            "mode": a.mode,
            "clips": len(plan["timeline"]),
            "hero_moment": plan["cinematic_director"]["hero_moment"],
            "evidence_level": plan["cinematic_director"]["evidence_level"],
            "validation": plan["validation"],
        }))
        return 0 if plan["validation"]["ok"] else 1
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
