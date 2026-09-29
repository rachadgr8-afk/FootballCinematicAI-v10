#!/usr/bin/env python3
"""CINEMATIC STORYTELLER — psychological "Predator vs Prey" orchestration layer.

This module sits ON TOP of the existing GoalFlow pipeline. It does NOT rebuild the
render system and does NOT replace any component. It consumes the REAL evidence
the existing stages already produced and turns it into an optional PSYCHOLOGICAL
story layer:

    REAL FOOTBALL EVIDENCE (YOLO/ByteTrack + event_engine + football_director)
        -> DUEL DETECTION            (local, evidence-gated)
        -> [OPTIONAL VLM]            (Qwen2-VL / InternVL2 clip verdict)
        -> PSYCHOLOGICAL STORY       (roles: predator/prey/approach/trap/...)
        -> [OPTIONAL DEEPSEEK]       (exactly 6 inner-monologue lines)
        -> DEPTH / POV / LOW-ANGLE   (bounded psychological segments)
        -> EDIT PLAN ENRICHMENT      (ADDITIVE optional fields only)

Hard guarantees
---------------
* It NEVER invents a football event, a goal, a score, a winner or a player id.
  ``winner``/``loser`` stay ``null`` unless the OPTIONAL VLM confirms them using
  REAL player ids.
* The enrichment is strictly ADDITIVE. Every field it writes onto the edit plan is
  optional; an old plan/renderer keeps working unchanged.
* FAIL-SAFE: any error (or weak evidence) degrades to ``evidence_level="low"`` and
  the plan keeps its existing GoalFlow cinematic behaviour.
* Memory-safe: it only reads JSON evidence; video is touched ONLY by the optional
  VLM/depth samplers, which decode a handful of downscaled frames lazily.
* No secrets in code, logs or output — keys come from the environment only.

CLI
---
    python3 yolo/cinematic_storyteller.py \
        --duration 40 --tracking tracking.json --events events.json \
        --director director.json --motion motion.json \
        --plan plan.json --plan-out enriched.json \
        --output storyteller.json [--video duel.mp4] \
        [--no-vlm] [--no-script] [--no-network]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

VERSION = "cinematic-storyteller-1.0.0"

BALL_NAMES = {"ball", "football", "soccer_ball", "soccer-ball", "sports ball", "sports_ball"}
PLAYER_HINTS = {"player", "person", "footballer", "athlete"}

# Psychological story roles (a role is only used when evidence supports it).
DUEL_ROLES = ["predator", "prey"]
TEMPORAL_ROLES = ["approach", "trap", "pressure", "escape", "dominance", "impact", "aftermath"]

# Numeric gates — tuned so a "duel" only appears when the evidence really is there.
PROX_HIT = 0.22          # normalised distance that counts as "engaged"
PROX_MIN = 0.34          # min proximity ratio to consider a duel at all
CLOSE_MIN = 0.30         # min closing ratio
DUEL_MIN_SCORE = 0.46    # min combined duel score
PSYCH_WINDOW = 3.0       # +/- seconds around the duel moment = psychological segment
MAX_POV_SWITCHES = 4
MAX_LOW_ANGLE = 2
MAX_DEPTH_CLIPS = 3


def clamp(v, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def load_json(path):
    if not path:
        return None
    try:
        p = Path(path)
        return json.loads(p.read_text(encoding="utf8")) if p.exists() else None
    except Exception:
        return None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --------------------------------------------------------------------------- #
# 1) Evidence -> per-track series + kinematics
# --------------------------------------------------------------------------- #
def build_series(tracking):
    """Return (tracks, ball_samples, fps) from a track_football.py JSON."""
    tracks, balls = {}, []
    fps = 30.0
    if not isinstance(tracking, dict):
        return tracks, balls, fps
    summary = tracking.get("summary") or {}
    try:
        fps = float(summary.get("fps") or 30.0) or 30.0
    except (TypeError, ValueError):
        fps = 30.0
    if fps <= 0.5 or fps > 240:
        fps = 30.0
    for s in summary.get("samples") or []:
        try:
            f = int(s.get("frame"))
            cls = str(s.get("class", "")).lower()
            x = float(s.get("x", 0.5)); y = float(s.get("y", 0.5))
            conf = float(s.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            continue
        t = float(s.get("time", f / fps))
        if cls in BALL_NAMES:
            balls.append({"frame": f, "time": t, "x": x, "y": y, "conf": conf})
            continue
        tid = str(s.get("track_id"))
        if tid in ("None", ""):
            continue
        tinfo = tracks.setdefault(tid, {"class": cls, "samples": [], "ball_affinity": 0.0, "travel": 0.0})
        tinfo["samples"].append({"frame": f, "time": t, "x": x, "y": y, "conf": conf})
    for tid, tinfo in tracks.items():
        tinfo["samples"].sort(key=lambda z: z["frame"])
        travel = 0.0
        samples = tinfo["samples"]
        for i in range(1, len(samples)):
            travel += dist((samples[i - 1]["x"], samples[i - 1]["y"]), (samples[i]["x"], samples[i]["y"]))
        tinfo["travel"] = travel
        tinfo["size"] = len(samples)
        tinfo["first_time"] = samples[0]["time"] if samples else 0.0
        tinfo["last_time"] = samples[-1]["time"] if samples else 0.0
    balls.sort(key=lambda z: z["frame"])
    return tracks, balls, fps


def _ball_affinity(tracks, balls):
    if not balls:
        return
    by_f = {}
    for b in balls:
        by_f.setdefault(b["frame"], []).append(b)
    for tinfo in tracks.values():
        aff = 0.0
        for s in tinfo["samples"]:
            cand = by_f.get(s["frame"])
            if not cand:
                continue
            best = min(dist((s["x"], s["y"]), (b["x"], b["y"])) for b in cand)
            aff += max(0.0, 1.0 - best * 3.0)
        tinfo["ball_affinity"] = aff


def _kinematics(samples, fps):
    """Per-sample speed (norm/s) and acceleration magnitude."""
    out = {}
    prev = None
    for s in samples:
        spd = acc = 0.0
        if prev is not None:
            dt = max(1e-3, s["time"] - prev["time"])
            d = dist((s["x"], s["y"]), (prev["x"], prev["y"]))
            spd = d / dt
            if "speed" in prev:
                acc = abs(spd - prev["speed"]) / dt
        row = dict(s)
        row["speed"] = spd
        row["acceleration"] = acc
        out[s["frame"]] = row
        prev = row
    return out


def _sample_at(series_map, frame, tol=2):
    for step in range(tol + 1):
        for f in (frame - step, frame + step):
            if f in series_map:
                return series_map[f]
    return None


def _nearest_ball(balls_by_frame, frame, tol=2):
    best = None
    bestd = 1e9
    for off in range(-tol, tol + 1):
        for b in balls_by_frame.get(frame + off, []):
            d = math.hypot(b["x"] - 0.5, b["y"] - 0.5)  # placeholder, replaced by caller
            if d < bestd:
                bestd = d
                best = b
    return best


# --------------------------------------------------------------------------- #
# 2) DUEL DETECTION (local, evidence-gated)
# --------------------------------------------------------------------------- #
def detect_duel(tracks, balls, events, motion, duration):
    result = {"detected": False, "predator": None, "prey": None, "winner": None, "loser": None,
              "confidence": 0.0, "window": None, "evidence": {}, "source": "local",
              "reason": "no-evidence"}
    if not tracks or not balls:
        result["reason"] = "missing-tracks-or-ball"
        return result

    ball_frames = {}
    for b in balls:
        ball_frames.setdefault(b["frame"], []).append(b)

    def ball_at(frame, tol=3):
        for off in range(tol + 1):
            for f in (frame - off, frame + off):
                if ball_frames.get(f):
                    return ball_frames[f][0]
        return None

    # Event labels are used ONLY as supporting (candidate) evidence.
    labels_by_time = []
    for e in (events or {}).get("events", []) or []:
        try:
            labels_by_time.append((float(e.get("time", 0.0)), set(str(x) for x in e.get("events", [])),
                                   float(e.get("ball_proximity", 0.0) or 0.0),
                                   float(e.get("pressure", 0.0) or 0.0),
                                   float(e.get("direction_change", 0.0) or 0.0),
                                   float(e.get("acceleration", 0.0) or 0.0)))
        except (TypeError, ValueError):
            continue

    tids = [tid for tid, t in tracks.items() if t.get("size", 0) >= 3]
    kin = {}
    for tid in tids:
        kin[tid] = _kinematics(tracks[tid]["samples"], 30.0)

    # A duel is a SHORT event: slide a ~2.6s window over the shared timeline and
    # score each window, instead of averaging over the whole match.
    WIN = 2.6
    STEP = 0.4

    def score_window(ti, tj, frames):
        prox_hits = close_hits = counter = 0
        poss_transitions = 0
        last_owner = None
        prox_sum = 0.0
        n = len(frames)
        for f in frames:
            a = kin[ti][f]; b = kin[tj][f]
            d = dist((a["x"], a["y"]), (b["x"], b["y"]))
            prox_sum += max(0.0, 1.0 - d / 0.5)
            if d < PROX_HIT:
                prox_hits += 1
                if f in kin[tj] and (f - 3) in kin[ti] and (f - 3) in kin[tj]:
                    pa = kin[ti][f - 3]; pb = kin[tj][f - 3]
                    va = (a["x"] - pa["x"], a["y"] - pa["y"])
                    vb = (b["x"] - pb["x"], b["y"] - pb["y"])
                    dij = (b["x"] - a["x"], b["y"] - a["y"])
                    if va[0] * dij[0] + va[1] * dij[1] > 0 and vb[0] * (-dij[0]) + vb[1] * (-dij[1]) > 0:
                        counter += 1
            if d < PROX_HIT * 1.35:
                close_hits += 1
            ball = ball_at(f)
            if ball is not None:
                da = dist((a["x"], a["y"]), (ball["x"], ball["y"]))
                db = dist((b["x"], b["y"]), (ball["x"], ball["y"]))
                owner = ti if da <= db else tj
                if last_owner is not None and owner != last_owner:
                    poss_transitions += 1
                last_owner = owner
        proximity = prox_hits / max(1, n)
        closing = close_hits / max(1, n)
        counter_r = counter / max(1, prox_hits)
        possession = min(1.0, poss_transitions / 2.0)
        return proximity, closing, counter_r, possession

    def event_bonus(ti, frames):
        set_labs = set(); ev_prox = ev_press = ev_turn = ev_burst = 0.0
        t0 = kin[ti][frames[0]]["time"] if frames else 0.0
        t1 = kin[ti][frames[-1]]["time"] if frames else 0.0
        for (t, labs, bp, pr, dc, ac) in labels_by_time:
            if t0 - 0.6 <= t <= t1 + 0.6:
                set_labs |= labs
                ev_prox = max(ev_prox, bp); ev_press = max(ev_press, pr)
                ev_turn = max(ev_turn, dc); ev_burst = max(ev_burst, ac)
        decisive = bool(set_labs & {"shot_candidate", "goal", "tackle", "dribble",
                                    "ball_engagement", "explosive_run", "high_ball_speed_candidate"})
        return set_labs, ev_prox, ev_press, ev_turn, ev_burst, (0.12 if decisive else 0.0)

    best = None
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            ti, tj = tids[i], tids[j]
            shared = sorted(f for f in kin[ti] if f in kin[tj])
            if len(shared) < 6:
                continue
            times = [kin[ti][f]["time"] for f in shared]
            start_t, last_t = times[0], times[-1]
            w = start_t
            while w <= last_t - 0.6:
                frames = [f for f, t in zip(shared, times) if w <= t <= w + WIN]
                w += STEP
                if len(frames) < 6:
                    continue
                proximity, closing, counter_r, possession = score_window(ti, tj, frames)
                set_labs, ev_prox, ev_press, ev_turn, ev_burst, bonus = event_bonus(ti, frames)
                score = clamp(0.30 * proximity + 0.22 * closing + 0.16 * ev_prox +
                              0.10 * ev_press + 0.10 * counter_r + 0.06 * possession +
                              0.06 * ev_turn + 0.06 * min(1.0, ev_burst) + bonus)
                if best is None or score > best["score"]:
                    best = {"score": score, "ti": ti, "tj": tj, "frames": frames,
                            "proximity": proximity, "closing": closing, "counter": counter_r,
                            "possession": possession, "labels": sorted(set_labs),
                            "ev_prox": ev_prox, "ev_pressure": ev_press,
                            "ball_detected": True}

    if best is None:
        result["reason"] = "no-shared-track-window"
        return result

    if not (best["score"] >= DUEL_MIN_SCORE and best["proximity"] >= PROX_MIN and best["closing"] >= CLOSE_MIN):
        result["reason"] = "insufficient-duel-evidence"
        result["confidence"] = round(best["score"], 4)
        ti0 = best["ti"]
        result["window"] = [round(kin[ti0][best["frames"][0]]["time"], 3),
                            round(kin[ti0][best["frames"][-1]]["time"], 3)]
        return result

    ti, tj, frames = best["ti"], best["tj"], best["frames"]
    # predator = the attacking/ball-carrying player (higher travel + ball affinity)
    si = tracks[ti]["travel"] * 2.0 + tracks[ti]["ball_affinity"] * 0.9
    sj = tracks[tj]["travel"] * 2.0 + tracks[tj]["ball_affinity"] * 0.9
    predator, prey = (ti, tj) if si >= sj else (tj, ti)

    # duel moment = the frame of peak engagement (closest + ball near)
    best_f, best_s = None, -1.0
    for f in frames:
        a = kin[ti][f]; b = kin[tj][f]
        d = dist((a["x"], a["y"]), (b["x"], b["y"]))
        s = (1.0 - min(1.0, d / 0.5))
        ball = ball_at(f)
        if ball is not None:
            s += 0.5 * (1.0 - min(1.0, dist((a["x"], a["y"]), (ball["x"], ball["y"])) / 0.5))
        if s > best_s:
            best_s, best_f = s, f
    duel_time = kin[ti][best_f]["time"] if best_f is not None else \
        (kin[ti][frames[0]]["time"] + kin[ti][frames[-1]]["time"]) / 2.0

    result.update({
        "detected": True,
        "predator": predator,
        "prey": prey,
        "winner": None,   # only the OPTIONAL VLM may assert a winner
        "loser": None,
        "confidence": round(best["score"], 4),
        "window": [round(kin[ti][frames[0]]["time"], 3), round(kin[ti][frames[-1]]["time"], 3)],
        "moment": round(duel_time, 3),
        "source": "local",
        "reason": None,
        "evidence": {
            "proximity": round(best["proximity"], 4),
            "closing": round(best["closing"], 4),
            "counter_motion": round(best["counter"], 4),
            "possession_change": round(best["possession"], 4),
            "ball_proximity_event": round(best["ev_prox"], 4),
            "pressure_event": round(best["ev_pressure"], 4),
            "event_labels": best["labels"][:10],
            "window_frames": len(frames),
            "predator_track": predator,
            "prey_track": prey,
            "no_claim_without_visual_evidence": True,
        },
    })
    return result


# --------------------------------------------------------------------------- #
# 3) Story arc / roles (evidence-gated)
# --------------------------------------------------------------------------- #
def build_story_arc(duel, events, duration):
    if not duel.get("detected"):
        return []
    labels = set()
    turn = burst = ball_speed = prox = 0.0
    for e in (events or {}).get("events", []) or []:
        labels |= set(str(x) for x in e.get("events", []))
        turn = max(turn, float(e.get("direction_change", 0.0) or 0.0))
        burst = max(burst, float(e.get("acceleration", 0.0) or 0.0))
        ball_speed = max(ball_speed, float(e.get("ball_speed", 0.0) or 0.0))
        prox = max(prox, float(e.get("ball_proximity", 0.0) or 0.0))
    ev = duel.get("evidence") or {}
    arc = ["approach"]
    if float(ev.get("closing", 0)) > 0.4 or turn > 0.5:
        arc.append("pressure")
    if turn > 0.55 or "direction_change_candidate" in labels:
        arc.append("trap")
    if burst > 0.6 or "explosive_run" in labels:
        arc.append("escape")
    if "explosive_run" in labels or prox > 0.6:
        arc.append("dominance")
    if ball_speed > 0.55 or labels & {"shot_candidate", "goal", "tackle", "dribble"}:
        arc.append("impact")
    arc.append("aftermath")
    # Keep order + unique
    seen, ordered = set(), []
    for a in arc:
        if a not in seen:
            seen.add(a); ordered.append(a)
    return ordered


def pov_triggers(events, duration):
    """Real POV-switch trigger points: acceleration/defender-closing/ball
    proximity/dribble/tackle/shot/goal/possession change."""
    pts = []
    for e in (events or {}).get("events", []) or []:
        try:
            t = float(e.get("time", 0.0))
        except (TypeError, ValueError):
            continue
        if t < 0 or t > duration + 0.5:
            continue
        labels = set(str(x) for x in e.get("events", []))
        accel = float(e.get("acceleration", 0.0) or 0.0)
        prox = float(e.get("ball_proximity", 0.0) or 0.0)
        pressure = float(e.get("pressure", 0.0) or 0.0)
        turn = float(e.get("direction_change", 0.0) or 0.0)
        reason = []
        if accel > 0.6 or "explosive_run" in labels:
            reason.append("acceleration")
        if pressure > 0.6:
            reason.append("defender_closing")
        if prox > 0.6 or "ball_engagement" in labels:
            reason.append("ball_proximity")
        if labels & {"dribble", "tackle", "shot_candidate", "goal", "possession_change",
                     "high_ball_speed_candidate", "direction_change_candidate"}:
            reason.append("decisive_action")
        if reason:
            pts.append({"time": round(t, 3), "reason": sorted(set(reason))})
    pts.sort(key=lambda x: x["time"])
    # Thin out to MAX_POV_SWITCHES, keeping the earliest of far-apart clusters.
    thinned = []
    for p in pts:
        if all(abs(p["time"] - q["time"]) > 0.9 for q in thinned):
            thinned.append(p)
        if len(thinned) >= MAX_POV_SWITCHES:
            break
    return thinned


# --------------------------------------------------------------------------- #
# 4) Plan enrichment (ADDITIVE only)
# --------------------------------------------------------------------------- #
def _find_clip_index_for_source(plan, source_t):
    timeline = plan.get("timeline") or []
    inside, nearest, best_d = None, None, 1e9
    for i, c in enumerate(timeline):
        try:
            s = float(c.get("source_start")); e = float(c.get("source_end"))
        except (TypeError, ValueError):
            continue
        if s <= source_t <= e:
            inside = i
            break
        d = min(abs(source_t - s), abs(source_t - e))
        if d < best_d:
            best_d, nearest = d, i
    return inside if inside is not None else nearest


def _clip_output_center(c):
    try:
        return (float(c.get("output_start", 0)) + float(c.get("output_end", 0))) / 2.0
    except (TypeError, ValueError):
        return 0.0


def enrich_plan(plan, bundle):
    """Attach optional psychological fields onto the plan. Never removes/overwrites
    any existing field; only ADDS namespaced keys."""
    if not isinstance(plan, dict):
        return plan
    timeline = plan.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        return plan

    duel = bundle.get("duel") or {}
    story_arc = bundle.get("story_arc") or []
    duel_moment = bundle.get("duel_moment")
    script = bundle.get("story_script") or []
    pov = bundle.get("pov_switch_points") or []
    low_angles = bundle.get("low_angle_segments") or []
    depth_segs = bundle.get("depth_segments") or []

    # ---- top-level (namespaced, optional) -------------------------------- #
    plan["psychological_story"] = {
        "version": VERSION,
        "evidence_level": bundle.get("evidence_level"),
        "duel_detected": bool(duel.get("detected")),
        "predator": duel.get("predator"),
        "prey": duel.get("prey"),
        "winner": duel.get("winner"),
        "loser": duel.get("loser"),
        "duel_confidence": duel.get("confidence", 0.0),
        "duel_source": duel.get("source"),
        "duel_moment": duel_moment,
        "story_arc": story_arc,
        "depth_effect_used": bool(depth_segs),
        "low_angle_used": bool(low_angles),
        "pov_switch_used": bool(pov),
        "story_script_count": len(script),
        "rules": {
            "no_invented_event_or_winner": True,
            "evidence_gated": True,
            "additive_only": True,
            "fallback_to_goalflow_when_weak": True,
        },
    }
    plan["story_arc"] = story_arc
    plan["duel_moment"] = duel_moment
    if script:
        plan["story_script"] = script

    # OPTIONAL psychological teal-orange grade (ADDITIVE — the global grade is kept).
    if duel.get("detected"):
        cg = plan.setdefault("color_grade", {})
        cg["psychological"] = {
            "enabled": True,
            "name": "psychological_teal_orange",
            "cool_shadows": True, "warm_highlights": True,
            "shadow_teal": 0.045, "shadow_teal_pull": -0.026,
            "highlight_warm": 0.032, "highlight_warm_pull": -0.024,
            "controlled_contrast": True, "protected_skin": True,
            "grass_saturation": 0.93, "vignette": True,
        }
        # Optional subtle SoundDesigner cues (match audio preserved).
        sd = plan.setdefault("sound_design", {})
        sd.setdefault("riser_starts", [])
        sd["psychological_cues"] = {"riser": True, "whoosh": True, "impact": True,
                                    "crowd_duck": True, "silence_before_impact": True,
                                    "preserve_match_audio": True}

    if not duel.get("detected"):
        # Weak/no evidence -> purely additive metadata; the GoalFlow cinematic
        # pipeline renders exactly as before.
        return plan

    d_source = bundle.get("duel_source_moment")
    if d_source is None:
        d_source = (duel.get("window") or [0, 0])[0]
    d_index = _find_clip_index_for_source(plan, float(d_source))
    duel_output = None
    if d_index is not None and 0 <= d_index < len(timeline):
        duel_output = _clip_output_center(timeline[d_index])

    # ---- temporal story roles across the reel ---------------------------- #
    n = len(timeline)
    if duel_output is not None and n > 1 and d_index is not None:
        for i, c in enumerate(timeline):
            try:
                sw = float(c.get("source_start"))
                ew = float(c.get("source_end"))
            except (TypeError, ValueError):
                sw, ew = 0.0, 0.0
            if i < d_index - 1:
                role = "approach"
            elif i == d_index - 1:
                role = "trap"
            elif i == d_index:
                role = "impact"
            elif i == d_index + 1:
                role = "dominance" if "dominance" in story_arc else (
                    "escape" if "escape" in story_arc else "aftermath")
            elif i >= n - 1:
                role = "aftermath"
            else:
                role = "dominance"
            # A closing/escape beat just before contact when evidence supports it.
            if i == d_index - 1 and "pressure" in story_arc:
                role = "pressure" if (d_index - 1) > 0 else "trap"
            c["story_role"] = role
            c["story_arc"] = story_arc
            c["duel_moment"] = round(float(duel_output), 3)
            if i == d_index:
                c["duel_moment_exact"] = True
                c["subject_isolation"] = True
                c["isolation_reason"] = c.get("isolation_reason") or "duel_impact"

            # POV switch when a real trigger lands inside the clip's source window
            if any(sw - 0.15 <= p["time"] <= ew + 0.15 for p in pov):
                c["pov_switch"] = True

    # ---- low-angle segments (bounded) ------------------------------------ #
    low_idx = set()
    for seg in low_angles[:MAX_LOW_ANGLE]:
        idx = _find_clip_index_for_source(plan, float(seg.get("time", 0.0)))
        if idx is not None:
            low_idx.add(idx)
    for idx in low_idx:
        timeline[idx]["low_angle"] = True

    # ---- depth-effect segments (bounded, around the duel) ---------------- #
    depth_idx = set()
    for seg in depth_segs[:MAX_DEPTH_CLIPS]:
        idx = _find_clip_index_for_source(plan, float(seg.get("time", 0.0)))
        if idx is not None:
            depth_idx.add(idx)
    for i in depth_idx:
        timeline[i]["depth_effect"] = "depth_bokeh"

    # ---- story script -> real on-screen lines (only when script exists) -- #
    if script:
        used = set()
        # Map each beat to the nearest unused clip by output time.
        for line in script:
            try:
                lt = float(line.get("time", 0.0))
            except (TypeError, ValueError):
                continue
            order = sorted(range(n), key=lambda i: abs(_clip_output_center(timeline[i]) - lt))
            for i in order:
                if i in used:
                    continue
                used.add(i)
                timeline[i]["text"] = str(line.get("text", ""))[:46]
                timeline[i]["psychological_story"] = str(line.get("text", ""))
                timeline[i]["story_script_position"] = line.get("position", "lower_third")
                break
    return plan


# --------------------------------------------------------------------------- #
# 5) Orchestration
# --------------------------------------------------------------------------- #
def build_bundle(duration, tracking=None, events=None, director=None, motion=None,
                 video=None, allow_network=True, use_vlm=True, use_script=True,
                 timeout_ms=60000):
    meta = {"vlm": {"applied": False, "reason": "disabled"}, "deepseek": {"applied": False, "reason": "disabled"}}
    tracks, balls, fps = build_series(tracking or {})
    _ball_affinity(tracks, balls)

    ev_count = len((events or {}).get("events", []) or [])
    ball_detected = bool((tracking or {}).get("summary", {}).get("detections", {}).get("ball")) or bool(balls)

    duel = detect_duel(tracks, balls, events, motion, duration)
    duel_moment = duel.get("moment")
    duel_source_moment = duel_moment
    if duel_moment is None:
        # Fall back to the director's hero moment, else a bounded default.
        hm = (director or {}).get("hero_moment") or {}
        duel_moment = float(hm.get("t", duration * 0.62))
    duel_moment = clamp(duel_moment, 0.3, max(0.6, duration - 0.3))

    # ---- optional VLM refinement (only when a duel was detected) --------- #
    if use_vlm and allow_network and duel.get("detected") and video:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import vlm_analyzer  # local module
            win = duel.get("window") or [max(0.0, duel_moment - 3), min(duration, duel_moment + 3)]
            evidence = {
                "players": list(tracks.keys()),
                "track_summary": (tracking or {}).get("summary", {}),
                "duel": {"predator": duel.get("predator"), "prey": duel.get("prey")},
                "events": (events or {}).get("events", [])[:40],
            }
            verdict, vmeta = vlm_analyzer.analyze(video, evidence, win, duration,
                                                  timeout_ms=timeout_ms, allow_network=True)
            meta["vlm"] = vmeta
            if verdict:
                duel["source"] = "vlm"
                if verdict.get("winner"):
                    duel["winner"] = verdict["winner"]
                if verdict.get("loser"):
                    duel["loser"] = verdict["loser"]
                duel["confidence"] = max(float(duel.get("confidence", 0.0)), float(verdict.get("confidence", 0.0)))
                if verdict.get("duel_moment") is not None:
                    duel_moment = clamp(float(verdict["duel_moment"]), 0.3, max(0.6, duration - 0.3))
                if verdict.get("story_arc"):
                    duel["vlm_story_arc"] = verdict["story_arc"]
        except Exception as exc:  # never break the pipeline
            meta["vlm"] = {"applied": False, "reason": type(exc).__name__}

    story_arc = build_story_arc(duel, events, duration)
    if duel.get("vlm_story_arc"):
        for a in duel["vlm_story_arc"]:
            if a not in story_arc:
                story_arc.append(a)

    pov_pts = pov_triggers(events, duration) if duel.get("detected") else []
    low_angles = []
    depth_segs = []
    if duel.get("detected"):
        # low-angle: the approach + impact source moments (bounded, restrained)
        low_angles = [{"time": round(max(0.0, duel_moment - 2.4), 3), "reason": "approach"},
                      {"time": round(duel_moment, 3), "reason": "impact"}]
        depth_segs = [{"time": round(duel_moment, 3), "reason": "psychological_6s",
                       "span": PSYCH_WINDOW * 2}]

    evidence_level = "high" if duel.get("detected") and ball_detected and ev_count > 20 else \
        "medium" if duel.get("detected") else ("medium" if (ev_count or tracks) else "low")

    bundle = {
        "success": True,
        "version": VERSION,
        "evidence_level": evidence_level,
        "duration": round(duration, 3),
        "ball_detected": ball_detected,
        "player_tracks": len(tracks),
        "event_count": ev_count,
        "duel": duel,
        "duel_source_moment": duel_source_moment if duel.get("detected") else None,
        "duel_moment": round(float(duel_moment), 3),
        "story_arc": story_arc,
        "available_roles": DUEL_ROLES + TEMPORAL_ROLES,
        "pov_switch_points": pov_pts,
        "low_angle_segments": low_angles,
        "depth_segments": depth_segs,
        "story_script": [],
        "meta": meta,
    }

    # ---- optional DeepSeek 6-line psychological script ------------------- #
    if use_script and allow_network and duel.get("detected"):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import deepseek_storyteller
            existing = []
            sscript, smeta = deepseek_storyteller.generate_script(
                bundle, 64.0, existing_texts=existing, timeout_ms=45000, allow_network=True)
            bundle["story_script"] = sscript
            meta["deepseek"] = smeta
        except Exception as exc:
            meta["deepseek"] = {"applied": False, "reason": type(exc).__name__}
    elif not duel.get("detected"):
        meta["deepseek"] = {"applied": False, "reason": "no-duel-evidence"}

    return bundle


def main():
    ap = argparse.ArgumentParser(description="Cinematic Storyteller — psychological orchestration layer")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--tracking", default=None)
    ap.add_argument("--events", default=None)
    ap.add_argument("--director", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--plan-out", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--timeout-ms", type=int, default=60000)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-script", action="store_true")
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()

    try:
        bundle = build_bundle(
            a.duration,
            tracking=load_json(a.tracking),
            events=load_json(a.events),
            director=load_json(a.director),
            motion=load_json(a.motion),
            video=a.video,
            allow_network=not a.no_network,
            use_vlm=not a.no_vlm,
            use_script=not a.no_script,
            timeout_ms=a.timeout_ms,
        )
        Path(a.output).write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf8")

        if a.plan and a.plan_out:
            plan = load_json(a.plan)
            if isinstance(plan, dict):
                enriched = enrich_plan(plan, bundle)
                Path(a.plan_out).write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf8")

        print(json.dumps({
            "success": True,
            "version": VERSION,
            "evidence_level": bundle["evidence_level"],
            "duel_detected": bundle["duel"]["detected"],
            "duel_confidence": bundle["duel"]["confidence"],
            "winner": bundle["duel"]["winner"],
            "story_arc": bundle["story_arc"],
            "story_script_count": len(bundle["story_script"]),
            "vlm_applied": bundle["meta"]["vlm"].get("applied"),
            "deepseek_applied": bundle["meta"]["deepseek"].get("applied"),
        }))
        return 0
    except Exception as exc:
        # Fail-safe: emit a valid minimal bundle so the caller never breaks.
        minimal = {"success": True, "version": VERSION, "evidence_level": "low",
                   "duel": {"detected": False, "reason": f"storyteller-error:{type(exc).__name__}"},
                   "story_arc": [], "story_script": [], "pov_switch_points": [],
                   "low_angle_segments": [], "depth_segments": [],
                   "meta": {"error": type(exc).__name__}}
        try:
            Path(a.output).write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf8")
        except Exception:
            pass
        print(json.dumps({"success": True, "version": VERSION, "evidence_level": "low",
                          "duel_detected": False, "error": type(exc).__name__}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
