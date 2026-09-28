#!/usr/bin/env python3
"""Reference Style Analyzer (FootballCinematicAI-v10) — REFERENCE style extraction.

Measures the REAL editorial/visual DNA of a reference video and turns it into a
`ReferenceStyleProfile`: a set of *style parameters* (not timestamps, not a
shot-by-shot sequence) that the Cinematic Director can apply adaptively to any
new match footage.

What is measured (all from real decoded frames + real audio, no LLM):
  * shot duration distribution (mean / median / p10 / p90 / min / max)
  * cut density (cuts per second) and the real cut list
  * camera movement classes (static / follow / pan / push_in / pull_out)
  * zoom intensity (global similarity-scale deviation, median + p90)
  * speed handling (duplicate-frame ratio => real slow-motion usage)
  * subject framing mix (close-up / medium / wide / action / crowd proxies)
  * subject isolation usage (foreground blob area proxy)
  * player-tracking density (how often a tracked blob exists at all)
  * text frequency (caption OCR-free proxy: high-contrast text band in the
    lower third), caption height ratio and caption zones
  * transition mix (hard_cut / dissolve / flash / fade) with real detection
  * colour characteristics (mean BGR, contrast, saturation, shadow/highlight
    colour temperature, skin-tone ratio, neon-grass ratio)
  * audio: impact count, riser count, silence ratio and how many cuts land on
    an audio impact (cut/impact sync)
  * hero-moment structure (which window carries the strongest subject+face
    signal) and the ending structure

Usage:
  python3 yolo/reference_style_analyzer.py --input ref.mp4 --output profile.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

SMALL_W, SMALL_H = 216, 384


def _percentile(values, q):
    return round(float(np.percentile(values, q)), 4) if len(values) else 0.0


def _mean(values, nd=4):
    return round(float(np.mean(values)), nd) if len(values) else 0.0


# ---------------------------------------------------------------------------
# Pass 1 — decode once, collect every per-frame signal (cheap, bounded memory)
# ---------------------------------------------------------------------------
def collect(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open reference video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=26, detectShadows=False)

    small = []      # grey thumbnails (cut detection + global flow)
    hists = []      # 64-bin histograms
    bright = []     # mean luminance (transition classification)
    skin_r = []     # skin-tone coverage (close-up / subject proxy)
    blob_r = []     # foreground blob height ratio (subject isolation proxy)
    text_r = []     # caption band present on this frame
    text_h = []     # caption glyph height / frame height
    col = []        # colour samples (every 3rd frame, full res)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        s = cv2.resize(frame, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        small.append(g)
        h = cv2.calcHist([g], [0], None, [64], [0, 256])
        cv2.normalize(h, h)
        hists.append(h)
        bright.append(float(g.mean()))

        # Subject proxy: skin-tone coverage on the thumbnail (a real close-up or
        # a large player body produces a large, stable skin ratio; a wide pitch
        # shot produces almost none).
        if idx % 2 == 0:
            yc = cv2.cvtColor(s, cv2.COLOR_BGR2YCrCb)
            sk = (yc[:, :, 1] > 135) & (yc[:, :, 1] < 180) & (yc[:, :, 2] > 85) & (yc[:, :, 2] < 135)
            skin_r.append(float(sk.mean()))
        else:
            skin_r.append(float("nan"))

        # Subject isolation proxy: largest foreground blob height ratio.
        m = bg.apply(s, learningRate=0.02)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        br = 0.0
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > SMALL_W * SMALL_H * 0.004:
                br = cv2.boundingRect(c)[3] / SMALL_H
        blob_r.append(br)

        # Caption proxy: a bright, wide, short horizontal glyph band in the
        # lower third (the reference reel writes small white uppercase captions
        # near the bottom — never an opaque subtitle box).
        y0 = int(SMALL_H * 0.52)
        roi = g[y0:SMALL_H, :]
        tb = cv2.threshold(roi, 197, 255, cv2.THRESH_BINARY)[1]
        tb = cv2.morphologyEx(tb, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
        cs, _ = cv2.findContours(tb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cs:
            x, y, w, hh = cv2.boundingRect(c)
            if 6 < hh < 26 and 26 < w < SMALL_W * 0.92 and w / max(hh, 1) > 2.4 and cv2.contourArea(c) > 44:
                boxes.append(hh)
        text_r.append(1 if boxes else 0)
        text_h.append(float(np.median(boxes)) / SMALL_H if boxes else 0.0)

        if idx % 3 == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            Y = ycc[:, :, 0].astype(np.float32)
            p10, p50, p90 = np.percentile(Y, [10, 50, 90])
            m_low = Y <= p10
            m_high = Y >= p90
            b = frame[:, :, 0].astype(np.float32)
            gc = frame[:, :, 1].astype(np.float32)
            r = frame[:, :, 2].astype(np.float32)
            sat = hsv[:, :, 1].astype(np.float32)
            hue = hsv[:, :, 0]
            skin = (ycc[:, :, 1] > 135) & (ycc[:, :, 1] < 180) & (ycc[:, :, 2] > 85) & (ycc[:, :, 2] < 135)
            grass = (hue >= 35) & (hue <= 85)
            neon = grass & (sat > 120)
            col.append({
                "t": round(idx / fps, 3),
                "B": float(b.mean()), "G": float(gc.mean()), "R": float(r.mean()),
                "p10": float(p10), "p50": float(p50), "p90": float(p90),
                "contrast": float(p90 - p10), "sat": float(sat.mean()),
                "shadow_BR": float((b[m_low] - r[m_low]).mean()) if m_low.sum() else 0.0,
                "high_BR": float((b[m_high] - r[m_high]).mean()) if m_high.sum() else 0.0,
                "skin": float(skin.mean()),
                "neon": float(neon.mean()),
                "grass": float(grass.mean()),
                "sat_skin": float(sat[skin].mean()) if skin.sum() else 0.0,
            })
        idx += 1

    cap.release()
    return dict(fps=fps, width=width, height=height, frames=len(small), small=small,
                hists=hists, bright=bright, skin_r=skin_r, blob_r=blob_r,
                text_r=text_r, text_h=text_h, col=col)


# ---------------------------------------------------------------------------
# Pass 2 — cut detection + motion/zoom/tracking, per-shot aggregation
# ---------------------------------------------------------------------------
def analyse(data):
    fps = data["fps"]
    small = data["small"]
    hists = data["hists"]
    bright = data["bright"]
    n = data["frames"]
    dur = n / fps

    pix = np.zeros(n)
    score = np.zeros(n)
    for i in range(1, n):
        corr = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL)
        diff = float(cv2.absdiff(small[i - 1], small[i]).mean()) / 255.0
        pix[i] = diff
        score[i] = (1.0 - corr) + 1.4 * diff

    body = score[1:]
    # ROBUST cut threshold.
    #
    # A dense montage contains FEW but very large cut spikes. A mean + k*sigma
    # threshold is inflated by those very spikes and then silently misses the
    # cuts (measured on a real render: only 3 of 27 cuts were found, which made
    # the QC comparison meaningless). Median + MAD is outlier-resistant, so the
    # SAME detector measures a reference reel and a produced master correctly.
    med = float(np.median(body))
    mad = float(np.median(np.abs(body - med))) or 1e-6
    sigma = 1.4826 * mad
    thr = max(0.045, med + 3.0 * sigma)
    local_max = [i for i in range(1, n - 1) if score[i] >= score[i - 1] and score[i] >= score[i + 1]]
    candidates = [i for i in local_max if score[i] > thr]
    # Floor: any real edit has at least ~1 cut every 4 seconds. If the robust
    # threshold is too strict, relax it rather than reporting a single 21s shot.
    floor = max(2, int(dur * 0.25))
    if len(candidates) < floor:
        relaxed = max(0.035, med + 1.5 * sigma)
        candidates = [i for i in local_max if score[i] > relaxed]
    cuts = []
    for i in sorted(candidates, key=lambda x: -score[x]):
        if all(abs(i - j) > int(fps * 0.33) for j in cuts):
            cuts.append(i)
        if len(cuts) >= min(60, int(dur * 0.75)):
            break
    cuts.sort()

    bounds = [0] + cuts + [n]
    shots = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        shots.append({"i": k, "a": round(a / fps, 3), "b": round(b / fps, 3),
                      "dur": round((b - a) / fps, 3), "f0": a,
                      "cut_score": round(float(score[a]), 3) if a > 0 else 0.0})

    # Motion + camera-motion estimation (global similarity transform between
    # small thumbnails; the scale component is the real zoom signal).
    prev = small[0]
    prev_grey = small[0]
    aff = {}
    flow = np.zeros(n)
    for i in range(1, n):
        f = cv2.calcOpticalFlowFarneback(prev, small[i], None, 0.5, 3, 15, 3, 5, 1.1, 0)
        flow[i] = float(np.median(np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)))
        if i % 2 == 0:
            p = cv2.goodFeaturesToTrack(prev_grey, maxCorners=140, qualityLevel=0.01, minDistance=6)
            if p is not None and len(p) >= 8:
                nxt, _, _ = cv2.calcOpticalFlowPyrLK(prev_grey, small[i], p, None, winSize=(19, 19), maxLevel=3)
                if nxt is not None:
                    m, _ = cv2.estimateAffinePartial2D(p.reshape(-1, 2), nxt.reshape(-1, 2), method=cv2.RANSAC)
                    if m is not None:
                        aff[i] = (math.hypot(float(m[0, 0]), float(m[1, 0])), float(m[0, 2]), float(m[1, 2]))
            prev_grey = small[i]
        prev = small[i]

    def translate_of(i):
        if (i - 2) in aff and i in aff:
            a, b = aff[i - 2], aff[i]
            return math.hypot(b[1] - a[1], b[2] - a[2])
        return 0.0

    for s in shots:
        ia, ib = max(s["f0"], 1), max(int(s["b"] * fps), s["f0"] + 1)
        seg = flow[ia:ib]
        scales = [aff[i][0] for i in aff if ia <= i < ib]
        zoom_dev = [abs(z - 1.0) for z in scales]
        pans = [translate_of(i) for i in range(ia, ib, 2)]
        s["flow"] = _mean(seg)
        s["flow_p90"] = _percentile(seg, 90)
        s["zoom_dev"] = round(float(np.median(zoom_dev)) if zoom_dev else 0.0, 5)
        s["zoom_dev_p90"] = round(float(np.percentile(zoom_dev, 90)) if zoom_dev else 0.0, 5)
        s["pan_px"] = round(float(np.median(pans)) if pans else 0.0, 2)
        s["dup_ratio"] = round(float(np.mean([1.0 if pix[i] < 0.003 else 0.0 for i in range(ia, ib)])), 3)
        s["subject_ratio"] = round(float(np.nanmax(data["skin_r"][ia:ib])) if ib > ia else 0.0, 4)
        s["isolation_ratio"] = round(float(np.percentile(data["blob_r"][ia:ib], 80)) if ib > ia else 0.0, 4)
        s["text_frame_ratio"] = round(float(np.mean(data["text_r"][ia:ib])) if ib > ia else 0.0, 3)
        s["text_height_ratio"] = round(float(np.median([t for t in data["text_h"][ia:ib] if t > 0] or [0.0])), 4)
        s["bright"] = round(float(np.mean(bright[ia:ib])) if ib > ia else 0.0, 1)

        if s["dur"] > 1.2:
            half = max(1, len(seg) // 2)
            s["ramp_ratio"] = round(float(np.median(seg[half:]) / max(np.median(seg[:half]), 1e-6)), 3)
        else:
            s["ramp_ratio"] = None
        # Real slow motion = held shot + high duplicate-frame ratio + low flow.
        s["slow_motion"] = bool(s["dur"] > 1.0 and s["dup_ratio"] >= 0.30 and s["flow"] < 0.5)

        # Framing class: subject signal first, then isolation, then width.
        if s["subject_ratio"] > 0.05:
            shot_type = "close_up"
        elif s["isolation_ratio"] >= 0.42:
            shot_type = "action" if s["flow"] > 0.35 else "medium"
        elif s["isolation_ratio"] >= 0.20:
            shot_type = "medium"
        else:
            shot_type = "wide"
        if shot_type == "wide" and s["flow"] < 0.12 and s["zoom_dev"] < 0.004:
            shot_type = "crowd" if s["bright"] > 60 else "wide"
        s["shot_type"] = shot_type

        # Camera movement class.
        mean_scale = float(np.median(scales)) if scales else 1.0
        if s["zoom_dev"] > 0.012 and mean_scale > 1.0:
            cam = "push_in"
        elif s["zoom_dev"] > 0.012:
            cam = "pull_out"
        elif s["pan_px"] > 60:
            cam = "pan"
        elif s["flow_p90"] > 2.2 and s["pan_px"] < 25:
            cam = "handheld_whip"
        elif s["flow"] < 0.05 and s["zoom_dev"] < 0.002:
            cam = "static"
        else:
            cam = "follow"
        s["camera"] = cam

        cc = [c for c in data["col"] if s["a"] <= c["t"] < s["b"]]
        for key in ("sat", "contrast", "skin", "neon", "grass", "shadow_BR", "high_BR", "p10", "p50", "p90", "sat_skin"):
            if cc:
                s[key] = round(float(np.mean([c[key] for c in cc])), 3)

    # Transition classification per cut.
    transitions = []
    for ci in cuts:
        pre = bright[max(0, ci - 4):ci]
        post = bright[ci:min(n, ci + 4)]
        pr = float(np.mean(pre)) if pre else 0.0
        po = float(np.mean(post)) if post else 0.0
        if pr < 26:
            kind = "fade_from_black"
        elif po < 26:
            kind = "fade_to_black"
        elif max(po, pr) > 1.75 * max(min(pr, po), 1.0):
            kind = "flash"
        elif sum(1 for i in range(max(1, ci - 3), min(n - 1, ci + 4)) if 0.5 * score[ci] < score[i] < score[ci]) >= 2:
            kind = "dissolve"
        else:
            kind = "hard_cut"
        transitions.append({"frame": ci, "t": round(ci / fps, 3), "kind": kind})

    # Caption zones (merged runs of frames carrying a caption).
    zones = []
    for s in shots:
        if sum(data["text_r"][s["f0"]:int(s["b"] * fps)]) >= 3:
            if zones and s["a"] - zones[-1][1] < 1.2:
                zones[-1] = (zones[-1][0], s["b"])
            else:
                zones.append((s["a"], s["b"]))

    # Audio: impacts, risers, silence, cut/impact sync.
    impacts, risers, silence_ratio, rms_mean = _audio(video_path_from(data), fps)
    cut_times = [round(c / fps, 2) for c in cuts]
    on_impact = len([c for c in cut_times if any(abs(c - i) < 0.4 for i in impacts)])

    return dict(fps=fps, dur=dur, shots=shots, cuts=cuts, transitions=transitions,
                zones=zones, impacts=impacts, risers=risers, silence_ratio=silence_ratio,
                rms_mean=rms_mean, on_impact=on_impact, pix=pix, flow=flow,
                thr=thr, cut_times=cut_times)


def video_path_from(data):
    return data.get("_path", "")


def _audio(video_path, fps):
    if not video_path or not os.path.exists(video_path):
        return [], [], 0.0, 0.0
    tmp = tempfile.NamedTemporaryFile(suffix=".pcm", delete=False)
    tmp.close()
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-ac", "1", "-ar", "16000",
                        "-f", "s16le", tmp.name], check=True, capture_output=True)
        raw = np.fromfile(tmp.name, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return [], [], 0.0, 0.0
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    hop = 160  # 10 ms
    nf = len(raw) // hop
    if nf < 4:
        return [], [], 0.0, 0.0
    rms = np.sqrt(np.mean(raw[:nf * hop].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    impacts = []
    for i in range(3, nf - 3):
        if rms[i] > rms[i - 1] and rms[i] >= rms[i + 1] and rms[i] > rms.mean() + 1.7 * rms.std():
            if not impacts or (i - impacts[-1]) > 18:
                impacts.append(i)
            elif rms[i] > rms[impacts[-1]]:
                impacts[-1] = i
    risers = []
    for ti in impacts:
        for look in (50, 70, 90):
            if ti - look - 4 < 0:
                continue
            seg = rms[ti - look - 4:ti - 4]
            if len(seg) > 10:
                slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
                if slope > 0 and seg[-1] > 2.0 * seg[0] and seg[0] < rms.mean() * 1.2:
                    risers.append(ti)
                    break
    silence = float(np.mean(rms < max(0.015, rms.mean() * 0.25)))
    return [round(i * 0.01, 2) for i in impacts], [round(i * 0.01, 2) for i in risers], round(silence, 3), round(float(rms.mean()), 5)


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------
def build_profile(data, res):
    fps = data["fps"]
    shots = res["shots"]
    dur = res["dur"]
    durs = [s["dur"] for s in shots]
    zones = res["zones"]
    total = len(shots) if shots else 1

    def weights(key):
        out = {}
        for s in shots:
            out[s[key]] = out.get(s[key], 0) + 1
        return {k: round(v / total, 3) for k, v in sorted(out.items(), key=lambda x: -x[1])}

    trans_w = {}
    for t in res["transitions"]:
        trans_w[t["kind"]] = trans_w.get(t["kind"], 0) + 1
    trans_w = {k: round(v / max(len(res["transitions"]), 1), 3) for k, v in sorted(trans_w.items(), key=lambda x: -x[1])}

    col = data["col"]
    hero_pool = sorted(shots, key=lambda s: (s["isolation_ratio"] * 2.0 + s["flow"]) * max(s["subject_ratio"] * 3 + 0.5, 0.5), reverse=True)
    ending = [s for s in shots if s["a"] >= dur - 6]

    profile = {
        "version": "v11-reference-style-1.0",
        "source": os.path.basename(data.get("_path", "reference.mp4")),
        "resolution": f"{data['width']}x{data['height']}",
        "aspect": "9:16" if data["height"] > data["width"] else "16:9",
        "fps": round(fps, 3),
        "duration": round(dur, 3),

        # -- pacing / cutting ------------------------------------------------
        "shot_count": len(shots),
        "avg_shot_duration": _mean(durs, 3),
        "median_shot_duration": _percentile(durs, 50),
        "shot_duration_p10": _percentile(durs, 10),
        "shot_duration_p90": _percentile(durs, 90),
        "shot_duration_min": round(min(durs), 3) if durs else 0.0,
        "shot_duration_max": round(max(durs), 3) if durs else 0.0,
        "shot_duration_std": round(float(np.std(durs)), 3) if durs else 0.0,
        "cut_density": round(len(shots) / dur, 3) if dur else 0.0,
        "cut_threshold": round(res["thr"], 4),

        # -- framing / camera -------------------------------------------------
        "shot_type_weights": weights("shot_type"),
        "camera_weights": weights("camera"),
        "zoom_intensity": round(float(np.median([s["zoom_dev"] for s in shots])) if shots else 0.0, 5),
        "zoom_intensity_p90": round(float(np.percentile([s["zoom_dev"] for s in shots], 90)) if shots else 0.0, 5),
        "pan_px_median": round(float(np.median([s["pan_px"] for s in shots])) if shots else 0.0, 2),
        "global_flow_median": round(float(np.median(res["flow"][1:])), 4),
        "global_flow_p90": _percentile(res["flow"][1:], 90),

        # -- speed handling --------------------------------------------------
        "slow_motion_shot_ratio": round(sum(1 for s in shots if s["slow_motion"]) / total, 3),
        "ramp_shot_ratio": round(sum(1 for s in shots if s["ramp_ratio"] not in (None,) and s["ramp_ratio"] < 0.7) / total, 3),
        "duplicate_frame_ratio": round(float(np.mean([1.0 if res["pix"][i] < 0.003 else 0.0 for i in range(1, data["frames"])])), 4),

        # -- subject focus (player is the subject, not the pitch) -------------
        "subject_shot_ratio": round(sum(1 for s in shots if s["subject_ratio"] > 0.05) / total, 3),
        "isolation_shot_ratio": round(sum(1 for s in shots if s["isolation_ratio"] >= 0.42) / total, 3),
        "subject_isolation_median": round(float(np.median([s["isolation_ratio"] for s in shots])) if shots else 0.0, 4),
        "tracking_density": round(sum(1 for s in shots if s["isolation_ratio"] > 0.10) / total, 3),

        # -- typography ------------------------------------------------------
        "text_zone_count": len(zones),
        "text_zones": [[round(a, 2), round(b, 2)] for a, b in zones],
        "text_per_shot": round(len(zones) / total, 3),
        "text_frame_ratio": round(float(np.mean(data["text_r"])), 4),
        "text_height_ratio": round(float(np.median([t for t in data["text_h"] if t > 0] or [0.0])), 4),

        # -- transitions -----------------------------------------------------
        "transition_weights": trans_w,
        "hard_cut_ratio": trans_w.get("hard_cut", 0.0),

        # -- colour ----------------------------------------------------------
        "color": {
            "mean_B": _mean([c["B"] for c in col], 2),
            "mean_G": _mean([c["G"] for c in col], 2),
            "mean_R": _mean([c["R"] for c in col], 2),
            "contrast": _mean([c["contrast"] for c in col], 2),
            "saturation": _mean([c["sat"] for c in col], 2),
            "shadow_B_minus_R": _mean([c["shadow_BR"] for c in col], 2),
            "highlight_B_minus_R": _mean([c["high_BR"] for c in col], 2),
            "p10": _mean([c["p10"] for c in col], 1),
            "p50": _mean([c["p50"] for c in col], 1),
            "p90": _mean([c["p90"] for c in col], 1),
            "skin_ratio": _mean([c["skin"] for c in col], 4),
            "skin_saturation": _mean([c["sat_skin"] for c in col], 1),
            "neon_grass_ratio": _mean([c["neon"] for c in col], 4),
            "grass_ratio": _mean([c["grass"] for c in col], 4),
        },

        # -- sound -----------------------------------------------------------
        "audio": {
            "rms_mean": res["rms_mean"],
            "impact_count": len(res["impacts"]),
            "riser_count": len(res["risers"]),
            "silence_ratio": res["silence_ratio"],
            "impact_times": res["impacts"][:80],
            "riser_times": res["risers"][:40],
            "cuts_on_impact": res["on_impact"],
            "cut_impact_sync_ratio": round(res["on_impact"] / max(len(res["cut_times"]), 1), 3),
        },

        # -- hero / ending structure -----------------------------------------
        "hero_structure": [
            {"a": s["a"], "b": s["b"], "shot_type": s["shot_type"], "isolation": s["isolation_ratio"],
             "subject": s["subject_ratio"], "flow": s["flow"], "slow_motion": s["slow_motion"],
             "camera": s["camera"]} for s in hero_pool[:3]],
        "ending_structure": [
            {"a": s["a"], "b": s["b"], "shot_type": s["shot_type"], "dup_ratio": s["dup_ratio"],
             "flow": s["flow"], "bright": s["bright"]} for s in ending],

        # -- full evidence (kept for QC / diffing, never used as a sequence) ---
        "shots": shots,
        "cuts": res["cut_times"],
    }
    return profile


def analyse_video(path: str) -> dict:
    data = collect(path)
    data["_path"] = path
    res = analyse(data)
    return build_profile(data, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    try:
        profile = analyse_video(a.input)
        Path(a.output).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({
            "success": True,
            "profile": a.output,
            "shots": profile["shot_count"],
            "avg_shot_duration": profile["avg_shot_duration"],
            "cut_density": profile["cut_density"],
            "zoom_intensity": profile["zoom_intensity"],
            "subject_shot_ratio": profile["subject_shot_ratio"],
        }))
        return 0
    except Exception as e:  # never crash the caller: emit strict JSON
        print(json.dumps({"success": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
