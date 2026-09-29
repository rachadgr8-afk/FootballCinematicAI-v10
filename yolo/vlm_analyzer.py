#!/usr/bin/env python3
"""OPTIONAL Vision-Language duel analyzer (Qwen2-VL / InternVL2).

A thin, FAIL-SAFE bridge that asks an OpenAI-compatible VLM to interpret a
~6 second football duel clip and return a STRUCTURED verdict:

    {"duel_moment": <s>, "winner": <player_id|null>, "loser": <player_id|null>,
     "story_arc": ["approach", "pressure", ...], "confidence": <0..1>}

Design rules (same philosophy as the project's other optional layers):

* DEFAULT OFF / inert unless ``VLM_API_KEY`` is set. No key -> ``applied:false``.
* The model may ONLY choose from the REAL player ids the YOLO evidence already
  produced. Any hallucinated / unknown id is coerced to ``null``.
* ``winner``/``loser`` are ``null`` when the evidence or the reply is
  insufficient — we NEVER assert a winner from thin air.
* Strict JSON validation, a hard timeout and at most one retry. Any failure
  degrades to ``applied:false`` and the caller keeps its deterministic result.
* NO secrets in code/logs/output: the key is read from the environment only and
  is never echoed.
* Memory-safe: only a handful of frames are decoded (seek + read), downscaled,
  JPEG-encoded and released. The full video is never loaded.

Providers (``VLM_PROVIDER``)
---------------------------
  * ``qwen2-vl``   -> DashScope OpenAI-compatible endpoint (default)
  * ``internvl2``  -> generic OpenAI-compatible endpoint (set ``VLM_BASE_URL``)
  * ``openai``     -> any OpenAI-compatible vision endpoint (set ``VLM_BASE_URL``)

Env: ``VLM_PROVIDER, VLM_API_KEY, VLM_MODEL, VLM_BASE_URL, VLM_TIMEOUT_MS``.

CLI
---
    python3 yolo/vlm_analyzer.py --video duel.mp4 --evidence E.json \
        --duel-window 12.0 18.0 --duration 40 --output vlm.json [--no-network]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "vlm-duel-analyzer-1.0.0"

VALID_ARC = {"predator", "prey", "approach", "trap", "pressure",
             "escape", "dominance", "impact", "aftermath", "confrontation", "payoff"}

DEFAULT_BASE_URLS = {
    "qwen2-vl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "qwen2-vl-instruct": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "internvl2": "https://api.openai.com/v1",
    "openai": "https://api.openai.com/v1",
}
DEFAULT_MODELS = {
    "qwen2-vl": "qwen2-vl-72b-instruct",
    "qwen2-vl-instruct": "qwen2-vl-72b-instruct",
    "internvl2": "gpt-4o-mini",
    "openai": "gpt-4o-mini",
}
DEFAULT_TIMEOUT_MS = 60000
MAX_SAMPLE_FRAMES = 6
SAMPLE_WIDTH = 384


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(s or "").lower())


def resolve_config() -> dict:
    provider = (os.environ.get("VLM_PROVIDER") or "qwen2-vl").strip().lower()
    api_key = os.environ.get("VLM_API_KEY", "").strip()
    model = (os.environ.get("VLM_MODEL") or DEFAULT_MODELS.get(provider, "qwen2-vl-72b-instruct")).strip()
    base_url = (os.environ.get("VLM_BASE_URL")
                or DEFAULT_BASE_URLS.get(provider, DEFAULT_BASE_URLS["qwen2-vl"])).strip().rstrip("/")
    timeout_ms = int(os.environ.get("VLM_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS)) or DEFAULT_TIMEOUT_MS)
    return {"provider": provider, "api_key": api_key, "model": model,
            "base_url": base_url, "timeout_ms": timeout_ms, "enabled": bool(api_key)}


def _sample_frames(video_path: str, start: float, end: float, count: int = MAX_SAMPLE_FRAMES):
    """Decode a few frames across [start, end]; return list of JPEG data URLs.

    Lazy: cv2 is imported here so a deployment without OpenCV simply gets an
    empty list (the analyzer then degrades to ``applied:false``)."""
    frames = []
    try:
        import cv2  # lazy import
    except Exception:
        return frames
    if not video_path or not os.path.exists(video_path):
        return frames
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        if fps <= 0.5 or fps > 240:
            fps = 25.0
        span = max(0.4, float(end) - float(start))
        for i in range(count):
            t = float(start) + span * (i / max(1, count - 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > SAMPLE_WIDTH:
                scale = SAMPLE_WIDTH / float(w)
                frame = cv2.resize(frame, (SAMPLE_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            frames.append(f"data:image/jpeg;base64,{b64}")
            del frame, buf
    except Exception:
        return frames
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return frames


def _player_ids(evidence: dict) -> set:
    ids = set()
    for key in ("players", "player_ids", "track_ids"):
        vals = evidence.get(key)
        if isinstance(vals, list):
            for v in vals:
                ids.add(str(v))
    duel = evidence.get("duel") or {}
    for k in ("predator", "prey"):
        if duel.get(k) is not None:
            ids.add(str(duel.get(k)))
    if isinstance(evidence.get("track_summary"), dict):
        for tid in evidence["track_summary"].get("tracks", {}) or {}:
            ids.add(str(tid))
    return {i for i in ids if i and i.lower() != "none"}


def _build_messages(evidence: dict, frames: list, window) -> list:
    text = (
        "You are a football ANALYST reviewing a SHORT duel clip between two players.\n"
        "Use ONLY what you can actually see and the VERIFIED evidence below.\n"
        "Return STRICT JSON ONLY:\n"
        '{"duel_moment": <seconds>, "winner": <player_id|null>, "loser": <player_id|null>, '
        '"story_arc": ["approach","pressure",...], "confidence": <0..1>}\n'
        "Rules:\n"
        "- winner/loser MUST be one of the verified player ids, or null if unclear.\n"
        "- Never invent a player id, a goal, a score or an event.\n"
        "- story_arc may only use: " + ", ".join(sorted(VALID_ARC)) + ".\n"
        "- confidence reflects how certain you are (0 = guessing, 1 = certain).\n\n"
        "VERIFIED EVIDENCE:\n"
        + json.dumps({
            "clip_window_seconds": [round(float(window[0]), 3), round(float(window[1]), 3)],
            "verified_player_ids": sorted(_player_ids(evidence)),
            "duel": evidence.get("duel"),
            "events": evidence.get("events"),
            "kinematics": evidence.get("kinematics"),
        }, ensure_ascii=False)
    )
    content = [{"type": "text", "text": text}]
    for url in frames:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": "You are a precise sports-video analyst. Reply with strict JSON only."},
        {"role": "user", "content": content},
    ]


def _post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _extract_json(text: str):
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            obj = json.loads(text[s:e + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _coerce_verdict(obj: dict, evidence: dict, window, duration: float) -> dict:
    ids = _player_ids(evidence)
    winner = obj.get("winner")
    loser = obj.get("loser")
    winner = str(winner) if winner is not None else None
    loser = str(loser) if loser is not None else None
    if winner is not None and _norm(winner) not in {_norm(i) for i in ids}:
        winner = None
    if loser is not None and _norm(loser) not in {_norm(i) for i in ids}:
        loser = None
    if winner is not None and loser is not None and _norm(winner) == _norm(loser):
        loser = None

    arc = obj.get("story_arc")
    arc_clean = []
    if isinstance(arc, list):
        for a in arc:
            key = str(a).strip().lower().replace(" ", "_")
            if key in VALID_ARC and key not in arc_clean:
                arc_clean.append(key)

    duel_moment = obj.get("duel_moment")
    try:
        duel_moment = float(duel_moment)
    except (TypeError, ValueError):
        duel_moment = (float(window[0]) + float(window[1])) / 2.0
    duel_moment = _clamp(duel_moment, max(0.0, float(window[0]) - 2.0),
                         min(max(0.01, duration), float(window[1]) + 2.0))

    confidence = _clamp(obj.get("confidence", 0.0), 0.0, 1.0)
    return {"duel_moment": round(duel_moment, 3), "winner": winner, "loser": loser,
            "story_arc": arc_clean, "confidence": round(confidence, 4)}


def analyze(video_path: str, evidence: dict, window, duration: float,
            timeout_ms: int = DEFAULT_TIMEOUT_MS, allow_network: bool = True):
    """Return ``(verdict_or_None, meta)``. Never raises."""
    cfg = resolve_config()
    meta = {"provider": cfg["provider"], "model": cfg["model"], "applied": False,
            "reason": None, "version": VERSION}
    if not allow_network:
        meta["reason"] = "network-disabled"
        return None, meta
    if not cfg["enabled"]:
        meta["reason"] = "VLM_API_KEY not configured"
        return None, meta
    if not video_path or not os.path.exists(video_path):
        meta["reason"] = "clip-not-available"
        return None, meta

    frames = _sample_frames(video_path, float(window[0]), float(window[1]))
    if not frames:
        meta["reason"] = "no-frames-sampled"
        return None, meta

    url = f"{cfg['base_url']}/chat/completions"
    payload = {"model": cfg["model"], "messages": _build_messages(evidence, frames, window),
               "temperature": 0.2, "max_tokens": 400, "stream": False}
    timeout = max(3.0, float(timeout_ms) / 1000.0)

    obj = None
    last_error = None
    for _ in range(2):  # at most one retry
        try:
            resp = _post_json(url, payload, cfg["api_key"], timeout)
            choices = resp.get("choices") or []
            content = ""
            if choices and isinstance(choices[0], dict):
                content = str((choices[0].get("message") or {}).get("content") or "")
            obj = _extract_json(content)
            if obj is not None:
                break
            last_error = "invalid-json-in-reply"
        except urllib.error.HTTPError as exc:
            last_error = f"http-{exc.code}"
            if exc.code in (400, 401, 403, 404):
                break
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(0.5)

    if obj is None:
        meta["reason"] = last_error or "no-response"
        return None, meta

    verdict = _coerce_verdict(obj, evidence, window, duration)
    meta.update({"applied": True, "reason": None, "frames": len(frames)})
    return verdict, meta


def main():
    ap = argparse.ArgumentParser(description="Optional VLM duel analyzer (Qwen2-VL / InternVL2)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--duel-window", nargs=2, type=float, required=True, metavar=("START", "END"))
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()

    try:
        evidence = json.loads(Path(a.evidence).read_text(encoding="utf8"))
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"evidence unreadable: {exc}"}))
        return 1

    verdict, meta = analyze(a.video, evidence, a.duel_window, a.duration,
                            timeout_ms=a.timeout_ms, allow_network=not a.no_network)
    out = {"success": True, "version": VERSION, "verdict": verdict, "meta": meta}
    Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"success": True, "version": VERSION, "applied": meta.get("applied"),
                      "winner": (verdict or {}).get("winner"), "reason": meta.get("reason")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
