#!/usr/bin/env python3
"""DeepSeek R1 TEXT STORYTELLER (psychological football thriller).

This module is a TEXT-ONLY writer. It is deliberately NOT an event detector: it
never invents a football action, a winner, a score or a player identity. It only
turns already-VERIFIED evidence (YOLO/tracking/events/duel detection) into six
short, original, escalating inner-monologue lines in a
"Predator vs Prey" psychological-thriller register.

Contract
--------
Input : verified evidence JSON (the cinematic_storyteller bundle: duel + story
        arc + real player ids) + the duel_moment + the source duration.
Output: ``[{ "time": float, "text": str, "position": str }]`` with EXACTLY 6
        lines:
            line 1 approach      line 2 pressure    line 3 trap
            line 4 confrontation  line 5 decisive    line 6 payoff
        Timestamps are synced around ``duel_moment`` and clamped inside
        ``[0.3, duration - 0.3]`` with a guaranteed minimum spacing, so they can
        never collide with each other.

Fail-safe (never breaks the render)
-----------------------------------
Any failure — no API key, network error, timeout, invalid JSON, fewer than six
uniquely-valid lines — yields ``story_script = []`` and a ``reason``. The render
continues unchanged.

Security
--------
The API key is read from the environment ONLY (``DEEPSEEK_API_KEY``), is never
written to the output, never logged, and never embedded in the prompt. Model and
optional base URL come from ``DEEPSEEK_MODEL`` / ``DEEPSEEK_BASE_URL``.

CLI
---
    python3 yolo/deepseek_storyteller.py --storyteller S.json --duration 40 \
        --output script.json [--timeout-ms 45000] [--no-network]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VERSION = "deepseek-storyteller-1.0.0"

# The six narrative beats, in order. `position` is where the caption should sit
# on the vertical (9:16) frame so successive lines never overlap.
BEATS = [
    ("approach", "lower_third"),
    ("pressure", "center"),
    ("trap", "lower_third"),
    ("confrontation", "center"),
    ("action", "center"),
    ("payoff", "lower_third"),
]

# Tone REFERENCES shown to the model as *inspiration only*. They must not be
# reproduced verbatim; the prompt forbids copying and the validator rejects any
# line that matches one of them (or an existing on-screen caption).
TONE_REFERENCES = [
    "COME CLOSER...",
    "DEAD END",
    "A PREY WALKING STRAIGHT INTO HIS TRAP",
    "I'M LOCKING YOU DOWN RIGHT NOW",
    "TERRIFYING GENIUS...",
    "LIVE THE GOAL",
]

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_TIMEOUT_MS = 45000
DEFAULT_MAX_LINES = 6
MIN_GAP = 0.55          # minimum seconds between consecutive lines
EDGE_MARGIN = 0.3       # keep timestamps away from the very start/end
SPREAD = 1.35           # seconds between beats around the duel moment


def _norm(text: str) -> str:
    """Normalise a caption for collision checks (case/punct/space insensitive)."""
    return re.sub(r"[^a-z0-9 ]+", "", str(text or "").lower()).strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo


def _schedule_times(duel_moment: float, duration: float) -> list:
    """Deterministic beat times synced around the real duel moment."""
    m = _clamp(duel_moment, EDGE_MARGIN, max(EDGE_MARGIN + 0.01, duration - EDGE_MARGIN))
    offsets = [-1.55 * SPREAD, -1.05 * SPREAD, -0.5 * SPREAD, 0.0, 0.75 * SPREAD, 1.7 * SPREAD]
    times = [_clamp(m + off, EDGE_MARGIN, max(EDGE_MARGIN, duration - EDGE_MARGIN)) for off in offsets]
    return _enforce_spacing(times, duration)


def _enforce_spacing(times: list, duration: float) -> list:
    """Guarantee strictly increasing timestamps with at least MIN_GAP apart."""
    hi = max(EDGE_MARGIN + 0.01, duration - EDGE_MARGIN)
    out = []
    prev = -1e9
    for t in times:
        t = _clamp(t, EDGE_MARGIN, hi)
        if t < prev + MIN_GAP:
            t = prev + MIN_GAP
        t = _clamp(t, EDGE_MARGIN, hi)
        out.append(round(t, 3))
        prev = t
    return out


def _build_prompt(evidence: dict, forbidden: list) -> str:
    duel = evidence.get("duel") or {}
    arc = evidence.get("story_arc") or []
    predator = duel.get("predator")
    prey = duel.get("prey")
    moment = evidence.get("duel_moment")
    forbidden_block = ", ".join(f'"{f}"' for f in forbidden if f) or "(none)"

    return (
        "You are a football PSYCHOLOGICAL-THRILLER voice-over writer.\n"
        "Write EXACTLY 6 short, original lines of escalating inner monologue in a\n"
        "'predator vs prey' register: the strongest attacking player is the PREDATOR\n"
        "hunting a specific defender, the PREY. Each line must escalate tension.\n\n"
        "HARD RULES\n"
        "- Do NOT invent any football event, goal, score, name or outcome.\n"
        "- Do NOT state that a goal/assist/win happened unless the evidence says so.\n"
        "- Use ONLY the evidence below; if evidence is thin, stay atmospheric.\n"
        "- Exactly 6 lines, each under 42 characters, ALL CAPS, no emojis.\n"
        "- Do NOT copy these reference phrases literally (inspiration only): "
        + ", ".join(f'"{t}"' for t in TONE_REFERENCES) + "\n"
        f"- Do NOT reuse these already-on-screen captions: {forbidden_block}\n"
        "- Return STRICT JSON ONLY, an array of 6 objects:\n"
        '  [{"time": <seconds>, "text": "<LINE>", "position": "lower_third|center"}]\n'
        "- Order the lines: 1 approach, 2 pressure, 3 trap, 4 confrontation,\n"
        "  5 decisive action, 6 payoff. Centre the timing on the duel moment.\n\n"
        "VERIFIED EVIDENCE (do not fabricate beyond this):\n"
        + json.dumps({
            "duel_moment_seconds": moment,
            "predator_player_id": predator,
            "prey_player_id": prey,
            "duel_confidence": duel.get("confidence"),
            "duel_evidence": (duel.get("evidence") or {}),
            "story_arc": arc,
            "notes": evidence.get("notes"),
        }, ensure_ascii=False)
    )


def _post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (env-configured host)
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _extract_json_array(text: str):
    """Pull a JSON array out of a model reply (tolerates prose / code fences)."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None
    return None


def _validate_lines(lines, duration: float, duel_moment: float,
                    forbidden_norm: set) -> list:
    """Validate + clamp + de-duplicate the model's lines. Returns 0..6 valid rows."""
    if not isinstance(lines, list):
        return []
    cleaned = []
    for row in lines:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text).upper()[:42].strip()
        if not text:
            continue
        norm = _norm(text)
        if not norm or norm in forbidden_norm:
            continue
        position = str(row.get("position", "") or "").strip().lower()
        if position not in ("lower_third", "center", "top"):
            position = "lower_third"
        try:
            t = float(row.get("time"))
        except (TypeError, ValueError):
            t = None
        cleaned.append({"time": t, "text": text, "position": position})

    if not cleaned:
        return []

    # De-duplicate by normalised text (a repeated line breaks the escalation).
    seen = set()
    unique = []
    for row in cleaned:
        norm = _norm(row["text"])
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(row)

    # Fall back to the deterministic schedule when the model omitted timing.
    schedule = _schedule_times(duel_moment, duration)
    for i, row in enumerate(unique):
        if row["time"] is None:
            row["time"] = schedule[min(i, len(schedule) - 1)]

    times = _enforce_spacing([row["time"] for row in unique], duration)
    for row, t in zip(unique, times):
        row["time"] = t
    return unique


def generate_script(storyteller: dict, duration: float, existing_texts=None,
                    timeout_ms: int = DEFAULT_TIMEOUT_MS, allow_network: bool = True):
    """Return ``(script, meta)`` where script is ``[{time,text,position}]``.

    ``script`` is ``[]`` on any failure; ``meta`` explains why (never contains the
    API key)."""
    meta = {"provider": "deepseek", "applied": False, "reason": None, "version": VERSION}
    duration = max(1.0, float(duration or 1.0))
    duel = (storyteller or {}).get("duel") or {}
    duel_moment = storyteller.get("duel_moment")
    if duel_moment is None:
        duel_moment = (storyteller or {}).get("hero_moment") or duration * 0.62
    duel_moment = _clamp(duel_moment, EDGE_MARGIN, max(EDGE_MARGIN, duration - EDGE_MARGIN))

    forbidden = list(TONE_REFERENCES) + [str(t) for t in (existing_texts or []) if t]
    forbidden_norm = {_norm(f) for f in forbidden if _norm(f)}

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    meta["model"] = model

    if not allow_network:
        meta["reason"] = "network-disabled"
        return [], meta
    if not api_key:
        meta["reason"] = "DEEPSEEK_API_KEY not configured"
        return [], meta

    url = f"{base_url}/chat/completions"
    prompt = _build_prompt(storyteller or {}, forbidden)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You write terse psychological football-thriller captions. Reply with strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "max_tokens": 400,
        "stream": False,
    }

    timeout = max(3.0, float(timeout_ms) / 1000.0)
    raw_lines = None
    last_error = None
    # Bounded retries: one immediate retry on transient failure, nothing more.
    for attempt in range(2):
        try:
            resp = _post_json(url, payload, api_key, timeout)
            choices = resp.get("choices") or []
            content = ""
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                content = str(msg.get("content") or "")
            raw_lines = _extract_json_array(content)
            if raw_lines is not None:
                break
            last_error = "invalid-json-in-reply"
        except urllib.error.HTTPError as exc:  # never echo the key
            last_error = f"http-{exc.code}"
            if exc.code in (400, 401, 403, 404):
                break  # a bad key/model will not fix itself on retry
        except Exception as exc:  # timeout / network / DNS
            last_error = type(exc).__name__
        time.sleep(0.5)

    if raw_lines is None:
        meta["reason"] = last_error or "no-response"
        return [], meta

    valid = _validate_lines(raw_lines, duration, duel_moment, forbidden_norm)
    if len(valid) < DEFAULT_MAX_LINES:
        # Repair pass: tell the model exactly which lines were rejected.
        meta["reason"] = f"only-{len(valid)}-valid-lines"
        return [], meta

    script = [{"time": row["time"], "text": row["text"], "position": row["position"]}
              for row in valid[:DEFAULT_MAX_LINES]]
    meta.update({"applied": True, "reason": None, "count": len(script),
                 "duel_moment": duel_moment})
    return script, meta


def main():
    ap = argparse.ArgumentParser(description="DeepSeek R1 psychological-thriller text storyteller")
    ap.add_argument("--storyteller", required=True, help="cinematic_storyteller bundle JSON")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--plan", default=None, help="optional edit plan (for caption-collision checks)")
    ap.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()

    try:
        bundle = json.loads(Path(a.storyteller).read_text(encoding="utf8"))
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"storyteller unreadable: {exc}"}))
        return 1

    existing = []
    if a.plan and Path(a.plan).exists():
        try:
            plan = json.loads(Path(a.plan).read_text(encoding="utf8"))
            existing = [str(c.get("text", "")) for c in (plan.get("timeline") or [])]
        except Exception:
            existing = []

    script, meta = generate_script(bundle, a.duration, existing_texts=existing,
                                   timeout_ms=a.timeout_ms, allow_network=not a.no_network)
    out = {"success": True, "version": VERSION, "story_script": script, "meta": meta}
    Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"success": True, "version": VERSION, "count": len(script),
                      "applied": meta.get("applied"), "reason": meta.get("reason")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
