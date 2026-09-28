#!/usr/bin/env python3
"""
SAM CINEMATIC SEGMENTATION / TRACKING ENGINE  (OPTIONAL LAYER)
==============================================================

An OPT-IN, self-contained segmentation + subject-tracking service that sits
BETWEEN the existing football analysis and the existing CinematicEngine render.
It is an ENHANCEMENT layer, never a replacement:

    Football Analysis -> Event Detection -> Interesting Segment
        -> [ OPTIONAL SAM 3.1 : segmentation + tracking ]   <-- this module
        -> Edit Plan -> CinematicEngine -> QC -> Final MP4

Design guarantees (mirrors the project's existing fault-tolerant bridges such as
``server/cinematicEngine.ts`` and ``yolo/rife_interpolate.py``):

* DEFAULT OFF. Inert unless ``--sam-enabled true`` / ``SAM_ENABLED=true``.
  When disabled (or unavailable) it exits 0 with ``{"applied": false}`` and the
  caller keeps the existing YOLO/tracking path untouched.
* NEVER runs SAM over the whole video. It ONLY processes the specific
  interesting segments the edit plan already selected, with bounded frame
  sampling and a hard wall-clock deadline.
* YOLO = detection, SAM = segmentation. Reuses the existing YOLO/tracking
  output as prompts and NEVER re-detects the ball frame-by-frame.
* Never throws to the caller: failures degrade to ``{"applied": false}``.
* Emits a NORMALISED, compact result. Masks are stored as compressed RLE files
  referenced by id - never as raw arrays inside the edit plan.
* No API keys, no network, no secrets in logs.

CLI
---
    python3 python/sam_engine.py check   [--json]
    python3 python/sam_engine.py run --video V --detections D --plan P --output O
                                      [--events E] [--cache-dir C]
                                      [--mask-mode rle|png|none]
                                      [--sam-enabled true|false]
                                      [--stride N] [--preview-window S]
                                      [--max-seconds S] [--deadline-seconds S]
                                      [--max-segments N] [--max-objects-per-segment N]
                                      [--mask-resolution N] [--cache-max-entries N]
                                      [--mask-preview-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The adapter module lives next to this file; make the import robust regardless
# of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:  # pragma: no cover - import wiring
    import sam_adapters  # type: ignore
    from sam_adapters import create_backend, backend_availability  # type: ignore
except Exception:  # pragma: no cover - extremely defensive
    sam_adapters = None  # type: ignore

    def create_backend(*_a, **_k):  # type: ignore
        raise RuntimeError("sam_adapters unavailable")

    def backend_availability() -> Dict[str, Any]:  # type: ignore
        return {"meta_sam": False, "reason": "sam_adapters import failed"}

VERSION = "sam-1.0.0"

BALL_NAMES = {"ball", "football", "soccer_ball", "soccer-ball", "sports ball", "sports_ball"}
GOALIE_NAMES = {"goalkeeper", "goalie", "gk", "keeper"}
PLAYER_NAMES = {"player", "person", "footballer", "athlete"}

# Reasons that genuinely benefit from a subject-isolation mask. A plain LUT or a
# normal slow-motion shot is NOT in this set and therefore never triggers SAM.
ISOLATION_REASONS = {"shot", "goal", "dribble", "tackle", "save", "celebration"}

# Keyword -> reason mapping used to explain WHY a segment was segmented.
KEYWORD_REASONS: List[Tuple[str, str]] = [
    ("goal", "goal"),
    ("celebrat", "celebration"),
    ("save", "save"),
    ("tackle", "tackle"),
    ("dribb", "dribble"),
    ("nutmeg", "dribble"),
    ("feint", "dribble"),
    ("shot", "shot"),
    ("strike", "shot"),
    ("finish", "shot"),
]


# --------------------------------------------------------------------------- #
# Logging / progress (machine-readable on stderr, mirrors track_football.py)
# --------------------------------------------------------------------------- #
def log_progress(pct: float, msg: str = "") -> None:
    try:
        sys.stderr.write(f"PROGRESS {int(pct)} {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _clean(text: Any, limit: int = 160) -> str:
    """Strip anything that looks like a secret before it reaches a log/stdout."""
    s = str(text)
    for marker in ("ghp_", "sk-", "AIza", "xi-api-key", "Bearer "):
        if marker in s:
            s = s.replace(marker, "***")
    return s[:limit]


# --------------------------------------------------------------------------- #
# Compressed masks (COCO-style RLE; numpy-backed, no heavy deps)
# --------------------------------------------------------------------------- #
def encode_rle(mask) -> List[int]:
    """Row-major (Fortran order) run-length encoding, COCO-compatible."""
    import numpy as np

    flat = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [flat.size]))
    counts = (ends - starts).astype(int)
    if int(flat[0]) == 1:
        counts = np.concatenate(([0], counts))
    return counts.tolist()


def decode_rle(counts: List[int], height: int, width: int):
    """Inverse of :func:`encode_rle` (kept for tests / consumers)."""
    import numpy as np

    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    value = 0
    for c in counts:
        c = int(c)
        if value:
            flat[pos:pos + c] = 1
        pos += c
        value ^= 1
    return flat.reshape((height, width), order="F")


def _downscale_mask(mask, long_side: int):
    """Shrink a mask to a small, compression-friendly resolution."""
    import numpy as np

    h, w = mask.shape[:2]
    scale = min(1.0, float(long_side) / float(max(h, w) or 1))
    if scale >= 1.0:
        return mask.astype(np.uint8), h, w
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        import cv2
        small = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    except Exception:
        # Nearest-neighbour without cv2.
        yi = (np.arange(nh) / scale).astype(int).clip(0, h - 1)
        xi = (np.arange(nw) / scale).astype(int).clip(0, w - 1)
        small = mask.astype(np.uint8)[yi][:, xi]
    return small, nh, nw


def write_image(path: str, array) -> bool:
    """Portable image writer.

    Some OpenCV builds reject plain numpy arrays created outside OpenCV
    ("img is not a numpy array"), so Pillow is tried first (it ships with the
    project's ultralytics dependency) and OpenCV is the fallback. Returns True
    only when the file was actually produced.
    """
    import numpy as np

    arr = np.asarray(array)
    # Pillow path (1-channel grayscale or 3-channel BGR->RGB).
    try:
        from PIL import Image  # type: ignore

        if arr.ndim == 2:
            Image.fromarray(arr.astype(np.uint8)).save(path)
        else:
            Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]).astype(np.uint8)).save(path)
        return os.path.exists(path)
    except Exception:
        pass
    # OpenCV fallback.
    try:
        import cv2

        return bool(cv2.imwrite(path, np.ascontiguousarray(arr)))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Bounded, self-pruning mask cache  (key = video hash + segment + object/class)
# --------------------------------------------------------------------------- #
class MaskCache:
    """Disk cache of compressed masks keyed by content, with a hard entry cap."""

    def __init__(self, cache_dir: Path, max_entries: int = 500) -> None:
        self.dir = Path(cache_dir)
        self.max_entries = max(16, int(max_entries))
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.hits = 0
        self.misses = 0

    def key(self, video_hash: str, segment_index: int, cls: str, bbox: List[float], frame: int) -> str:
        payload = "|".join([
            VERSION, video_hash, str(segment_index), str(cls),
            ",".join(f"{float(v):.1f}" for v in bbox), str(int(frame)),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.dir / f"{key}.json"
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            os.utime(path, None)  # LRU touch
            self.hits += 1
            return data
        except Exception:
            self.misses += 1
            return None

    def put(self, key: str, payload: Dict[str, Any]) -> None:
        try:
            path = self.dir / f"{key}.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            return
        self._prune()

    def _prune(self) -> None:
        """Bound growth: evict least-recently-used entries beyond the cap."""
        try:
            entries = [(p.stat().st_mtime, p) for p in self.dir.glob("*.json")]
        except Exception:
            return
        if len(entries) <= self.max_entries:
            return
        entries.sort()
        for _, path in entries[: len(entries) - self.max_entries]:
            try:
                path.unlink()
            except Exception:
                pass


def video_content_hash(path: str, sample_bytes: int = 1 << 20) -> str:
    """Cheap, stable content fingerprint (size + mtime + leading sample)."""
    h = hashlib.sha256()
    st = os.stat(path)
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())
    try:
        with open(path, "rb") as fh:
            h.update(fh.read(sample_bytes))
    except Exception:
        pass
    return h.hexdigest()[:24]


# --------------------------------------------------------------------------- #
# Inputs: detections (YOLO/tracking), edit plan, optional events
# --------------------------------------------------------------------------- #
def load_json(path: Optional[str]) -> Any:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def index_detections(tracking: Any) -> Dict[str, Any]:
    """Group the existing YOLO samples by track and by frame for fast lookup.

    IMPORTANT: this REUSES the existing detection output. SAM is never asked to
    detect anything - it only receives boxes/points as prompts.
    """
    samples = []
    if isinstance(tracking, dict):
        samples = tracking.get("summary", {}).get("samples", []) or []
    players_by_track: Dict[str, List[Dict[str, Any]]] = {}
    ball_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for s in samples:
        try:
            f = int(s.get("frame", 0))
        except Exception:
            continue
        cls = str(s.get("class", "")).lower()
        if cls in BALL_NAMES:
            ball_by_frame.setdefault(f, []).append(s)
            continue
        tid = s.get("track_id")
        if tid is None:
            continue
        players_by_track.setdefault(str(tid), []).append(s)
    for tid in players_by_track:
        players_by_track[tid].sort(key=lambda z: int(z.get("frame", 0)))
    return {"players_by_track": players_by_track, "ball_by_frame": ball_by_frame}


def _bbox_px(sample: Dict[str, Any], width: int, height: int) -> Optional[List[float]]:
    if all(k in sample for k in ("x1", "y1", "x2", "y2")):
        x1 = float(sample["x1"]) * width
        y1 = float(sample["y1"]) * height
        x2 = float(sample["x2"]) * width
        y2 = float(sample["y2"]) * height
    else:
        cx = float(sample.get("x", 0.5)) * width
        cy = float(sample.get("y", 0.5)) * height
        bw = max(8.0, float(sample.get("w", 0.1)) * width)
        bh = max(8.0, float(sample.get("h", 0.2)) * height)
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _nearest_track_sample(track: List[Dict[str, Any]], frame: int, tol: int) -> Optional[Dict[str, Any]]:
    best = None
    best_d = tol + 1
    for s in track:
        d = abs(int(s.get("frame", 0)) - frame)
        if d < best_d:
            best, best_d = s, d
    return best if best is not None and best_d <= tol else None


# --------------------------------------------------------------------------- #
# Segment selection - WHICH moments deserve SAM (never the whole video)
# --------------------------------------------------------------------------- #
def classify_segment_reason(clip: Dict[str, Any], events: Optional[List[Dict[str, Any]]], start: float, end: float) -> str:
    text = " ".join([
        str(clip.get("action", "")),
        str(clip.get("text", "")),
        str(clip.get("shot_type", "")),
        str(clip.get("beat_role", "")),
        str((clip.get("madness") or {}).get("effect", "") if isinstance(clip.get("madness"), dict) else ""),
    ]).lower()
    for kw, reason in KEYWORD_REASONS:
        if kw in text:
            return reason
    if events:
        labels: List[str] = []
        for e in events:
            try:
                t = float(e.get("time", -999))
            except Exception:
                continue
            if start - 0.35 <= t <= end + 0.35:
                labels.extend([str(x).lower() for x in e.get("events", [])])
        joined = " ".join(labels)
        if "ball_engagement" in joined or "high_ball_speed" in joined:
            return "shot"
        if "pressure_candidate" in joined or "direction_change_candidate" in joined:
            return "tackle"
        if "explosive_run" in joined:
            return "dribble"
    return "motion"


def build_segments(plan: Any, fps: float, duration: float, max_segments: int,
                   force_all: bool = False) -> List[Dict[str, Any]]:
    """Turn the edit plan timeline into bounded segment windows (in frames)."""
    timeline = plan.get("timeline", []) if isinstance(plan, dict) else []
    events = (plan.get("_events") or []) if isinstance(plan, dict) else []
    segments: List[Dict[str, Any]] = []
    for i, clip in enumerate(timeline):
        try:
            start = max(0.0, float(clip.get("source_start", 0)))
            end = max(start + 0.1, float(clip.get("source_end", start + 1)))
        except Exception:
            continue
        reason = classify_segment_reason(clip, events, start, end)
        needs_isolation = reason in ISOLATION_REASONS
        if not needs_isolation and not force_all:
            continue
        segments.append({
            "index": len(segments),
            "timeline_index": clip.get("timeline_index", i),
            "start": round(start, 3),
            "end": round(min(end, duration if duration > 0 else end), 3),
            "startFrame": max(1, int(round(start * fps))),
            "endFrame": max(1, int(round(end * fps))),
            "reason": reason,
            "needsIsolation": True,
            "shotType": clip.get("shot_type"),
            "beatRole": clip.get("beat_role"),
        })
    # Most "interesting" first: explicit football events before generic motion.
    rank = {"goal": 0, "shot": 1, "save": 2, "tackle": 3, "dribble": 4, "celebration": 5, "motion": 9}
    segments.sort(key=lambda s: (rank.get(s["reason"], 9), s["start"]))
    return segments[:max(1, max_segments)]


# --------------------------------------------------------------------------- #
# Target selection within a segment (subject isolation only when justified)
# --------------------------------------------------------------------------- #
def select_segment_targets(segment: Dict[str, Any], idx: Dict[str, Any], meta: Dict[str, Any],
                           opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    players = idx["players_by_track"]
    ball_by_frame = idx["ball_by_frame"]
    width, height = meta["width"], meta["height"]
    fs, fe = segment["startFrame"], segment["endFrame"]

    candidates: List[Dict[str, Any]] = []
    for tid, track in players.items():
        in_window = [s for s in track if fs - 3 <= int(s.get("frame", 0)) <= fe + 3]
        if not in_window:
            continue
        cls = str(in_window[0].get("class", "player")).lower()
        conf = sum(float(s.get("confidence", 0)) for s in in_window) / len(in_window)
        area = 0.0
        have_ball = False
        for s in in_window:
            f = int(s.get("frame", 0))
            bb = _bbox_px(s, width, height)
            if bb:
                area = max(area, (bb[2] - bb[0]) * (bb[3] - bb[1]))
            for q in range(f - 3, f + 4):
                for b in ball_by_frame.get(q, []):
                    bx, by = float(b.get("x", .5)), float(b.get("y", .5))
                    if math.hypot(bx - float(s.get("x", .5)), by - float(s.get("y", .5))) < 0.16:
                        have_ball = True
                        break
        priority = 0.0
        kind = "player"
        if opts.get("subject_track_id") is not None and str(tid) == str(opts["subject_track_id"]):
            priority += 1.0
            kind = "target_player"
        if cls in GOALIE_NAMES:
            priority += 0.6
            kind = "goalkeeper"
        if have_ball:
            priority += 0.5
            kind = "player_ball"
        priority += min(0.4, area / (width * height * 0.25))
        priority += 0.2 * conf
        candidates.append({"track_id": tid, "class": cls, "kind": kind,
                           "priority": round(priority, 4), "confidence": round(conf, 3)})

    candidates.sort(key=lambda c: c["priority"], reverse=True)
    allow_ball = bool(opts.get("allow_ball"))
    max_objs = int(opts.get("max_objects_per_segment", 3))
    chosen = candidates[:max_objs]
    segment["targetKinds"] = [c["kind"] for c in chosen]
    return chosen


# --------------------------------------------------------------------------- #
# Video metadata + frame reader
# --------------------------------------------------------------------------- #
def video_meta(path: str) -> Dict[str, Any]:
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"success": False, "error": "video-open-failed"}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.5 or fps > 240:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if width <= 0 or height <= 0:
            return {"success": False, "error": "invalid-video-dimensions"}
        return {"success": True, "fps": fps, "width": width, "height": height,
                "frames": frames, "duration": round(frames / fps, 3) if frames else 0.0}
    except Exception as exc:
        return {"success": False, "error": f"cv2-unavailable: {_clean(exc)}"}


# --------------------------------------------------------------------------- #
# Core: segment + track the interesting subjects of ONE segment
# --------------------------------------------------------------------------- #
def process_segment(video: str, segment: Dict[str, Any], targets: List[Dict[str, Any]],
                    backend, cache: MaskCache, meta: Dict[str, Any], video_hash: str,
                    opts: Dict[str, Any], idx: Dict[str, Any]) -> Dict[str, Any]:
    import cv2

    width, height = meta["width"], meta["height"]
    stride = max(1, int(opts.get("stride", 4)))
    max_frames = max(1, int(opts.get("max_frames_per_segment", 6)))
    mask_res = int(opts.get("mask_resolution", 512))
    mask_mode = str(opts.get("mask_mode", "rle"))
    preview_dir = opts.get("mask_preview_dir")
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)

    fs, fe = segment["startFrame"], segment["endFrame"]
    span = max(1, fe - fs)
    # Evenly spaced sample frames inside the segment, bounded by max_frames.
    n = min(max_frames, max(1, span // stride))
    sample_frames = sorted({fs + int(round(span * k / max(1, n))) for k in range(n + 1)})

    objects: Dict[str, Dict[str, Any]] = {}
    cap = cv2.VideoCapture(video)
    sampled = 0
    warnings: List[str] = []
    try:
        for f in sample_frames:
            if f < 1:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled += 1
            prompts: List[Dict[str, Any]] = []
            prompt_targets: List[Dict[str, Any]] = []
            for t in targets:
                sample = _nearest_track_sample(idx["players_by_track"].get(t["track_id"], []), f, max(2, stride))
                if not sample:
                    continue
                bb = _bbox_px(sample, width, height)
                if not bb:
                    continue
                prompts.append({"bbox": bb})
                prompt_targets.append({"target": t, "sample": sample, "bboxPx": bb})
            if not prompts:
                continue
            try:
                results = backend.segment_frame(frame, prompts)
            except Exception as exc:  # never crash a segment on one bad frame
                warnings.append(f"segment {segment['index']} frame {f}: {_clean(exc)}")
                continue
            for info, res in zip(prompt_targets, results):
                t = info["target"]
                sample = info["sample"]
                bb = info["bboxPx"]
                mask = res.get("mask")
                score = float(res.get("score", 0.0))
                obj = objects.setdefault(t["track_id"], {
                    "id": f"seg{segment['index']}_obj{t['track_id']}",
                    "class": t["class"],
                    "kind": t["kind"],
                    "trackId": t["track_id"],
                    "confidence": round(float(t["confidence"]), 4),
                    "startFrame": f, "endFrame": f, "frames": [], "_scores": [],
                    "maskRefs": [], "keyframeMaskRef": None, "backend": res.get("source", "unknown"),
                })
                obj["startFrame"] = min(obj["startFrame"], f)
                obj["endFrame"] = max(obj["endFrame"], f)
                norm_bbox = [round(bb[0] / width, 5), round(bb[1] / height, 5),
                             round(bb[2] / width, 5), round(bb[3] / height, 5)]
                frame_rec: Dict[str, Any] = {"frame": f, "bbox": norm_bbox,
                                             "score": round(score, 4), "maskRef": None}
                if mask is not None and mask_mode != "none":
                    small, nh, nw = _downscale_mask(mask, mask_res)
                    rle = encode_rle(small)
                    key = cache.key(video_hash, segment["index"], t["class"], bb, f)
                    mask_path = None
                    if preview_dir and obj["keyframeMaskRef"] is None:
                        mask_path = str(Path(preview_dir) / f"{key}.png")
                        if not write_image(mask_path, (small * 255).astype("uint8")):
                            mask_path = None
                    ref = {
                        "maskRef": mask_path,
                        "rle": rle,
                        "shape": [nh, nw],
                        "class": t["class"],
                        "bbox": norm_bbox,
                        "score": round(score, 4),
                    }
                    cache.put(key, ref)
                    frame_rec["maskRef"] = mask_path or f"cache:{key}"
                    obj["maskRefs"].append(mask_path or f"cache:{key}")
                    if mask_path and score >= float(obj.get("_bestScore", -1)):
                        obj["_bestScore"] = score
                        obj["keyframeMaskRef"] = mask_path
                obj["frames"].append(frame_rec)
                obj["_scores"].append(score)
    finally:
        cap.release()

    out_objects = []
    for obj in objects.values():
        scores = obj.pop("_scores", [])
        obj.pop("_bestScore", None)
        obj["maskQuality"] = round(sum(scores) / len(scores), 4) if scores else 0.0
        obj["framesSampled"] = len(obj["frames"])
        out_objects.append(obj)
    out_objects.sort(key=lambda o: (o["kind"] != "target_player", -o["maskQuality"]))

    # Compose the preview image for the segment (single frame + subject outline).
    preview = None
    if preview_dir and out_objects:
        preview = _write_segment_preview(video, segment, out_objects, meta, opts)

    return {
        "index": segment["index"],
        "start": segment["start"],
        "end": segment["end"],
        "startFrame": fs,
        "endFrame": fe,
        "reason": segment["reason"],
        "needsIsolation": True,
        "shotType": segment.get("shotType"),
        "beatRole": segment.get("beatRole"),
        "targetKinds": segment.get("targetKinds", []),
        "objects": out_objects,
        "framesSampled": sampled,
        "keyframeMaskRef": next((o["keyframeMaskRef"] for o in out_objects if o["keyframeMaskRef"]), None),
        "warnings": warnings,
    }


def _write_segment_preview(video: str, segment: Dict[str, Any], objects: List[Dict[str, Any]],
                           meta: Dict[str, Any], opts: Dict[str, Any]) -> Optional[str]:
    """Render a single inspection frame with subject outlines (QC / debugging)."""
    try:
        import cv2
        import numpy as np

        preview_dir = Path(opts["mask_preview_dir"])
        preview_dir.mkdir(parents=True, exist_ok=True)
        f = objects[0]["frames"][0]["frame"] if objects and objects[0]["frames"] else segment["startFrame"]
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f - 1))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        for color, obj in zip([(0, 255, 0), (255, 160, 0), (0, 200, 255)], objects[:3]):
            for fr in obj["frames"][:1]:
                b = fr["bbox"]
                x1, y1 = int(b[0] * meta["width"]), int(b[1] * meta["height"])
                x2, y2 = int(b[2] * meta["width"]), int(b[3] * meta["height"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, obj["kind"], (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        out = preview_dir / f"segment_{segment['index']:02d}_preview.jpg"
        write_image(str(out), frame)
        return str(out)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    deadline = float(args.deadline_seconds)
    video = str(Path(args.video).expanduser())
    if not os.path.exists(video):
        return {"success": False, "applied": False, "error": "video-not-found", "version": VERSION}

    meta = video_meta(video)
    if not meta.get("success"):
        return {"success": False, "applied": False, "error": meta.get("error", "video-unreadable"),
                "version": VERSION}

    tracking = load_json(args.detections)
    plan = load_json(args.plan) or {}
    if isinstance(plan, dict):
        plan["_events"] = (load_json(args.events) or {}).get("events", []) if args.events else []
    idx = index_detections(tracking) if tracking else {"players_by_track": {}, "ball_by_frame": {}}

    opts = {
        "stride": args.stride,
        "max_frames_per_segment": args.max_frames_per_segment,
        "max_objects_per_segment": args.max_objects_per_segment,
        "mask_resolution": args.mask_resolution,
        "mask_mode": args.mask_mode,
        "mask_preview_dir": args.mask_preview_dir,
        "subject_track_id": args.subject_track_id,
        "allow_ball": args.allow_ball,
    }

    segments = build_segments(plan, meta["fps"], meta["duration"], args.max_segments, args.force_all_segments)
    if not segments:
        return {"success": True, "applied": True, "version": VERSION, "backend": None,
                "segments": [], "segmentsProcessed": 0, "objectsTracked": 0,
                "message": "no interesting segment required subject isolation",
                "video": {"path": video, "fps": meta["fps"], "width": meta["width"],
                          "height": meta["height"], "duration": meta["duration"]},
                "elapsedSeconds": round(time.time() - started, 3)}

    # Backend lifecycle: load the model ONCE for the whole run.
    backend = create_backend(mode=os.environ.get("SAM_ADAPTER", "auto"),
                             model_path=os.environ.get("SAM_MODEL_PATH"))
    backend_name = getattr(backend, "name", "unknown")

    cache = MaskCache(Path(args.cache_dir) if args.cache_dir else (Path(__file__).resolve().parent / "cache" / "sam_masks"),
                      args.cache_max_entries)
    video_hash = video_content_hash(video)

    out_segments: List[Dict[str, Any]] = []
    objects_tracked = 0
    frames_sampled = 0
    all_warnings: List[str] = []
    for i, segment in enumerate(segments):
        if time.time() - started > deadline:
            all_warnings.append(f"deadline reached after {i} segment(s); remaining segments skipped")
            log_progress(90, "deadline reached")
            break
        targets = select_segment_targets(segment, idx, meta, opts)
        if not targets:
            continue
        log_progress(35 + int(55 * (i / max(1, len(segments)))),
                     f"segment {i + 1}/{len(segments)} ({segment['reason']})")
        seg_out = process_segment(video, segment, targets, backend, cache, meta, video_hash, opts, idx)
        objects_tracked += len(seg_out["objects"])
        frames_sampled += seg_out["framesSampled"]
        all_warnings.extend(seg_out.get("warnings", []))
        out_segments.append(seg_out)

    return {
        "success": True,
        "applied": True,
        "version": VERSION,
        "backend": backend_name,
        "video": {"path": video, "hash": video_hash, "fps": meta["fps"], "width": meta["width"],
                  "height": meta["height"], "frames": meta.get("frames", 0), "duration": meta["duration"]},
        "segmentsProcessed": len(out_segments),
        "objectsTracked": objects_tracked,
        "framesSampled": frames_sampled,
        "cache": {"hits": cache.hits, "misses": cache.misses},
        "segments": out_segments,
        "warnings": all_warnings,
        "elapsedSeconds": round(time.time() - started, 3),
    }


def check() -> Dict[str, Any]:
    """Readiness probe (no model load, no network) used by /api/sam/status."""
    avail = backend_availability()
    try:
        import cv2
        cv2_ok = True
        cv2_ver = getattr(cv2, "__version__", "?")
    except Exception:
        cv2_ok, cv2_ver = False, None
    try:
        import numpy
        np_ver = numpy.__version__
    except Exception:
        np_ver = None
    return {
        "success": True,
        "version": VERSION,
        "adapter": os.environ.get("SAM_ADAPTER", "auto"),
        "modelPathSet": bool(os.environ.get("SAM_MODEL_PATH")),
        "metaSamAvailable": bool(avail.get("meta_sam")),
        "metaBackend": avail.get("meta_backend"),
        "metaSamReason": avail.get("reason"),
        "opencv": cv2_ok,
        "opencvVersion": cv2_ver,
        "numpyVersion": np_ver,
        "python": sys.version.split()[0],
        "notes": "SAM runs in an optional isolated interpreter (SAM_PYTHON_BIN); main stack is untouched.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Optional SAM cinematic segmentation/tracking engine")
    sub = ap.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("check", help="Report engine/backend readiness")
    pc.add_argument("--json", action="store_true")

    pr = sub.add_parser("run", help="Segment + track the interesting segments")
    pr.add_argument("--video", required=True)
    pr.add_argument("--detections", default=None, help="Existing YOLO/tracking JSON")
    pr.add_argument("--plan", default=None, help="Edit plan JSON")
    pr.add_argument("--events", default=None, help="Optional event_engine events JSON")
    pr.add_argument("--output", default=None)
    pr.add_argument("--cache-dir", default=None)
    pr.add_argument("--cache-max-entries", type=int, default=int(os.environ.get("SAM_CACHE_MAX_ENTRIES", "500")))
    pr.add_argument("--mask-mode", choices=["rle", "png", "none"], default=os.environ.get("SAM_MASK_MODE", "rle"))
    pr.add_argument("--mask-resolution", type=int, default=int(os.environ.get("SAM_MASK_RESOLUTION", "512")))
    pr.add_argument("--mask-preview-dir", default=None)
    pr.add_argument("--stride", type=int, default=int(os.environ.get("SAM_STRIDE", "4")))
    pr.add_argument("--max-frames-per-segment", type=int, default=int(os.environ.get("SAM_MAX_FRAMES_PER_SEGMENT", "6")))
    pr.add_argument("--max-segments", type=int, default=int(os.environ.get("SAM_MAX_SEGMENTS", "8")))
    pr.add_argument("--max-objects-per-segment", type=int, default=int(os.environ.get("SAM_MAX_OBJECTS_PER_SEGMENT", "3")))
    pr.add_argument("--subject-track-id", default=os.environ.get("SAM_SUBJECT_TRACK_ID"))
    pr.add_argument("--allow-ball", action="store_true", default=os.environ.get("SAM_ALLOW_BALL") == "true")
    pr.add_argument("--force-all-segments", action="store_true", help="Segment every clip (debug only)")
    pr.add_argument("--deadline-seconds", type=float, default=float(os.environ.get("SAM_DEADLINE_SECONDS", "120")))
    args = ap.parse_args()

    if args.command == "check":
        result = check()
    else:
        try:
            result = run(args)
        except Exception as exc:  # never crash the pipeline
            result = {"success": False, "applied": False, "error": _clean(exc), "version": VERSION}

    if getattr(args, "output", None):
        try:
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
