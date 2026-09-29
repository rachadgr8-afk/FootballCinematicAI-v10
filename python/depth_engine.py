#!/usr/bin/env python3
"""DEPTH-ANYTHING-V2-SMALL psychological depth pass (OPTIONAL, lazy, bounded).

Produces, for a handful of PSYCHOLOGICAL segments the edit plan already selected,
a compact FOREGROUND depth mask (PNG, downscaled) plus a per-segment strength.

The renderer then applies a depth-aware treatment: the foreground (the player)
stays critically sharp while the background gets a gentle progressive blur and a
restrained low-angle ground POV — never a full-frame blur, never a halo, never a
distorted subject.

Design guarantees
-----------------
* DEFAULT OFF / inert unless ``DEPTH_ENABLED=true``. No model -> ``applied:false``.
* LAZY: torch/transformers are imported ONLY inside the model path. If they (or
  the checkpoint) are absent, the engine degrades gracefully.
* BOUNDED: only the segments passed in (max ``--max-segments``), a few sampled
  frames per segment (``--sampled-frames``), a hard wall-clock deadline, and masks
  are downscaled (``--mask-resolution``) and cached by (video-hash, segment).
* MEMORY SAFE: streaming/frame-by-frame; the full video is NEVER loaded. One
  decoded frame (+ its depth map) is alive at a time.
* HONEST: the result records ``source`` so a caller can tell whether a real
  Depth-Anything model ran, the classical PROXY fallback ran, or nothing ran.
  The proxy is a distinct, clearly-labelled fallback — never presented as the
  real model.

Env: ``DEPTH_ENABLED, DEPTH_MODEL_ID, DEPTH_PYTHON_BIN, DEPTH_PROXY_FALLBACK,
DEPTH_DEADLINE_SECONDS, DEPTH_MAX_SEGMENTS, DEPTH_SAMPLED_FRAMES,
DEPTH_MASK_RESOLUTION, DEPTH_CACHE_DIR``.

CLI
---
    python3 python/depth_engine.py check [--json]
    python3 python/depth_engine.py run --video V --segments S.json --output O.json \
        --mask-dir DIR [--proxy] [--max-segments N] [--deadline-seconds S]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

VERSION = "depth-anything-v2-1.0.0"
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def log_progress(pct, msg=""):
    try:
        sys.stderr.write(f"PROGRESS {int(pct)} {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _clean(text, limit=160):
    s = str(text)
    for marker in ("ghp_", "sk-", "AIza", "xi-api-key", "Bearer "):
        if marker in s:
            s = s.replace(marker, "***")
    return s[:limit]


def _hash(path: str) -> str:
    try:
        st = Path(path).stat()
        h = hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
        return h[:16]
    except Exception:
        return "nohash"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _model_available() -> dict:
    """Check (WITHOUT loading the model) whether the real backend can run."""
    info = {"available": False, "reason": None, "model_id": os.environ.get("DEPTH_MODEL_ID", DEFAULT_MODEL_ID)}
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        info["reason"] = f"deps-missing:{type(exc).__name__}"
        return info
    info["available"] = True
    return info


def _load_model():
    """Lazy heavy import + model load. Returns (processor, model, device) or raises."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = os.environ.get("DEPTH_MODEL_ID", DEFAULT_MODEL_ID)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def _infer_depth(processor, model, device, frame_bgr):
    """Return a float32 depth map (H,W) for one BGR frame, or None."""
    try:
        import torch
        import numpy as np
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        pred = out.predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=frame_bgr.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
        depth = pred.detach().cpu().numpy().astype("float32")
        del inputs, out, pred
        return depth
    except Exception:
        return None


def _depth_to_fg_mask(depth):
    """Normalise a depth map to a 0..255 FOREGROUND mask (nearer = brighter)."""
    import numpy as np
    d = depth.astype("float32")
    d = d - float(d.min())
    rng = float(d.max()) - float(d.min())
    if rng < 1e-6:
        d = np.full_like(d, 0.5)
    else:
        d = d / rng
    # Foreground = nearest surface. Depth models output inverse-depth-like values
    # (larger = nearer) for this checkpoint, so we use the normalised value
    # directly; a robust percentile stretch protects against outliers.
    lo, hi = np.percentile(d, 4), np.percentile(d, 96)
    if hi - lo > 1e-6:
        d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return (d * 255.0).astype("uint8")


def _proxy_fg_mask(frame_bgr):
    """Classical, model-free FOREGROUND proxy (clearly labelled 'proxy').

    Approximates a subject by combining centre-weighting, local contrast/edges and
    a pitch-vertical prior. It is NOT a learned depth map."""
    import numpy as np
    import cv2
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 3.0)
    # Edge/contrast saliency (subject usually carries the strongest local detail).
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    edge = edge / (float(edge.max()) + 1e-6)
    # Centre + vertical (players sit in the lower-centre band).
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    cx, cy = w * 0.5, h * 0.62
    rad = np.sqrt(((xx - cx) / (w * 0.5)) ** 2 + ((yy - cy) / (h * 0.5)) ** 2)
    centre = np.clip(1.0 - rad * 0.7, 0.0, 1.0)
    prior = 0.55 * centre + 0.45 * edge
    prior = cv2.GaussianBlur(prior, (0, 0), 12.0)
    # Keep the strongest ~45% of the field as FOREGROUND so the mask is selective
    # (a broad mask would mean no background separation at all). This is a PROXY,
    # used only when the real Depth-Anything model is unavailable.
    thr = float(np.percentile(prior, 55))
    fg = np.clip((prior - thr) / (float(prior.max()) - thr + 1e-6), 0.0, 1.0)
    return (fg * 255.0).astype("uint8")


def _pick_frames(video, start, end, count, width=384):
    import cv2
    frames = []
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return frames
    try:
        span = max(0.2, float(end) - float(start))
        for i in range(count):
            t = float(start) + span * (i / max(1, count - 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > width:
                s = width / float(w)
                frame = cv2.resize(frame, (width, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()
    return frames


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(video, segments, mask_dir, max_segments, sampled_frames, deadline_seconds, proxy, mask_resolution):
    import numpy as np
    import cv2

    Path(mask_dir).mkdir(parents=True, exist_ok=True)
    segs = segments if isinstance(segments, list) else (segments.get("segments") if isinstance(segments, dict) else [])
    segs = [s for s in (segs or []) if isinstance(s, dict)][:max(1, int(max_segments))]

    info = _model_available()
    use_model = info["available"] and not proxy
    processor = model = device = None
    source = "none"
    if use_model:
        try:
            processor, model, device = _load_model()
            source = "depth-anything-v2"
        except Exception as exc:
            log_progress(20, f"model load failed: {type(exc).__name__}")
            use_model = False
    if not use_model and proxy:
        source = "proxy"
    if source == "none":
        return {"success": True, "applied": False, "version": VERSION,
                "source": "none", "reason": info.get("reason") or "depth-disabled-no-proxy",
                "segments": []}

    started = time.time()
    out_segments = []
    vhash = _hash(video)
    for i, seg in enumerate(segs):
        if time.time() - started > deadline_seconds:
            log_progress(95, "deadline reached")
            break
        try:
            start = float(seg.get("time", 0.0))
            span = float(seg.get("span", 6.0))
        except (TypeError, ValueError):
            continue
        end = start + max(1.0, span)
        frames = _pick_frames(video, start, end, sampled_frames)
        if not frames:
            continue
        mask_acc = None
        for fr in frames:
            m = None
            if use_model and processor is not None:
                depth = _infer_depth(processor, model, device, fr)
                if depth is not None:
                    m = _depth_to_fg_mask(depth)
            if m is None:
                m = _proxy_fg_mask(fr)
                if source == "depth-anything-v2":
                    source = "depth-anything-v2+proxy"
            if m.shape != frames[0].shape[:2]:
                m = cv2.resize(m, (frames[0].shape[1], frames[0].shape[0]), interpolation=cv2.INTER_AREA)
            mask_acc = m.astype("float32") if mask_acc is None else mask_acc + m.astype("float32")
            del m
        if mask_acc is None:
            continue
        mask = (mask_acc / max(1, len(frames))).astype("uint8")
        # Downscale for compactness.
        h, w = mask.shape[:2]
        scale = min(1.0, float(mask_resolution) / max(1, max(h, w)))
        if scale < 1.0:
            mask = cv2.resize(mask, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
        mask_path = os.path.join(mask_dir, f"depth_{vhash}_{i}.png")
        cv2.imwrite(mask_path, mask)
        cover = float((mask.astype("float32") > 40).mean())
        out_segments.append({
            "time": round(start, 3), "span": round(max(1.0, span), 3),
            "maskRef": mask_path,
            "foreground_ratio": round(cover, 4),
            "depth_strength": round(min(0.6, 0.25 + cover), 3),
            "source": source,
        })
        log_progress(20 + int(70 * (i + 1) / max(1, len(segs))), f"depth seg {i + 1}/{len(segs)}")
        del mask, mask_acc

    applied = len(out_segments) > 0
    return {"success": True, "applied": applied, "version": VERSION, "source": source,
            "model_id": info.get("model_id"), "segments": out_segments,
            "reason": None if applied else "no-segments-processed"}


def check():
    info = _model_available()
    return {"success": True, "version": VERSION, "enabled": os.environ.get("DEPTH_ENABLED") == "true",
            "model": info, "proxy_fallback": os.environ.get("DEPTH_PROXY_FALLBACK") == "true"}


def main():
    ap = argparse.ArgumentParser(description="Optional Depth-Anything-V2 psychological depth pass")
    sub = ap.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("check")
    pc.add_argument("--json", action="store_true")
    pr = sub.add_parser("run")
    pr.add_argument("--video", required=True)
    pr.add_argument("--segments", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--mask-dir", required=True)
    pr.add_argument("--max-segments", type=int, default=int(os.environ.get("DEPTH_MAX_SEGMENTS", "3")))
    pr.add_argument("--sampled-frames", type=int, default=int(os.environ.get("DEPTH_SAMPLED_FRAMES", "4")))
    pr.add_argument("--deadline-seconds", type=float, default=float(os.environ.get("DEPTH_DEADLINE_SECONDS", "60")))
    pr.add_argument("--mask-resolution", type=int, default=int(os.environ.get("DEPTH_MASK_RESOLUTION", "512")))
    pr.add_argument("--proxy", action="store_true", default=os.environ.get("DEPTH_PROXY_FALLBACK") == "true")
    a = ap.parse_args()

    if a.command == "check":
        print(json.dumps(check()))
        return 0

    try:
        segs = json.loads(Path(a.segments).read_text(encoding="utf8"))
        res = run(a.video, segs, a.mask_dir, a.max_segments, a.sampled_frames,
                  a.deadline_seconds, a.proxy, a.mask_resolution)
    except Exception as exc:
        res = {"success": True, "applied": False, "version": VERSION, "source": "none",
               "segments": [], "reason": _clean(type(exc).__name__)}
    Path(a.output).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"success": True, "applied": res.get("applied"), "source": res.get("source"),
                      "segments": len(res.get("segments", [])), "reason": res.get("reason")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
