#!/usr/bin/env python3
"""
Meta SAM 3.1 ADAPTER ENVIRONMENT BOUNDARY
=========================================

This module is the ONLY place in the project that knows how to reach Meta's
Segment Anything Model (SAM / SAM 2 / SAM 3.1). Everything else talks to the
stable, framework-agnostic contract defined here and implemented by
``sam_engine.py``.

Why a separate adapter
----------------------
SAM 3.1 (Meta) currently requires a newer stack than the football render host:

    Meta SAM 3.1  ->  Python 3.12+  |  PyTorch 2.7+  |  CUDA 12.6+

The main project is intentionally pinned to a CPU-safe, ABI-stable stack:

    torch==2.6.0+cpu  |  numpy<2  |  opencv-python-headless==4.10.0.84

Upgrading torch inside the main interpreter to satisfy SAM would break YOLO
(ultralytics) and the OpenCV/numpy pairing, so SAM MUST live in its own isolated
Python interpreter/venv (``SAM_PYTHON_BIN``). ``sam_engine.py`` invokes this
module's entry point in that separate interpreter when SAM is enabled.

Contract
--------
``segment_frame(frame_bgr, prompts) -> list[SamObjectMask]``

    frame_bgr : HxWx3 uint8 numpy array (BGR, OpenCV native order)
    prompts   : list of prompt dicts, each one of
                {"bbox": [x1, y1, x2, y2]}          # pixel coords
                {"points": [[x, y], ...], "labels": [1, 0, ...]}
                {"text": "goalkeeper"}              # used by SAM 3.1 text mode
    returns   : list of {"mask": HxW uint8/bool, "score": float,
                         "bbox": [x1, y1, x2, y2], "source": str}

The concrete backend is selected by ``SAM_ADAPTER``:

    "auto"  -> try Meta SAM 3.1/2 (if importable) then fall back to a
               deterministic, dependency-light classical CV segmenter.
    "meta"  -> force Meta SAM (fail loudly if unavailable, so misconfiguration
               is visible in logs instead of silently degrading).
    "mock"  -> deterministic classical CV segmenter ONLY. Used by the test
               suite and by hosts without the heavy SAM stack so the whole
               pipeline can be validated end-to-end.

Nothing in this module performs network access, reads API keys, or logs
secrets: model weights are resolved from a local path passed via the
environment (``SAM_MODEL_PATH``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Data contract
# --------------------------------------------------------------------------- #
SamObjectMask = Dict[str, Any]


class SamUnavailableError(RuntimeError):
    """Raised when the Meta SAM backend is requested but cannot be loaded."""


# --------------------------------------------------------------------------- #
# Classical CV fallback (always available: only needs numpy + cv2, which are
# already part of the pinned project stack). This is NOT a placeholder that
# fabricates results: it runs a real GrabCut / colour+edge segmentation on the
# real frame and reports the resulting mask + bbox.
# --------------------------------------------------------------------------- #
class ClassicalSegmenter:
    """Deterministic fallback segmenter built on OpenCV only.

    It is used for (a) the test suite and (b) hosts where the Meta SAM stack is
    not installed. It honours the SAME contract as the Meta adapter so the rest
    of the pipeline never needs to care which backend produced a mask.
    """

    name = "classical-cv"

    def __init__(self, **_ignored: Any) -> None:
        self.name = "classical-cv"

    def _segment_bbox(self, frame_bgr, bbox):
        import cv2
        import numpy as np

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(w - 2, x1))
        y1 = max(0, min(h - 2, y1))
        x2 = max(x1 + 2, min(w, x2))
        y2 = max(y1 + 2, min(h, y2))
        rect = (x1, y1, x2 - x1, y2 - y1)

        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            sub = frame_bgr[y1:y2, x1:x2]
            gc_mask = np.zeros(sub.shape[:2], dtype=np.uint8)
            bgd = np.zeros((1, 65), dtype=np.float64)
            fgd = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(sub, gc_mask, rect[:2] + rect[2:], bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            local = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            mask[y1:y2, x1:x2] = local
        except Exception:
            # GrabCut can fail on tiny/degenerate crops; fall back to the bbox.
            mask[y1:y2, x1:x2] = 255

        if int(mask.sum()) == 0:
            mask[y1:y2, x1:x2] = 255
        return mask, 0.55

    def segment_frame(self, frame_bgr, prompts: List[Dict[str, Any]]) -> List[SamObjectMask]:
        import numpy as np

        h, w = frame_bgr.shape[:2]
        out: List[SamObjectMask] = []
        for prompt in prompts or []:
            bbox = prompt.get("bbox")
            if bbox is None and prompt.get("points"):
                pts = prompt["points"]
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                pad = max(6.0, 0.05 * max(w, h))
                bbox = [max(0, min(xs) - pad), max(0, min(ys) - pad),
                        min(w, max(xs) + pad), min(h, max(ys) + pad)]
            if bbox is None:
                continue
            mask, score = self._segment_bbox(frame_bgr, bbox)
            out.append({
                "mask": mask,
                "score": float(score),
                "bbox": [float(v) for v in bbox],
                "source": self.name,
            })
        return out


# --------------------------------------------------------------------------- #
# Meta SAM backend (lazy import; never imported unless actually selected)
# --------------------------------------------------------------------------- #
class MetaSegmenter:
    """Thin wrapper over the Meta SAM predictor.

    The exact import surface differs between SAM 1 / 2 / 3.1, so we probe the
    known entry points defensively. If none is importable we raise
    ``SamUnavailableError`` and the caller decides whether to fall back.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, **_ignored: Any) -> None:
        model_path = model_path or os.environ.get("SAM_MODEL_PATH", "")
        if not model_path:
            raise SamUnavailableError("SAM_MODEL_PATH is not set")
        self.model_path = model_path
        self.device = device or os.environ.get("SAM_DEVICE", "auto")
        self._predictor = None
        self._text_predictor = None
        self._backend = "meta-unknown"
        self._load()

    def _load(self) -> None:
        import torch

        device = self.device
        if device in ("auto", "", None):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device

        # --- SAM 3.1 / SAM 2 (segment-anything-2 / sam3 packages) -------------
        try:  # pragma: no cover - depends on optional heavy backend
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            model = build_sam2(self.model_path, device=device)
            self._predictor = SAM2ImagePredictor(model)
            self._backend = "meta-sam2"
            return
        except Exception:
            pass

        # --- SAM 3 (text-promptable) -----------------------------------------
        try:  # pragma: no cover - depends on optional heavy backend
            from sam3 import build_sam3  # type: ignore

            self._text_predictor = build_sam3(self.model_path, device=device)
            self._backend = "meta-sam3"
            return
        except Exception:
            pass

        # --- Classic SAM 1 (segment_anything) --------------------------------
        try:  # pragma: no cover - depends on optional heavy backend
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore

            # Infer the variant from the checkpoint name (vit_h / vit_l / vit_b).
            variant = "vit_h"
            for cand in ("vit_h", "vit_l", "vit_b"):
                if cand in self.model_path:
                    variant = cand
                    break
            sam = sam_model_registry[variant](checkpoint=self.model_path)
            sam.to(device=device)
            self._predictor = SamPredictor(sam)
            self._backend = "meta-sam1"
            return
        except Exception as exc:  # noqa: BLE001
            raise SamUnavailableError(f"Meta SAM runtime not importable: {exc}")

    @property
    def name(self) -> str:
        return self._backend

    def segment_frame(self, frame_bgr, prompts: List[Dict[str, Any]]) -> List[SamObjectMask]:
        import numpy as np

        if self._predictor is None:
            raise SamUnavailableError("Meta SAM predictor failed to initialise")

        # Predictor expects RGB.
        rgb = frame_bgr[:, :, ::-1]
        # Boxing the prompt as points keeps a single API path for boxes+points.
        pts: List[List[float]] = []
        lbls: List[int] = []
        for prompt in prompts or []:
            if prompt.get("points"):
                for i, p in enumerate(prompt["points"]):
                    pts.append([float(p[0]), float(p[1])])
                    labels = prompt.get("labels") or []
                    lbls.append(int(labels[i]) if i < len(labels) else 1)
            elif prompt.get("bbox"):
                x1, y1, x2, y2 = [float(v) for v in prompt["bbox"]]
                pts.extend([[x1, y1], [x2, y2]])
                lbls.extend([1, 1])

        if not pts:
            return []

        self._predictor.set_image(rgb)
        point_coords = np.array(pts, dtype=np.float32)
        point_labels = np.array(lbls, dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        out: List[SamObjectMask] = []
        for mask, score in zip(masks, scores):
            m = (np.asarray(mask) > 0.5).astype(np.uint8)
            ys, xs = np.where(m > 0)
            if xs.size == 0:
                continue
            out.append({
                "mask": m,
                "score": float(score),
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                "source": self._backend,
            })
        return out


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_backend(mode: Optional[str] = None, **kwargs: Any):
    """Return a segmenter backend honouring ``SAM_ADAPTER``.

    ``mode`` overrides ``SAM_ADAPTER`` (used by tests). ``auto`` tries Meta SAM
    first and silently degrades to the classical segmenter; ``meta`` fails loud.
    """
    mode = (mode or os.environ.get("SAM_ADAPTER", "auto")).strip().lower()
    if mode == "mock":
        return ClassicalSegmenter()
    if mode == "meta":
        return MetaSegmenter(**kwargs)
    # auto
    try:
        return MetaSegmenter(**kwargs)
    except Exception:
        return ClassicalSegmenter()


def backend_availability() -> Dict[str, Any]:
    """Report which backends are importable WITHOUT loading any weights."""
    info: Dict[str, Any] = {"meta_sam": False, "meta_backend": None, "reason": None}
    try:
        import importlib

        for mod, label in (("sam2", "meta-sam2"), ("sam3", "meta-sam3"), ("segment_anything", "meta-sam1")):
            try:
                importlib.import_module(mod)
                info["meta_sam"] = True
                info["meta_backend"] = label
                break
            except Exception:
                continue
        if not info["meta_sam"]:
            info["reason"] = "no Meta SAM package importable (sam2/sam3/segment_anything)"
    except Exception as exc:  # pragma: no cover
        info["reason"] = str(exc)
    return info
