#!/usr/bin/env python3
"""
video_engine.py — CinematicEngine
================================================================================
A self-contained, memory-safe cinematographic engine built ONLY on
OpenCV + NumPy. It is designed to run on Render's free/entry containers where
the resident memory budget is ~512MB.

HARD RULES (do not break):
  * NO MoviePy.
  * NO FFmpeg inside this module (no subprocess, no `ffmpeg` shell-out).
  * Strictly frame-by-frame streaming — the full video is NEVER loaded into RAM.
  * Exactly ONE decoded frame pair (+ its flow + remap maps) is alive at a time.
  * All OpenCV handles (VideoCapture / VideoWriter) are released even on error.

Public surface:
    class CinematicEngine
        .slow_motion_optical_flow(input_path, output_path,
                                  slow_factor=0.25,
                                  target_segment=(2.0, 4.0))
        .add_cinematic_lut(input_path, output_path, intensity=0.7)

The module also exposes a small CLI so the Node render pipeline can invoke it as
a subprocess and read a JSON result on stdout:

    python3 video_engine.py slow-motion --input a.mp4 --output b.mp4 \
        --slow-factor 0.25 --seg-start 2.0 --seg-end 4.0
    python3 video_engine.py lut --input a.mp4 --output c.mp4 --intensity 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Codec used for every OpenCV write. `mp4v` is bundled with OpenCV (FFmpeg is
# the underlying decoder library used by the OpenCV build, but this module never
# calls the ffmpeg CLI, never spawns a process and never imports MoviePy).
_OUTPUT_FOURCC = "mp4v"

# Optical-flow working resolution. Farneback at this width is ~20x cheaper than
# at 1080p and the resulting flow is upscaled back to the real frame size. This
# keeps a 512MB instance comfortable while preserving a smooth warp.
_FLOW_WIDTH = 192


def _validate_segment(target_segment) -> "tuple[float, float]":
    """Normalise + validate a (start, end) slow-motion window in seconds."""
    if target_segment is None:
        raise ValueError("target_segment must be a (start, end) pair, got None")
    try:
        start, end = float(target_segment[0]), float(target_segment[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"target_segment must be a (start, end) pair of numbers: {exc}"
        ) from exc
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("target_segment values must be finite numbers")
    if start < 0:
        raise ValueError("target_segment start must be >= 0")
    if end <= start:
        raise ValueError("target_segment end must be greater than start")
    return start, end


def _read_fps(cap: "cv2.VideoCapture") -> float:
    """Read a sane FPS, falling back to 25.0 when the container reports junk."""
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        fps = 0.0
    if not (fps > 0.5 and fps < 240.0):
        fps = 25.0
    return fps


class CinematicEngine:
    """Frame-by-frame cinematographic processor (OpenCV + NumPy only)."""

    # ------------------------------------------------------------------
    # A) SMART SLOW MOTION via dense optical flow
    # ------------------------------------------------------------------
    def slow_motion_optical_flow(
        self,
        input_path: str,
        output_path: str,
        slow_factor: float = 0.25,
        target_segment=(2.0, 4.0),
    ) -> dict:
        """
        Apply slow motion ONLY inside `target_segment`, interpolating the missing
        in-between frames with dense optical flow (cv2.calcOpticalFlowFarneback)
        and warping them into place with cv2.remap.

        Parameters
        ----------
        input_path : str
            Source video. Read one frame at a time (never fully buffered).
        output_path : str
            Destination .mp4 (codec mp4v).
        slow_factor : float
            Playback factor in (0, 1]. 0.25 -> 4x slower -> 3 interpolated frames
            are synthesised between every real pair inside the segment.
        target_segment : (float, float)
            Inclusive-ish (start, end) window in seconds where the effect applies.

        Returns
        -------
        dict with frame/fps/size stats.
        """
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"input video not found: {input_path}")
        if not output_path:
            raise ValueError("output_path is required")

        # Validate the slow factor BEFORE opening any handle.
        try:
            sf = float(slow_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"slow_factor must be a number: {exc}") from exc
        if not np.isfinite(sf) or sf <= 0.0 or sf > 1.0:
            raise ValueError("slow_factor must be in the range (0, 1]")

        seg_start, seg_end = _validate_segment(target_segment)

        # slow_factor == 1.0 -> nothing to interpolate (pass-through copy).
        num_interpolated = int(round(1.0 / sf)) - 1
        if num_interpolated < 0:
            num_interpolated = 0

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cap: "cv2.VideoCapture | None" = None
        writer: "cv2.VideoWriter | None" = None
        written = 0
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"could not open input video: {input_path}")

            fps = _read_fps(cap)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError(
                    f"invalid video dimensions reported by decoder: {width}x{height}"
                )

            fourcc = cv2.VideoWriter_fourcc(*_OUTPUT_FOURCC)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"could not create VideoWriter for: {output_path}")

            # Base sampling grid for cv2.remap (built once, reused for every frame).
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )

            # Flow is computed on a downscaled gray pair; these are the scale
            # factors used to blow the flow back up to full resolution.
            flow_w = min(_FLOW_WIDTH, width)
            flow_h = max(1, int(round(flow_w * height / float(width))))
            scale_x = width / float(flow_w)
            scale_y = height / float(flow_h)

            prev_frame: "np.ndarray | None" = None
            prev_flow_gray: "np.ndarray | None" = None
            idx = 0

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                t_now = idx / fps

                if prev_frame is None:
                    # First decoded frame: emit it as-is and remember it.
                    writer.write(frame)
                    written += 1
                    prev_frame = frame
                    prev_flow_gray = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        (flow_w, flow_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    idx += 1
                    continue

                in_segment = (t_now <= seg_end) and ((idx - 1) / fps >= seg_start)

                if in_segment and num_interpolated > 0:
                    cur_gray = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        (flow_w, flow_h),
                        interpolation=cv2.INTER_AREA,
                    )

                    # Cheap dense flow (downscaled), then upscale to full res.
                    flow_small = cv2.calcOpticalFlowFarneback(
                        prev_flow_gray,
                        cur_gray,
                        None,
                        pyr_scale=0.5,
                        levels=2,
                        winsize=13,
                        iterations=2,
                        poly_n=5,
                        poly_sigma=1.1,
                        flags=0,
                    )
                    flow = cv2.resize(
                        flow_small, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                    flow[..., 0] *= scale_x
                    flow[..., 1] *= scale_y

                    # Synthesise the in-between frames: warp the PREVIOUS frame
                    # forward along the flow by a fraction t in (0, 1).
                    for k in range(1, num_interpolated + 1):
                        frac = k / float(num_interpolated + 1)
                        map_x = grid_x + flow[..., 0] * frac
                        map_y = grid_y + flow[..., 1] * frac
                        interp = cv2.remap(
                            prev_frame,
                            map_x,
                            map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
                        writer.write(interp)
                        written += 1
                        del interp

                    del flow
                    del flow_small
                    del cur_gray
                    del map_x, map_y
                    prev_flow_gray = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        (flow_w, flow_h),
                        interpolation=cv2.INTER_AREA,
                    )

                # Always emit the real current frame.
                writer.write(frame)
                written += 1
                prev_frame = frame
                idx += 1

            del prev_frame, prev_flow_gray, grid_x, grid_y

            if written == 0:
                raise RuntimeError("no frames were decoded from the input video")

            return {
                "success": True,
                "engine": "CinematicEngine.slow_motion_optical_flow",
                "frames_written": written,
                "fps": round(fps, 5),
                "width": width,
                "height": height,
                "num_interpolated": num_interpolated,
                "slow_factor": sf,
                "target_segment": [seg_start, seg_end],
                "output_path": os.path.abspath(output_path),
            }
        finally:
            # Guaranteed release on the happy path AND on any exception.
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # B) CINEMATIC LUT (CLAHE on LAB-L + Teal & Orange grade)
    # ------------------------------------------------------------------
    def add_cinematic_lut(
        self,
        input_path: str,
        output_path: str,
        intensity: float = 0.7,
    ) -> dict:
        """
        Apply a cinematic look frame-by-frame:
          1. CLAHE on the L channel in LAB space (local contrast).
          2. Teal & Orange split-tone color grade.
          3. Linear blend with the original via cv2.addWeighted by `intensity`.

        Parameters
        ----------
        intensity : float
            Blend weight of the graded image, in [0, 1].
        """
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"input video not found: {input_path}")
        if not output_path:
            raise ValueError("output_path is required")

        try:
            amount = float(intensity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"intensity must be a number: {exc}") from exc
        if not np.isfinite(amount) or amount < 0.0 or amount > 1.0:
            raise ValueError("intensity must be in the range [0, 1]")

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cap: "cv2.VideoCapture | None" = None
        writer: "cv2.VideoWriter | None" = None
        written = 0
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"could not open input video: {input_path}")

            fps = _read_fps(cap)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError(
                    f"invalid video dimensions reported by decoder: {width}x{height}"
                )

            fourcc = cv2.VideoWriter_fourcc(*_OUTPUT_FOURCC)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"could not create VideoWriter for: {output_path}")

            # CLAHE object is stateless across frames and cheap to reuse.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                graded = self._grade_frame(frame, clahe)

                if amount >= 1.0:
                    out = graded
                elif amount <= 0.0:
                    out = frame.copy()
                else:
                    # out = (1-amount)*original + amount*graded
                    out = cv2.addWeighted(frame, 1.0 - amount, graded, amount, 0.0)

                writer.write(out)
                written += 1

                del graded, out

            if written == 0:
                raise RuntimeError("no frames were decoded from the input video")

            return {
                "success": True,
                "engine": "CinematicEngine.add_cinematic_lut",
                "frames_written": written,
                "fps": round(fps, 5),
                "width": width,
                "height": height,
                "intensity": amount,
                "output_path": os.path.abspath(output_path),
            }
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal: single-frame cinematic grade
    # ------------------------------------------------------------------
    @staticmethod
    def _grade_frame(frame: "np.ndarray", clahe: "cv2.CLAHE") -> "np.ndarray":
        """Apply CLAHE(L in LAB) + Teal & Orange split-tone to one BGR frame."""
        # 1) CLAHE on the L channel of LAB (memory: one frame copy).
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge((l_chan, a_chan, b_chan))
        contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        del lab, l_chan, a_chan, b_chan

        # 2) Teal & Orange split-tone. Work in float32 for the masks, then clip.
        img = contrast.astype(np.float32)
        # Perceptual luminance (Rec. 601) normalised to 0..1.
        lum = (
            0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
        ) / 255.0
        lum = np.clip(lum, 0.0, 1.0)

        # Shadows -> teal (raise B/G, pull R), highlights -> orange (raise R/G,
        # pull B). Soft falloff via smoothstep-ish curves keeps skin tones sane.
        shadow_w = (1.0 - lum) ** 2  # strong in shadows
        high_w = lum ** 2            # strong in highlights

        teal = np.array([26.0, 14.0, -16.0], dtype=np.float32)   # B, G, R
        warm = np.array([-18.0, 6.0, 22.0], dtype=np.float32)     # B, G, R

        img[..., 0] += teal[0] * shadow_w + warm[0] * high_w
        img[..., 1] += teal[1] * shadow_w + warm[1] * high_w
        img[..., 2] += teal[2] * shadow_w + warm[2] * high_w

        # Slight contrast S-curve around mid-grey to firm up the look.
        img = (img - 128.0) * 1.06 + 128.0

        graded = np.clip(img, 0.0, 255.0).astype(np.uint8)
        del img, lum, shadow_w, high_w
        return graded


# ---------------------------------------------------------------------------
# CLI (used by the Node pipeline as a subprocess; JSON result on stdout)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CinematicEngine CLI — OpenCV + NumPy only (no FFmpeg, no MoviePy)"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_slow = sub.add_parser("slow-motion", help="optical-flow slow motion on a segment")
    p_slow.add_argument("--input", required=True)
    p_slow.add_argument("--output", required=True)
    p_slow.add_argument("--slow-factor", type=float, default=0.25)
    p_slow.add_argument("--seg-start", type=float, default=2.0)
    p_slow.add_argument("--seg-end", type=float, default=4.0)

    p_lut = sub.add_parser("lut", help="cinematic CLAHE + teal&orange LUT")
    p_lut.add_argument("--input", required=True)
    p_lut.add_argument("--output", required=True)
    p_lut.add_argument("--intensity", type=float, default=0.7)

    return ap


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    engine = CinematicEngine()
    try:
        if args.command == "slow-motion":
            result = engine.slow_motion_optical_flow(
                args.input,
                args.output,
                slow_factor=args.slow_factor,
                target_segment=(args.seg_start, args.seg_end),
            )
        else:
            result = engine.add_cinematic_lut(
                args.input, args.output, intensity=args.intensity
            )
    except Exception as exc:  # always return valid JSON so the caller can parse
        print(json.dumps({"success": False, "error": str(exc)}))
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
