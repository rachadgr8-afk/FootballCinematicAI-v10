#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path


def log_progress(pct, msg=""):
    """Emit a machine-readable progress line on stderr (non-blocking, flushed)."""
    try:
        sys.stderr.write(f"PROGRESS {int(pct)} {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--json", action="store_true")
    # Bounded work: never process more than N seconds of footage and skip frames
    # with a stride. On CPU this turns a multi-hour job into a few minutes.
    ap.add_argument("--max-seconds", type=float, default=float(os.environ.get("YOLO_MAX_SECONDS", "120")))
    ap.add_argument("--stride", type=int, default=int(os.environ.get("YOLO_STRIDE", "3")))
    # Absolute wall-clock budget for the WHOLE tracking pass. On a slow CPU-only
    # host each frame can take tens of seconds, so we must stop by time, not just
    # by frame count — otherwise the render appears frozen for hours.
    ap.add_argument("--deadline-seconds", type=float, default=float(os.environ.get("YOLO_DEADLINE_SECONDS", "90")))
    ap.add_argument("--save-video", action="store_true", default=os.environ.get("YOLO_SAVE_VIDEO") == "true")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
        import cv2
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Missing YOLO runtime: {e}"}))
        return 2

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_progress(5, "loading model")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    # Frames we intend to analyze (bounded by --max-seconds).
    max_frames = int(min(total if total > 0 else (args.max_seconds * fps), args.max_seconds * fps))
    stride = max(1, args.stride)

    log_progress(10, f"tracking up to {max_frames} frames (stride {stride})")

    annotated = out / "tracked.mp4"
    results = model.track(
        source=args.source,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        save=args.save_video,
        vid_stride=stride,
        project=str(out),
        name="render",
        exist_ok=True,
    )

    counts = {}
    samples = []
    tracks = {}
    frame_idx = 0
    analyzed = 0
    last_beat = time.time()
    started_at = time.time()
    time_exhausted = False

    for r in results:
        frame_idx += 1
        analyzed += 1

        # Heartbeat so the parent process/UI never looks frozen.
        now = time.time()
        if now - last_beat >= 2.0:
            last_beat = now
            pct = 10 + min(80, int((analyzed / max(1, max_frames)) * 80))
            log_progress(pct, f"{analyzed} frames analyzed")

        # STOP on EITHER bound: enough frames analyzed OR the wall-clock budget
        # is spent. This guarantees the pass can never run for hours.
        if analyzed >= max_frames:
            break
        if (time.time() - started_at) >= args.deadline_seconds:
            time_exhausted = True
            log_progress(90, f"time budget reached after {analyzed} frames")
            break

        names = r.names or {}
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        cls = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(xyxy)
        conf = boxes.conf.cpu().tolist() if boxes.conf is not None else [0] * len(xyxy)
        for b, tid, c, cf in zip(xyxy, ids, cls, conf):
            label = str(names.get(c, c)).lower()
            counts[label] = counts.get(label, 0) + 1
            cx = (b[0] + b[2]) / 2 / width
            cy = (b[1] + b[3]) / 2 / height
            rec = {"frame": frame_idx, "time": frame_idx / fps, "track_id": tid,
                   "class": label, "confidence": round(float(cf), 3),
                   "x": round(cx, 5), "y": round(cy, 5),
                   "w": round((b[2] - b[0]) / width, 5), "h": round((b[3] - b[1]) / height, 5),
                   "x1": round(float(b[0]) / width, 5), "y1": round(float(b[1]) / height, 5),
                   "x2": round(float(b[2]) / width, 5), "y2": round(float(b[3]) / height, 5)}
            samples.append(rec)
            if tid is not None:
                tracks.setdefault(str(tid), {"class": label, "frames": 0, "first_frame": frame_idx, "last_frame": frame_idx})
                tracks[str(tid)]["frames"] += 1
                tracks[str(tid)]["last_frame"] = frame_idx

    log_progress(92, "writing tracking json")

    video_url = None
    if args.save_video:
        candidates = list(out.rglob("*.mp4"))
        if candidates:
            best = max(candidates, key=lambda p: p.stat().st_size)
            if best != annotated:
                try:
                    best.replace(annotated)
                except Exception:
                    pass
            video_url = f"/videos/{out.name}/tracked.mp4"

    data = {"success": True,
            "videoUrl": video_url,
            "jsonUrl": f"/videos/{out.name}/tracking.json",
            "summary": {"fps": fps, "width": width, "height": height,
                        "frames": analyzed, "sourceFrames": total or frame_idx,
                        "maxSeconds": args.max_seconds, "stride": stride,
                        "timeExhausted": time_exhausted,
                        "elapsedSeconds": round(time.time() - started_at, 2),
                        "detections": counts, "samples": samples, "tracks": tracks}}
    (out / "tracking.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
    log_progress(95, "done")
    print(json.dumps(data, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
