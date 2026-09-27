#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--json", action="store_true")
    args=ap.parse_args()
    try:
        from ultralytics import YOLO
        import cv2
    except Exception as e:
        print(json.dumps({"success":False,"error":f"Missing YOLO runtime: {e}"}))
        return 2

    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model=YOLO(args.model)
    cap=cv2.VideoCapture(args.source)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    annotated=out/"tracked.mp4"
    # Ultralytics writes annotated frames while preserving the source timing.
    results=model.track(source=args.source, persist=True, tracker="bytetrack.yaml",
                        stream=True, verbose=False, save=True, project=str(out),
                        name="render", exist_ok=True)

    counts={}
    samples=[]
    tracks={}
    frame_idx=0
    for r in results:
        frame_idx += 1
        names=r.names or {}
        boxes=getattr(r,"boxes",None)
        if boxes is None: continue
        xyxy=boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        ids=boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(xyxy)
        cls=boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0]*len(xyxy)
        conf=boxes.conf.cpu().tolist() if boxes.conf is not None else [0]*len(xyxy)
        for b,tid,c,cf in zip(xyxy,ids,cls,conf):
            label=str(names.get(c,c)).lower()
            counts[label]=counts.get(label,0)+1
            cx=(b[0]+b[2])/2/width; cy=(b[1]+b[3])/2/height
            rec={"frame":frame_idx,"time":frame_idx/fps,"track_id":tid,
                 "class":label,"confidence":round(float(cf),3),
                 "x":round(cx,5),"y":round(cy,5),
                 "w":round((b[2]-b[0])/width,5),"h":round((b[3]-b[1])/height,5),
                 "x1":round(float(b[0])/width,5),"y1":round(float(b[1])/height,5),
                 "x2":round(float(b[2])/width,5),"y2":round(float(b[3])/height,5)}
            samples.append(rec)
            if tid is not None:
                tracks.setdefault(str(tid),{"class":label,"frames":0,"first_frame":frame_idx,"last_frame":frame_idx})
                tracks[str(tid)]["frames"]+=1; tracks[str(tid)]["last_frame"]=frame_idx

    # Locate Ultralytics saved MP4 robustly.
    candidates=list(out.rglob("*.mp4"))
    if candidates:
        best=max(candidates,key=lambda p:p.stat().st_size)
        if best != annotated:
            best.replace(annotated)
    data={"success":True,"videoUrl":f"/videos/{out.name}/tracked.mp4",
          "jsonUrl":f"/videos/{out.name}/tracking.json",
          "summary":{"fps":fps,"width":width,"height":height,"frames":total or frame_idx,
                     "detections":counts,"samples":samples,"tracks":tracks}}
    (out/"tracking.json").write_text(json.dumps(data,ensure_ascii=False),encoding="utf8")
    print(json.dumps(data,ensure_ascii=False))
    return 0

if __name__=="__main__":
    sys.exit(main())
