#!/usr/bin/env python3
"""Media and pipeline quality gate for production outputs."""
import argparse, json, subprocess, cv2, math
from pathlib import Path

def probe(path):
    p=subprocess.run(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)],
                     capture_output=True,text=True)
    if p.returncode!=0: raise RuntimeError(p.stderr[-500:])
    return json.loads(p.stdout)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--min-duration",type=float,default=.5)
    a=ap.parse_args()
    path=Path(a.video)
    report={"success":False,"video":str(path),"checks":{}}
    try:
        meta=probe(path)
        streams=meta.get("streams",[])
        vs=next((s for s in streams if s.get("codec_type")=="video"),None)
        aud=next((s for s in streams if s.get("codec_type")=="audio"),None)
        duration=float(meta.get("format",{}).get("duration") or 0)
        report["checks"]={
            "exists":path.exists(),"non_empty":path.exists() and path.stat().st_size>10000,
            "duration_ok":duration>=a.min_duration,
            "video_stream":bool(vs),"audio_stream":bool(aud),
            "width":int(vs.get("width",0)) if vs else 0,
            "height":int(vs.get("height",0)) if vs else 0,
            "fps":vs.get("r_frame_rate") if vs else None,
            "duration":round(duration,3)
        }
        # Lightweight decode smoke test at beginning/middle/end.
        cap=cv2.VideoCapture(str(path))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        positions=sorted(set([0,max(0,n//2),max(0,n-1)]))
        decoded=0
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
            ok,frame=cap.read()
            if ok and frame is not None and frame.size: decoded+=1
        cap.release()
        report["checks"]["decode_samples"]=decoded
        report["checks"]["decode_ok"]=decoded==len(positions)
        report["success"]=all([
            report["checks"]["exists"],report["checks"]["non_empty"],
            report["checks"]["duration_ok"],report["checks"]["video_stream"],
            report["checks"]["decode_ok"]
        ])
    except Exception as e:
        report["error"]=str(e)
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report["success"] else 2
if __name__=="__main__": raise SystemExit(main())
