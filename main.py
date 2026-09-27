#!/usr/bin/env python3
"""Football Cinematic AI — Exceptional Football Cinematic Intelligence pipeline."""
import argparse, json, os, shutil, subprocess, sys, time, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def run(cmd,cwd=ROOT):
    print("$"," ".join(map(str,cmd)),flush=True)
    p=subprocess.run(cmd,cwd=str(cwd))
    if p.returncode!=0: raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str,cmd))}")
def atomic_json(path,data):
    tmp=Path(str(path)+".tmp"); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8"); tmp.replace(path)
def fingerprint(video,model,options):
    h=hashlib.sha256()
    for p in (video,model):
        st=Path(p).stat(); h.update(str(Path(p).resolve()).encode()); h.update(str(st.st_size).encode()); h.update(str(st.st_mtime_ns).encode())
    h.update(json.dumps(options,sort_keys=True).encode())
    return h.hexdigest()[:20]
def rife(inp,out,exp):
    repo=Path(os.getenv("RIFE_REPO","/opt/rife")); script=repo/"inference_video.py"
    if not script.exists(): raise RuntimeError(f"RIFE not installed: {script}")
    before={x.resolve() for x in repo.rglob("*.mp4")}
    subprocess.run([sys.executable,str(script),"--exp",str(exp),"--video",str(Path(inp).resolve())],cwd=str(repo),check=True)
    after=[x for x in repo.rglob("*.mp4") if x.resolve() not in before and x.stat().st_size>10000]
    if not after:
        after=[x for x in Path(inp).parent.rglob("*.mp4") if x.resolve()!=Path(inp).resolve() and x.stat().st_size>10000]
    if not after: raise RuntimeError("RIFE finished but no interpolated MP4 was found.")
    shutil.copy2(max(after,key=lambda x:x.stat().st_mtime),out)
def main():
    ap=argparse.ArgumentParser(description="Football Cinematic AI Exceptional")
    ap.add_argument("--video",required=True); ap.add_argument("--player",required=True)
    ap.add_argument("--model",default="models/best.pt"); ap.add_argument("--output",default=None)
    ap.add_argument("--track",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--cinematic",action="store_true")
    ap.add_argument("--rife",choices=["off","2x","4x"],default="off")
    ap.add_argument("--export-json",action="store_true")
    ap.add_argument("--player-reference",action="append",default=None)
    ap.add_argument("--highlights",action="store_true"); ap.add_argument("--highlight-count",type=int,default=8)
    ap.add_argument("--scene-aware-rife",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--reid-min-score",type=float,default=.72); ap.add_argument("--reid-min-margin",type=float,default=.035)
    ap.add_argument("--force",action="store_true",help="Ignore cached tracking result")
    args=ap.parse_args()
    video=Path(args.video).expanduser().resolve()
    model=(ROOT/args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    if not video.exists(): raise SystemExit(f"Video not found: {video}")
    if not model.exists(): raise SystemExit(f"YOLO model not found: {model}")
    slug="".join(c if c.isalnum() or c in "-_" else "_" for c in args.player.upper())
    out=Path(args.output).expanduser().resolve() if args.output else ROOT/"output"/slug
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/"pipeline_manifest.json"
    options={"player":args.player,"rife":args.rife,"highlights":args.highlights,"cinematic":args.cinematic,
             "highlight_count":args.highlight_count,"reid_score":args.reid_min_score,"reid_margin":args.reid_min_margin}
    fp=fingerprint(video,model,options)
    state={"pipeline":"EXCEPTIONAL","version":"2026-09-27-exceptional-v10","fingerprint":fp,
           "started_at":time.time(),"stages":[]}
    atomic_json(manifest,state)
    def stage(name,status="running",**extra):
        state["stages"].append({"name":name,"status":status,"time":time.time(),**extra}); atomic_json(manifest,state)
    try:
        report_path=out/"tracking"/"tracking.json"; identity_path=None; target_track=None; matched=[]
        if args.track:
            stage("tracking")
            report_path.parent.mkdir(parents=True,exist_ok=True)
            cache=report_path.with_suffix(".cache.json")
            cached=False
            if not args.force and report_path.exists():
                try: cached=json.loads(report_path.read_text()).get("fingerprint")==fp
                except Exception: cached=False
            if not cached:
                run([sys.executable,str(ROOT/"yolo/track_football.py"),"--source",str(video),"--model",str(model),
                     "--output-dir",str(report_path.parent),"--json"])
                data=json.loads(report_path.read_text(encoding="utf8")); data["fingerprint"]=fp; atomic_json(report_path,data)
            else: data=json.loads(report_path.read_text()); stage("tracking","cached")
            summary=data.get("summary",{}); player=args.player.lower()
            matched=[x for x in summary.get("samples",[]) if str(x.get("class","")).lower()==player]
            if args.player_reference:
                stage("identity")
                identity_path=out/"player_identity.json"
                cmd=[sys.executable,str(ROOT/"yolo/player_reid.py"),"--video",str(video),"--tracking",str(report_path),
                     "--output",str(identity_path),"--min-score",str(args.reid_min_score),"--min-margin",str(args.reid_min_margin)]
                for ref in args.player_reference: cmd += ["--reference",str(Path(ref).expanduser().resolve())]
                run(cmd)
                ident=json.loads(identity_path.read_text())
                if ident.get("accepted"): target_track=ident.get("selected_track_id")
        else: data={"success":True,"summary":{}}
        stage("tracking","complete",detections=len(data.get("summary",{}).get("samples",[])))
        # Evidence-driven Football Director: event semantics are computed locally
        # before cinematic selection so the AI director receives measurable football
        # evidence instead of relying only on generic visual salience.
        events_path=out/"football_events.json"
        director_path=out/"football_director.json"
        if args.track and report_path.exists():
            stage("football_event_engine")
            run([sys.executable,str(ROOT/"yolo/event_engine.py"),"--tracking",str(report_path),"--output",str(events_path)])
            stage("football_director")
            run([sys.executable,str(ROOT/"yolo/football_director.py"),"--events",str(events_path),
                 "--output",str(director_path),"--duration",str(data.get("summary",{}).get("frames",0)/max(.1,float(data.get("summary",{}).get("fps",25))))])
        highlights_path=None; current=video; rife_applied="off"
        if args.highlights or args.cinematic:
            stage("event_highlights")
            highlights_path=out/"highlights.json"
            cmd=[sys.executable,str(ROOT/"yolo/scene_highlights.py"),"--video",str(video),"--tracking",str(report_path),
                 "--output",str(highlights_path),"--top-k",str(args.highlight_count)]
            if target_track is not None: cmd += ["--track-id",str(target_track)]
            run(cmd)
            selected=out/"highlights.mp4"; exp=1 if args.rife=="2x" else 2 if args.rife=="4x" else 0
            stage("highlight_render")
            render_cmd=[sys.executable,str(ROOT/"yolo/render_highlights.py"),"--video",str(video),
                        "--highlights",str(highlights_path),"--output",str(selected)]
            if args.scene_aware_rife and exp: render_cmd += ["--rife-exp",str(exp)]
            run(render_cmd); current=selected
            if exp and args.scene_aware_rife: rife_applied=f"scene-aware-{args.rife}"
        if args.cinematic:
            stage("cinematic_master")
            cinematic=out/"cinematic.mp4"
            run(["ffmpeg","-y","-i",str(current),"-vf",
                 "scale=1920:-2:force_original_aspect_ratio=decrease,format=yuv420p",
                 "-c:v","libx264","-preset","medium","-crf","18","-c:a","aac","-b:a","192k","-movflags","+faststart",str(cinematic)])
            current=cinematic
        if args.rife!="off" and not (args.scene_aware_rife and (args.highlights or args.cinematic)):
            stage("rife")
            interpolated=out/f"{slug}_{args.rife}.mp4"; rife(current,interpolated,1 if args.rife=="2x" else 2); current=interpolated; rife_applied=args.rife
        stage("quality_control")
        qc=out/"quality_report.json"
        q=subprocess.run([sys.executable,str(ROOT/"yolo/quality_control.py"),"--video",str(current),"--output",str(qc)],
                         cwd=str(ROOT))
        quality=json.loads(qc.read_text()) if qc.exists() else {"success":False}
        if not quality.get("success"): raise RuntimeError("Final media failed quality gate.")
        result={"success":True,"pipeline":"EXCEPTIONAL","version":"2026-09-27-exceptional-v10",
                "fingerprint":fp,"player":args.player,"input":str(video),"output":str(current),
                "tracking_json":str(report_path) if args.track else None,"matched_detections":len(matched),
                "identity_report":str(identity_path) if identity_path else None,"selected_track_id":target_track,
                "highlights":str(highlights_path) if highlights_path else None,"football_events":str(events_path) if events_path.exists() else None,"football_director":str(director_path) if director_path.exists() else None,"rife":args.rife,
                "rife_applied":rife_applied,"quality_report":str(qc),"summary":data.get("summary",{})}
        state["result"]=result; state["finished_at"]=time.time(); state["status"]="complete"; atomic_json(manifest,state)
        atomic_json(out/"result.json",result); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
    except Exception as e:
        state["status"]="failed"; state["error"]=str(e); state["finished_at"]=time.time(); atomic_json(manifest,state)
        print(f"PIPELINE FAILED: {e}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
