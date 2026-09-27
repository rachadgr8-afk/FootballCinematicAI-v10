#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--exp",type=int,choices=[1,2],default=1)
    a=ap.parse_args()
    repo=Path(os.getenv("RIFE_REPO","/opt/rife"))
    script=repo/"inference_video.py"
    if not script.exists(): raise SystemExit(f"RIFE not installed: {script}")
    src=Path(a.input).resolve(); out=Path(a.output).resolve()
    before={p.resolve() for p in repo.rglob("*.mp4")}
    subprocess.run([sys.executable,str(script),"--exp",str(a.exp),"--video",str(src)],cwd=str(repo),check=True)
    after=[p for p in repo.rglob("*.mp4") if p.resolve() not in before and p.stat().st_size>10000]
    if not after:
        # Some RIFE versions overwrite/create beside source.
        after=[p for p in src.parent.rglob("*.mp4") if p.resolve()!=src and p.stat().st_size>10000]
    if not after: raise SystemExit("RIFE completed but no output MP4 was found.")
    newest=max(after,key=lambda p:p.stat().st_mtime)
    out.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(newest,out)
    print(out)
if __name__=="__main__": main()
