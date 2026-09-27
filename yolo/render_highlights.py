#!/usr/bin/env python3
"""Render selected highlights, optionally interpolating each shot before concat."""
import argparse, json, subprocess, tempfile, sys, os
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True); ap.add_argument('--highlights',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--width',type=int,default=1920); ap.add_argument('--rife-exp',type=int,choices=[1,2],default=0)
    a=ap.parse_args()
    h=json.loads(Path(a.highlights).read_text(encoding='utf8')); clips=h.get('highlights',[])
    if not clips: raise SystemExit('No highlights were selected.')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fca_clips_') as td:
        files=[]
        for i,c in enumerate(clips):
            raw=Path(td)/f'{i:03d}_raw.mp4'; final=Path(td)/f'{i:03d}.mp4'
            start=max(0,float(c['start'])); dur=max(.2,float(c['duration']))
            cmd=['ffmpeg','-y','-ss',str(start),'-i',a.video,'-t',str(dur),
                 '-vf',f'scale={a.width}:-2:force_original_aspect_ratio=decrease,format=yuv420p',
                 '-c:v','libx264','-preset','medium','-crf','18','-c:a','aac','-b:a','192k',str(raw)]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            source=raw
            if a.rife_exp:
                # Interpolate each shot independently so RIFE never invents
                # frames across a hard cut between two unrelated highlights.
                subprocess.run([sys.executable,str(Path(__file__).with_name('rife_interpolate.py')),
                                '--input',str(raw),'--output',str(final),'--exp',str(a.rife_exp)],
                               check=True,stdout=subprocess.DEVNULL)
                source=final
            files.append(source)
        manifest=Path(td)/'concat.txt'
        manifest.write_text('\n'.join("file '"+str(f)+"'" for f in files),encoding='utf8')
        subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',str(manifest),
                        '-c','copy','-movflags','+faststart',str(out)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print(out)
if __name__=='__main__': main()
