#!/usr/bin/env python3
"""Event-aware highlight selection from YOLO/ByteTrack timelines."""
import argparse, json
from pathlib import Path
import cv2, numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True); ap.add_argument('--tracking',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--scene-threshold',type=float,default=.48); ap.add_argument('--top-k',type=int,default=8)
    ap.add_argument('--clip',type=float,default=5.5); ap.add_argument('--track-id',default=None)
    a=ap.parse_args()
    data=json.loads(Path(a.tracking).read_text(encoding='utf8'))
    all_samples=data.get('summary',{}).get('samples',[])
    fps=max(.1,float(data.get('summary',{}).get('fps',25)))
    samples=[s for s in all_samples if a.track_id is None or str(s.get('track_id'))==str(a.track_id)]
    samples=[s for s in samples if s.get('track_id') is not None]
    by_frame={}
    for s in samples:
        by_frame.setdefault(int(s.get('frame',0)),[]).append(s)

    # Scene-cut detection
    cap=cv2.VideoCapture(a.video); prev=None; cuts=[]; frame=0
    while True:
        ok,img=cap.read()
        if not ok: break
        frame+=1
        small=cv2.resize(img,(160,90)); gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff=float(np.mean(cv2.absdiff(gray,prev))/255.0)
            if diff>=a.scene_threshold: cuts.append(frame)
        prev=gray
    cap.release()

    # Per-track kinematics: speed, acceleration and bbox-size change are useful
    # proxies for action intensity when a ball/event model is unavailable.
    history={}
    for s in samples:
        tid=str(s['track_id']); f=int(s.get('frame',0))
        history.setdefault(tid,[]).append((f,float(s.get('x',.5)),float(s.get('y',.5)),
                                           float(s.get('w',.1)),float(s.get('h',.2)),
                                           float(s.get('confidence',0))))
    stats={}
    for tid,arr in history.items():
        arr.sort()
        for i,(f,x,y,w,h,cf) in enumerate(arr):
            if i==0: speed=accel=0.0
            else:
                pf,px,py,pw,ph,_=arr[i-1]
                dt=max(1,(f-pf))/fps
                speed=((x-px)**2+(y-py)**2)**0.5/dt
                accel=0.0
                if i>1:
                    ppf,ppx,ppy,_,_,_=arr[i-2]
                    pdt=max(1,(pf-ppf))/fps
                    prev_speed=((px-ppx)**2+(py-ppy)**2)**0.5/pdt
                    accel=abs(speed-prev_speed)/max(dt,.05)
            stats[(tid,f)]=(speed,accel)

    frames=sorted(by_frame)
    half=max(1,int(fps*.6)); nearby={}
    left=right=0
    for f in frames:
        while left<len(frames) and frames[left]<f-half: left+=1
        right=max(right,left)
        while right<len(frames) and frames[right]<=f+half: right+=1
        nearby[f]=right-left

    # Optional event engine enriches generic motion scoring with ball/player context.
    event_scores={}
    try:
        import subprocess, sys, tempfile
        with tempfile.TemporaryDirectory(prefix="fca_events_") as td:
            ep=Path(td)/"events.json"
            cmd=[sys.executable,str(Path(__file__).with_name("event_engine.py")),
                 "--tracking",str(a.tracking),"--output",str(ep)]
            if a.track_id is not None: cmd += ["--track-id",str(a.track_id)]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            ed=json.loads(ep.read_text(encoding="utf8"))
            event_scores={(str(e.get("track_id")),int(e.get("frame",0))):e for e in ed.get("events",[])}
            ball_detected=bool(ed.get("ball_detected"))
    except Exception:
        ball_detected=False

    candidates=[]
    for s in samples:
        f=int(s.get('frame',0)); tid=str(s['track_id'])
        conf=float(s.get('confidence',0)); x=float(s.get('x',.5)); y=float(s.get('y',.55))
        w=float(s.get('w',.1)); h=float(s.get('h',.2))
        speed,accel=stats.get((tid,f),(0,0))
        density=min(nearby.get(f,1)/16,1)
        ev=event_scores.get((tid,f),{})
        event_score=float(ev.get("event_score",0))
        proximity=float(ev.get("ball_proximity",0))
        # Action proxy: confidence + player prominence + local density +
        # motion/acceleration. Penalize very tiny/edge detections and cuts.
        prominence=min(1.0,(w*h)/.08)
        motion=min(1.0,speed/.8)
        burst=min(1.0,accel/1.2)
        central=max(0.0,1-abs(x-.5)*1.7)
        score=(.18*conf+.14*prominence+.12*density+.16*motion+.10*burst+.08*central+
                .17*event_score+.05*proximity)
        if y<.06 or y>.97: score*=.7
        if any(abs(f-c)<=int(fps*.7) for c in cuts): score*=.30
        candidates.append((score,f,s,{
            "event_score": event_score,
            "ball_proximity": proximity,
            "speed": speed,
            "acceleration": accel,
            "burst": burst,
            "reason": ('ball_engagement' if proximity > .72 else 'burst' if burst > .72 else 'sprint' if motion > .62 else 'motion')
        }))

    candidates.sort(key=lambda z:z[0],reverse=True)
    chosen=[]; min_gap=max(1,int(fps*a.clip*.78))
    for score,f,s,evmeta in candidates:
        if all(abs(f-c['frame'])>min_gap for c in chosen):
            chosen.append({
                'frame':f,'time':round(f/fps,3),'track_id':s.get('track_id'),
                'score':round(float(score),4),'start':max(0,round(f/fps-a.clip/2,3)),
                'duration':a.clip,'confidence':round(float(s.get('confidence',0)),3),
                'motion':round(float(evmeta['speed']),3),
                'acceleration':round(float(evmeta['acceleration']),3),
                'event_score':round(float(evmeta['event_score']),4),
                'ball_proximity':round(float(evmeta['ball_proximity']),4),
                'reason':evmeta['reason']
            })
        if len(chosen)>=max(1,a.top_k): break

    out={'success':True,'fps':fps,'scene_cuts':[round(c/fps,3) for c in cuts],
         'target_track_id':a.track_id,'scoring':'confidence+prominence+density+motion+acceleration+event_score+ball_proximity',
         'ball_detection_used':ball_detected,
         'highlights':chosen}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
