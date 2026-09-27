#!/usr/bin/env python3
"""Football event intelligence engine.

Builds explainable event candidates from YOLO/ByteTrack samples. It deliberately
uses neutral labels such as ``shot_candidate`` or ``pressure_candidate`` when
visual evidence is insufficient to assert a real football action.
"""
import argparse, json, math
from pathlib import Path

BALL_NAMES={"ball","football","soccer_ball","soccer-ball","sports ball","sports_ball"}
PLAYER_HINTS={"player","person","footballer","athlete"}

def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def center(s): return float(s.get("x",.5)), float(s.get("y",.5))
def clamp(v,a=0.0,b=1.0): return max(a,min(b,float(v)))

def nearest_ball(xy, frame, by_frame):
    best=1.0; best_frame=None
    for bf in range(frame-3, frame+4):
        for b in by_frame.get(bf,[]):
            d=dist(xy,center(b))
            if d<best: best=d; best_frame=bf
    return best,best_frame

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tracking",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--track-id",default=None)
    a=ap.parse_args()
    data=json.loads(Path(a.tracking).read_text(encoding="utf8"))
    sm=data.get("summary",{}).get("samples",[])
    fps=max(.1,float(data.get("summary",{}).get("fps",25)))
    balls=[s for s in sm if str(s.get("class","")).lower() in BALL_NAMES]
    players=[s for s in sm if str(s.get("class","")).lower() in PLAYER_HINTS or str(s.get("class","")).lower() not in BALL_NAMES]
    if a.track_id is not None:
        players=[s for s in players if str(s.get("track_id"))==str(a.track_id)]
    by_frame={}
    for s in balls: by_frame.setdefault(int(s.get("frame",0)),[]).append(s)

    # Kinematics per tracked player.
    histories={}
    for s in sorted(players,key=lambda z:int(z.get("frame",0))):
        tid=str(s.get("track_id")); histories.setdefault(tid,[]).append(s)
    kin={}
    prev_state={}
    for tid,arr in histories.items():
        arr.sort(key=lambda z:int(z.get("frame",0)))
        last_dir=None
        for s in arr:
            f=int(s.get("frame",0)); x,y=center(s)
            speed=accel=direction_change=0.0; dx=dy=0.0
            if tid in prev_state:
                pf,px,py,ps,pdx,pdy=prev_state[tid]
                dt=max(1,f-pf)/fps; dx=x-px; dy=y-py
                speed=math.hypot(dx,dy)/dt
                accel=abs(speed-ps)/max(dt,.05)
                if math.hypot(pdx,pdy)>.008 and math.hypot(dx,dy)>.008:
                    dot=pdx*dx+pdy*dy
                    denom=math.hypot(pdx,pdy)*math.hypot(dx,dy)
                    angle=math.degrees(math.acos(clamp(dot/denom,-1,1)))
                    direction_change=clamp(angle/90)
            prev_state[tid]=(f,x,y,speed,dx,dy)
            kin[(tid,f)]={"speed":speed,"acceleration":accel,"direction_change":direction_change,"dx":dx,"dy":dy}

    # Ball kinematics, when a ball class exists.
    ball_history=[]
    for f in sorted(by_frame):
        # choose highest-confidence ball per frame
        b=max(by_frame[f],key=lambda z:float(z.get("confidence",0)))
        ball_history.append((f,*center(b)))
    ball_kin={}; prev=None
    for f,x,y in ball_history:
        bs=0.0
        if prev:
            pf,px,py=prev; bs=dist((x,y),(px,py))/(max(1,f-pf)/fps)
        ball_kin[f]=bs; prev=(f,x,y)

    # Frame-local pressure: proximity to the nearest *other tracked player*.
    frame_players={}
    for s in players:
        frame_players.setdefault(int(s.get("frame",0)),[]).append(s)

    events=[]
    for s in sorted(players,key=lambda z:int(z.get("frame",0))):
        f=int(s.get("frame",0)); tid=str(s.get("track_id")); x,y=center(s)
        k=kin.get((tid,f),{})
        speed=float(k.get("speed",0)); accel=float(k.get("acceleration",0)); turn=float(k.get("direction_change",0))
        nearest,bf=nearest_ball((x,y),f,by_frame)
        proximity=clamp(1-min(nearest/.22,1)) if balls else 0.0
        pressure=0.0
        for other in frame_players.get(f,[]):
            if str(other.get("track_id"))==tid: continue
            pressure=max(pressure,clamp(1-dist((x,y),center(other))/.24))
        motion=clamp(speed/.75); burst=clamp(accel/1.2); ball_speed=clamp(ball_kin.get(bf,0)/1.4) if bf is not None else 0.0
        conf=clamp(float(s.get("confidence",0)))
        continuity=0.0
        # A short ball-engagement window is more meaningful than one isolated frame.
        for q in range(max(0,f-5),f+6):
            if any(dist((x,y),center(b))<.22 for b in by_frame.get(q,[])):
                continuity += 1
        continuity=clamp(continuity/11)
        event=.24*proximity+.18*motion+.14*burst+.12*conf+.10*pressure+.10*turn+.07*ball_speed+.05*continuity

        labels=[]
        if proximity>.72: labels.append("ball_engagement")
        if pressure>.72 and proximity>.35: labels.append("pressure_candidate")
        if turn>.70 and motion>.35: labels.append("direction_change_candidate")
        if burst>.72: labels.append("explosive_run")
        if proximity>.55 and ball_speed>.72: labels.append("high_ball_speed_candidate")
        if not labels: labels.append("motion")

        events.append({
            "frame":f,"time":round(f/fps,3),"track_id":s.get("track_id"),
            "event_score":round(event,4),"events":labels,"event":labels[0],
            "ball_proximity":round(proximity,4),"ball_frame":bf,
            "ball_speed":round(ball_speed,4),"pressure":round(pressure,4),
            "speed":round(speed,4),"acceleration":round(accel,4),
            "direction_change":round(turn,4),"engagement_continuity":round(continuity,4),
            "confidence":round(conf,4)
        })

    out={"success":True,"version":"v10","fps":fps,"ball_detected":bool(balls),
         "player_samples":len(players),"events":events,
         "weights":{"ball_proximity":.24,"motion":.18,"acceleration":.14,"confidence":.12,
                    "pressure":.10,"direction_change":.10,"ball_speed":.07,"continuity":.05},
         "semantic_policy":{"assertive_actions":[],"candidate_actions":["pressure_candidate","direction_change_candidate","high_ball_speed_candidate"],
                             "no_claim_without_visual_evidence":True}}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"success":True,"version":"v10","ball_detected":bool(balls),"events":len(events)}))
if __name__=="__main__": main()
