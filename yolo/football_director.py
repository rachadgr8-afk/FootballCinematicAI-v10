#!/usr/bin/env python3
"""Football Cinematic Director v10.

Turns event evidence into explainable tension peaks and a story map. It does
not invent football actions; uncertain actions remain explicit candidates.
"""
import argparse, json
from pathlib import Path

def clamp(v,a=0.0,b=1.0): return max(a,min(b,float(v)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--events",required=True); ap.add_argument("--output",required=True); ap.add_argument("--duration",type=float,default=None); a=ap.parse_args()
    data=json.loads(Path(a.events).read_text(encoding="utf8")); fps=max(.1,float(data.get("fps",25))); events=data.get("events",[])
    duration=a.duration
    if duration is None: duration=max([float(e.get("time",0)) for e in events] or [0])+1
    if not events:
        out={"success":True,"version":"v10","duration":duration,"evidence_level":"low","peaks":[],"story_windows":[],"music_curve":[]}
        Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8"); return
    bins={}
    for e in events:
        k=int(float(e.get("time",0))*2); b=bins.setdefault(k,{"scores":[],"prox":[],"pressure":[],"turn":[],"ball_speed":[],"burst":[],"continuity":[],"tracks":set(),"labels":[]})
        for key,target in (("event_score","scores"),("ball_proximity","prox"),("pressure","pressure"),("direction_change","turn"),("ball_speed","ball_speed"),("acceleration","burst"),("engagement_continuity","continuity")):
            b[target].append(float(e.get(key,0)))
        if e.get("track_id") is not None: b["tracks"].add(str(e["track_id"]))
        b["labels"] += list(e.get("events",[]))
    rows=[]
    for k,b in sorted(bins.items()):
        peak=max(b["scores"]); avg=sum(b["scores"])/len(b["scores"])
        prox=max(b["prox"]); pressure=max(b["pressure"]); turn=max(b["turn"]); bs=max(b["ball_speed"]); burst=max(b["burst"]); cont=max(b["continuity"])
        tension=clamp(.43*peak+.12*avg+.13*prox+.09*pressure+.08*turn+.07*bs+.05*burst+.03*cont)
        label_counts={x:b["labels"].count(x) for x in set(b["labels"])}
        dominant=max(label_counts,key=label_counts.get) if label_counts else "motion"
        rows.append({"start":round(k/2,3),"end":round(min(duration,k/2+.5),3),"tension":round(tension,4),"peak_event":round(peak,4),"ball_engagement":round(prox,4),"pressure":round(pressure,4),"direction_change":round(turn,4),"ball_speed":round(bs,4),"burst":round(burst,4),"continuity":round(cont,4),"track_count":len(b["tracks"]),"dominant_signal":dominant})
    ranked=sorted(rows,key=lambda x:x["tension"],reverse=True); peaks=[]
    for r in ranked:
        c=(r["start"]+r["end"])/2
        if all(abs(c-(p["start"]+p["end"])/2)>1.5 for p in peaks): peaks.append(r)
        if len(peaks)>=min(14,max(6,int(duration/4))): break
    peaks.sort(key=lambda x:x["start"])
    story=[]
    for i,r in enumerate(peaks):
        rel=((r["start"]+r["end"])/2)/max(duration,1)
        if i==0 or rel<.10: role="hook"
        elif rel<.28: role="setup"
        elif rel<.55: role="escalation"
        elif rel<.72: role="impact"
        elif rel<.86: role="reaction"
        elif rel<.96: role="climax"
        else: role="outro"
        story.append({**r,"role":role,"silence_before_ms":220 if r["tension"]>.78 else 100 if r["tension"]>.62 else 0,"replay_candidate":bool(r["tension"]>.82 and (r["ball_engagement"]>.55 or r["ball_speed"]>.65))})
    n=16; curve=[]
    for i in range(n):
        lo=i*duration/n; hi=(i+1)*duration/n; vals=[r["tension"] for r in rows if r["start"]<hi and r["end"]>lo]; curve.append(sum(vals)/len(vals) if vals else .12)
    mx=max(curve or [.12]); curve=[round(.22+.78*v/mx,3) for v in curve]
    out={"success":True,"version":"v10","duration":round(duration,3),"evidence_level":"high" if data.get("ball_detected") and len(events)>20 else "medium" if events else "low","ball_detected":bool(data.get("ball_detected")),"event_count":len(events),"peaks":peaks,"story_windows":story,"music_curve":curve,"director_rules":{"silence_before_high_tension_ms":220,"peak_separation_seconds":1.5,"replay_requires_evidence":True,"no_claim_without_visual_evidence":True,"candidate_labels_are_not_assertions":True}}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"success":True,"version":"v10","peaks":len(peaks),"evidence_level":out["evidence_level"]}))
if __name__=="__main__": main()
