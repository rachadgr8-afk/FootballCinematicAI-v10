#!/usr/bin/env python3
"""Appearance-based player track matching with confidence gating.

This is a lightweight Re-ID stage, not face recognition. It supports multiple
reference images and returns an explicit UNKNOWN decision when the evidence is
weak or ambiguous.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

def emb(img):
    img=cv2.resize(img,(128,256))
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256]).astype(np.float32)
    hist=cv2.normalize(hist,hist).flatten()
    small=cv2.resize(hsv,(16,16)).astype(np.float32).reshape(-1)
    small=(small-small.mean())/(small.std()+1e-6)
    return np.concatenate([hist,small])

def cosine(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d else 0.0

def crop_from_sample(img, s):
    h,w=img.shape[:2]
    if all(k in s for k in ("x1","y1","x2","y2")):
        x1=max(0,min(w-1,int(float(s["x1"])*w)))
        y1=max(0,min(h-1,int(float(s["y1"])*h)))
        x2=max(x1+2,min(w,int(float(s["x2"])*w)))
        y2=max(y1+2,min(h,int(float(s["y2"])*h)))
    else:
        cx=int(float(s.get("x",.5))*w); cy=int(float(s.get("y",.5))*h)
        bw=max(20,int(float(s.get("w",.12))*w)); bh=max(40,int(float(s.get("h",.30))*h))
        x1=max(0,cx-bw//2); y1=max(0,cy-bh//2); x2=min(w,cx+bw//2); y2=min(h,cy+bh//2)
    return img[y1:y2,x1:x2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True)
    ap.add_argument('--tracking',required=True)
    ap.add_argument('--reference',action='append',required=True,
                    help='Reference image; repeat for multiple views')
    ap.add_argument('--output',required=True)
    ap.add_argument('--top-k',type=int,default=5)
    ap.add_argument('--min-score',type=float,default=0.72)
    ap.add_argument('--min-margin',type=float,default=0.035)
    ap.add_argument('--sample-stride',type=int,default=12)
    args=ap.parse_args()

    refs=[]
    for ref_path in args.reference:
        ref=cv2.imread(ref_path)
        if ref is None: raise SystemExit(f'Reference image not found/readable: {ref_path}')
        refs.append(emb(ref))

    data=json.loads(Path(args.tracking).read_text(encoding='utf8'))
    samples=data.get('summary',{}).get('samples',[])
    cap=cv2.VideoCapture(args.video)
    scores={}; counts={}; best_each={}
    seen={}
    for s in samples:
        tid=s.get('track_id')
        if tid is None: continue
        seen[tid]=seen.get(tid,0)+1
        if seen[tid] % max(1,args.sample_stride) != 1: continue
        frame=max(0,int(s.get('frame',0))-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,img=cap.read()
        if not ok: continue
        crop=crop_from_sample(img,s)
        if crop.size==0: continue
        e=emb(crop)
        # Multi-view reference ensemble: best reference match is less brittle
        # than averaging unrelated camera angles.
        sim=max(cosine(e,r) for r in refs)
        scores[tid]=scores.get(tid,0.0)+sim
        counts[tid]=counts.get(tid,0)+1
        best_each[tid]=max(best_each.get(tid,0.0),sim)
    cap.release()

    ranked=sorted(
        ((tid,scores[tid]/max(1,counts[tid]),counts[tid],best_each.get(tid,0.0))
         for tid in scores),
        key=lambda x:(x[1],x[2],x[3]), reverse=True
    )
    rows=[{'track_id':tid,'score':round(avg,4),'samples':n,'best_frame_score':round(best,4)}
          for tid,avg,n,best in ranked[:args.top_k]]
    top=rows[0] if rows else None
    second=rows[1] if len(rows)>1 else None
    margin=(top['score']-second['score']) if top and second else 1.0
    accepted=bool(top and top['score']>=args.min_score and margin>=args.min_margin)
    result={
        'success':True,'references':args.reference,'accepted':accepted,
        'selected_track_id':top['track_id'] if accepted else None,
        'selection_reason':(
            'accepted' if accepted else
            'no_candidate' if not top else
            'score_below_threshold' if top['score']<args.min_score else
            'ambiguous_margin'
        ),
        'thresholds':{'min_score':args.min_score,'min_margin':args.min_margin},
        'ranked_tracks':rows
    }
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    main()
