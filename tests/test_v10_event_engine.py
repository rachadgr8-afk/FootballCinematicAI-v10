import json, subprocess, sys
from pathlib import Path

def test_event_engine_synthetic(tmp_path):
    samples=[]
    # Player accelerates toward a ball while another player closes in.
    for f in range(30):
        x=.30 + .012*f
        samples.append({'frame':f,'track_id':1,'class':'player','x':x,'y':.5,'w':.1,'h':.2,'confidence':.9})
        samples.append({'frame':f,'track_id':2,'class':'player','x':min(.65,.62+.001*f),'y':.5,'w':.1,'h':.2,'confidence':.88})
        samples.append({'frame':f,'track_id':99,'class':'ball','x':x+.02,'y':.5,'w':.03,'h':.03,'confidence':.92})
    tracking={'summary':{'fps':25,'samples':samples}}
    src=tmp_path/'tracking.json'; out=tmp_path/'events.json'; src.write_text(json.dumps(tracking))
    root=Path(__file__).parents[1]
    subprocess.run([sys.executable,str(root/'yolo/event_engine.py'),'--tracking',str(src),'--output',str(out)],check=True)
    data=json.loads(out.read_text())
    assert data['version']=='v10'
    assert data['ball_detected'] is True
    assert any('ball_engagement' in e['events'] for e in data['events'])
