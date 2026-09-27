import json, tempfile, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_event_engine_smoke():
    with tempfile.TemporaryDirectory() as td:
        t=Path(td); tracking={"summary":{"fps":25,"samples":[
            {"frame":1,"track_id":1,"class":"player","x":.5,"y":.5,"confidence":.9},
            {"frame":2,"track_id":1,"class":"player","x":.55,"y":.5,"confidence":.9},
            {"frame":2,"track_id":99,"class":"ball","x":.56,"y":.5,"confidence":.8}]}}
        inp=t/"tracking.json"; out=t/"events.json"; inp.write_text(json.dumps(tracking))
        p=subprocess.run([sys.executable,str(ROOT/"yolo/event_engine.py"),"--tracking",str(inp),"--output",str(out)])
        assert p.returncode==0 and out.exists()
def test_python_compile():
    files=[ROOT/"main.py"]+list((ROOT/"yolo").glob("*.py"))
    for f in files:
        p=subprocess.run([sys.executable,"-m","py_compile",str(f)])
        assert p.returncode==0, f

def test_v10_intelligence_files_exist():
    from pathlib import Path
    root=Path(__file__).parents[1]
    assert (root/'yolo'/'football_director.py').exists()
    assert (root/'yolo'/'event_engine.py').exists()
