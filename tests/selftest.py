#!/usr/bin/env python3
"""Fast dependency-light structural self-test."""
import ast, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
required=["main.py","yolo/track_football.py","yolo/player_reid.py","yolo/event_engine.py",
          "yolo/scene_highlights.py","yolo/render_highlights.py","yolo/rife_interpolate.py",
          "yolo/quality_control.py","config/exceptional.json"]
for rel in required:
    p=ROOT/rel
    assert p.exists(), f"missing {rel}"
    ast.parse(p.read_text(encoding="utf8"))
cfg=json.loads((ROOT/"config/exceptional.json").read_text())
assert cfg["version"].startswith("2026-09-27-exceptional")
req=(ROOT/"requirements.txt").read_text()
assert "torch==2.6.0+cpu" in req and "torchvision==0.21.0+cpu" in req
assert "download.pytorch.org/whl/cpu" in req
print("EXCEPTIONAL STRUCTURAL SELF-TEST: PASS")
