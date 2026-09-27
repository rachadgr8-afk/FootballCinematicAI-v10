#!/usr/bin/env python3
"""Deterministic cache fingerprint helper."""
import hashlib, json
from pathlib import Path
def fingerprint(video, model, options):
    h=hashlib.sha256()
    for p in [Path(video),Path(model)]:
        st=p.stat(); h.update(str(p.resolve()).encode()); h.update(str(st.st_size).encode()); h.update(str(st.st_mtime_ns).encode())
    h.update(json.dumps(options,sort_keys=True).encode())
    return h.hexdigest()[:20]
