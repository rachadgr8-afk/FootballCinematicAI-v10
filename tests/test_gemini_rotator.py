#!/usr/bin/env python3
"""Regression guard for the Gemini key rotator.

Bug: `isRotatableError()` only covered quota + overload errors, not
`isInvalidKeyError()`. A key pool containing any expired key (which the live
Render deployment had: 2 valid + 2 invalid out of 4) therefore made
`GeminiKeyRotator.run()` RETHROW on the first dead key instead of rotating to a
live one. Every render/analysis failed — the second cause of the outage.

This test drives the real rotator with a mixed pool and asserts it reaches a
working key. It needs no API keys and no network.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROTATOR = ROOT / "server" / "geminiRotator.ts"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not ROTATOR.exists():
        fail("server/geminiRotator.ts is missing")

    text = ROTATOR.read_text(encoding="utf-8")

    # 1. Static contract: an invalid key must be considered rotatable.
    m = re.search(r"export function isRotatableError\(err: any\): boolean \{\s*return ([^;]+);", text)
    if not m:
        fail("could not locate the isRotatableError() implementation")
    body = m.group(1)
    if "isInvalidKeyError" not in body:
        fail(
            "isRotatableError() no longer includes isInvalidKeyError(). A pool "
            "with any expired key will abort the whole render (the outage bug)."
        )
    print(f"isRotatableError() -> {body.strip()}")

    # 2. Behavioural contract: run the real rotator against a mixed key pool.
    #    This needs Node + node_modules (the rotator imports @google/genai), so it
    #    is skipped — never failed — in a Python-only environment such as the
    #    `python-quality` CI job, which installs no npm packages.
    node_modules = ROOT / "node_modules" / "@google" / "genai"
    if not node_modules.exists():
        print(
            "no node_modules/@google/genai present — behavioural harness skipped "
            "(static contract above is still enforced)."
        )
        print("\nGEMINI KEY ROTATOR CONTRACT: PASS (static)")
        return 0

    harness = ROOT / "_rotator_harness.ts"
    harness.write_text(
        """
import { GeminiKeyRotator, isRotatableError } from './server/geminiRotator';

const dead = () => new Error(JSON.stringify({ error: { code: 400,
  message: 'API key not valid. Please pass a valid API key.', status: 'INVALID_ARGUMENT',
  details: [{ reason: 'API_KEY_INVALID', domain: 'googleapis.com' }] } }));

process.env.GEMINI_API_KEYS = 'dead-key-1111111111,dead-key-2222222222,live-key-3333333333';
const rotator = new GeminiKeyRotator();

let calls = 0;
const seen: number[] = [];
rotator.run('unit-test', async (_client, state) => {
  calls += 1;
  seen.push(state.index);
  if (state.index < 2) throw dead();   // the two expired keys
  return 'ok';                          // the live key
}).then((result) => {
  const rotatable = isRotatableError(dead());
  console.log(JSON.stringify({ result, calls, seen, rotatable }));
}).catch((err) => {
  console.log(JSON.stringify({ result: 'THREW', error: String(err?.message || err).slice(0, 120) }));
  process.exit(3);
});
""",
        encoding="utf-8",
    )

    try:
        proc = subprocess.run(
            ["npx", "tsx", str(harness.name)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        harness.unlink(missing_ok=True)

    import json

    line = ""
    for ln in (proc.stdout or "").splitlines():
        if ln.strip().startswith("{"):
            line = ln.strip()
    if not line:
        fail(f"rotator harness produced no result.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    data = json.loads(line)
    print(f"rotator run -> {data}")

    if data.get("rotatable") is not True:
        fail("isRotatableError(invalidKey) is not true")
    if data.get("result") != "ok":
        fail(f"rotator did not fall through to the live key: {data}")
    if data.get("calls", 0) < 3:
        fail(f"rotator stopped early ({data.get('calls')} attempts) instead of rotating past dead keys")

    print("\nGEMINI KEY ROTATOR CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
