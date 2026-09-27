#!/usr/bin/env python3
"""Regression guard: every API route the frontend calls MUST be defined in server.ts.

Background
----------
On 2026-09-27 a commit titled "Update server.ts" (4d16525) replaced the whole
server.ts with a truncated 19 KB stub, silently deleting ~1115 lines and 16
routes. The deployed Render backend then answered HTTP 404 for /api/upload-video,
/api/analyze-video and /api/render-full-cinematic, which made the app freeze
around 20% of the render and finally stop working entirely.

This test parses the frontend sources for `safeFetchJson('<route>'` / `fetch`
calls and asserts that each one has a matching `app.<verb>('<route>'` in
server.ts, so a truncated server.ts can never ship again unnoticed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server.ts"
SRC = ROOT / "src"

# Routes resolved by helper functions (uploadVideo / veoManager) instead of a
# literal safeFetchJson() call, so they must be listed explicitly.
EXTRA_REQUIRED = {
    "/api/upload-video",
    "/api/video-download",
}

# Routes deliberately served by the client-side/local engine fallback only.
OPTIONAL = {
    "/api/upload",       # legacy alias still present server-side
    "/api/test-render-1",  # sanity probes, optional in production
    "/api/test-render-2",
    "/api/test-render-3",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not SERVER.exists():
        fail("server.ts is missing")
    if not SRC.exists():
        fail("src/ directory is missing")

    server_text = SERVER.read_text(encoding="utf-8")

    # Guard against the exact class of regression that broke production:
    # a truncated/stubbed server that silently drops hundreds of lines.
    line_count = len(server_text.splitlines())
    if line_count < 800:
        fail(
            f"server.ts has only {line_count} lines. The production server is "
            "~1560 lines; a value this low means the file was truncated or "
            "replaced by a stub (this is what caused the 2026-09-27 outage)."
        )

    defined_routes = set(re.findall(r"app\.(?:get|post|put|delete)\(\s*'([^']+)'", server_text))

    frontend_routes: set[str] = set()
    for path in SRC.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        # safeFetchJson('/api/...'  |  safeFetchJson<Type>('/api/...
        frontend_routes.update(re.findall(r"safeFetchJson(?:<[^>]*>)?\(\s*'([^']+)'", text))
        # getApiUrl('/api/...')
        frontend_routes.update(re.findall(r"getApiUrl\(\s*'([^']+)'", text))

    required = {r for r in (frontend_routes | EXTRA_REQUIRED) if r.startswith("/api/")}
    required -= OPTIONAL

    missing = sorted(
        r for r in required
        if not any(r == d or (d.endswith("*") and r.startswith(d[:-1])) for d in defined_routes)
    )

    print(f"server.ts lines: {line_count}")
    print(f"routes defined in server.ts: {len(defined_routes)}")
    print(f"frontend-required routes:    {len(required)}")

    if missing:
        print("\nMissing routes the frontend calls but server.ts does not define:")
        for r in missing:
            print(f"  - {r}")
        fail(
            f"{len(missing)} frontend route(s) are not implemented in server.ts. "
            "The app will get HTTP 404 at runtime — restore the full server.ts."
        )

    # The app refuses to render unless the backend advertises the v10 build.
    if "exceptional-v10" not in server_text:
        fail("server.ts no longer reports an 'exceptional-v10' BUILD_VERSION.")

    print("\nSERVER ROUTE CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
