# FootballCinematicAI — Professional Reference Reel Update

## Major changes

1. Gemini now receives the actual uploaded MP4 through the Gemini Files API and must inspect the video before producing timestamps.
2. 503/429 Gemini overload errors use retry/backoff; repeated failure is returned as an error instead of falling back to invented timestamps.
3. The old sample-video fallback was removed from the full render path.
4. The new `REFERENCE CINEMATIC REEL` preset reproduces the editorial language of the supplied reference: vertical framing, tight crops, close-ups/details, hard-cut pacing, selective slow motion, restrained flash impacts, white lower-third captions, dark cinematic grade and emotional climax/outro.
5. Gemini receives explicit instructions to build a 12–22 shot real-footage montage when enough source material exists.
6. Gemini output captions are preserved and rendered as small white uppercase editorial captions instead of boxed subtitles.
7. The FFmpeg engine now applies sharpening, vignette, subtle grain, color balance, contrast/saturation and transition-specific effects.
8. Final output is validated as 1080x1920, 30fps and exactly 64.00 seconds.
9. Optional `MUSIC_PATH` support mixes background music underneath original match audio when configured.
10. Gemini visual QC reviews the actual rendered MP4. If QC requests pacing corrections, the frontend performs one automatic real-footage polish pass using the same verified timeline.
11. Reference Style modal defaults now describe the supplied reference reel instead of an unrelated generic preset.

## Important

The reference video is used only as an editorial-style target. Its frames, players, logos and watermarks are not inserted into generated videos.

## 2026-09-26 — AI cinematic pipeline repair

- Connected the real server-side Veo 3.1 API to the master render pipeline.
- Added working `/api/generate-veo-shot`, `/api/video-status`, and `/api/video-download` endpoints.
- AI ENHANCED generates up to 1 Veo insert; AI CINEMATIC generates up to 3.
- Veo inserts use a real frame extracted from the uploaded football video as visual guidance and are passed to FFmpeg as local MP4 assets.
- Added an in-app AI Generation Mode selector and made AI CINEMATIC the default.
- Replaced the previous static zoom approximation with an animated `zoompan` punch-in while preserving Gemini focal points.
- Kept the main/climax moments grounded in the uploaded source footage; Veo is used for selected non-climax detail/close-up/reaction bridge shots.
- Added temporary cleanup for generated Veo assets after the master render.
- Added `VEO_MODEL`, `VEO_RESOLUTION`, and `VEO_MAX_POLLS` environment settings.

- 2026-09-26 — cinematic-v3: added `/api/version` so Android cannot silently use an older Render backend; densified Gemini-verified source ranges into 14–22 editorial shots; strengthened shot-type-dependent punch-ins; increased AI CINEMATIC Veo bridge capacity to 5 shots.

## v10.0.1 — Confirmed FFmpeg `[vo]` Graph Fix

The commentary mux graph previously consumed `[vo]` twice without explicitly splitting the stream. This can fail with an `Invalid stream specifier: vo` error depending on FFmpeg graph parsing.

Fixed graph:
`[1:a]volume=1.0[vo];[vo]asplit=2[vo_sc][vo_mix];[base][vo_sc]sidechaincompress=... [ducked];[ducked][vo_mix]amix=...`

Verification: FFmpeg successfully rendered a synthetic 2-second base/voice mix using the corrected graph.


## v10.0.2 — Release Hardening
- Unified `/api/version` build metadata with the v10 configuration.
- Renamed GitHub Actions APK artifacts to v10 labels.
