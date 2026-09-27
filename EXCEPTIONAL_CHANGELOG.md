# EXCEPTIONAL v7

## Core
- Production pipeline manifest with stage timing and failure state.
- Deterministic fingerprint for video/model/options.
- Safe tracking cache with `--force` override.
- Final quality gate using ffprobe + OpenCV decode samples.

## Football intelligence
- Event engine for ball/player proximity when ball detections are available.
- Motion + acceleration + confidence fusion.
- Explainable highlight reasons and event scores.
- Identity remains UNKNOWN when evidence is weak/ambiguous.

## Rendering
- Scene-aware RIFE remains per-shot.
- No interpolation across unrelated highlight cuts.
- Final output cannot be marked successful if quality gate fails.

## CI
- CPU-only Torch/Torchvision pair remains pinned.
- Structural self-test and Python compilation run in GitHub Actions.
- Android lint/build remains part of CI.


### v8 — Football Director
- Replaced single-block TTS commentary with timeline-aware Arabic performance generation.
- Added emotion/intensity/speed-aware ElevenLabs voice settings.
- Added per-line timing and natural silence.
- Added sidechain ducking against stadium/music audio.
- Commentary now uses the post-QC active edit plan rather than a stale pre-polish plan.
- Updated build version to `2026-09-27-exceptional-v8`.

## v9 — Football Cinematic Intelligence
- Added evidence-driven Football Director.
- Added measurable tension peaks, ball engagement, acceleration and story-role mapping.
- Gemini edit planning now receives compact football evidence instead of a huge raw tracker dump.
- Commentary Director receives the same evidence context and can choose silence when evidence is weak.
- Fixed stale event metadata in highlight selection.

## v10 — Football Cinematic Intelligence

- upgraded event engine with ball engagement continuity, pressure, direction change, ball speed and kinematics
- added evidence-aware tension and replay candidates
- made uncertain football actions explicit `*_candidate` signals instead of fabricated assertions
- upgraded director map and story roles
- versioned pipeline metadata to `2026-09-27-exceptional-v10`

## v10.0.1 — FFmpeg Commentary Graph Fix
- Fixed duplicated `[vo]` consumption in the commentary sidechain filter graph.
- Added `asplit=2` to create independent voice streams for `sidechaincompress` and `amix`.
- Verified the corrected filter graph with FFmpeg on synthetic audio; mux completed successfully.


## v10.0.2 — Release Hardening
- Unified `/api/version` build metadata with the v10 configuration.
- Renamed GitHub Actions APK artifacts to v10 labels.


## MADNESS ENGINE v5
- Added `yolo/madness_engine.py` as an evidence-gated cinematic escalation layer.
- Runs after the validated Gemini timeline and football event evidence.
- Supports five intensity levels, with Level 5 limited to one high-confidence climax.
- Added FFmpeg psychological treatments for Levels 3–5 while preserving real-footage-only policy.
