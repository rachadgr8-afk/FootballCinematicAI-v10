# Reference Style v11 — Cinematic Director driven by a MEASURED reference

## What changed (additive; nothing removed)

| Area | File | Change |
|---|---|---|
| Reference measurement | `yolo/reference_style_analyzer.py` (new) | Decodes a reference video frame-by-frame and measures shot durations, cut density, camera-movement classes, zoom intensity, speed handling (duplicate-frame ratio), framing mix, subject/isolation proxies, caption frequency + height, transition mix, colour characteristics (incl. skin and neon-grass ratios), audio impacts/risers/silence + cut-impact sync, hero-shot and ending structure. |
| Director | `yolo/cinematic_director.py` (new) | Cinematic Director v11: builds Edit Plan V2 from real motion + real events + real player/ball tracking, constrained by the measured style. Adds `cinematic_director`, `sound_design`, `color_grade` semantics, `subject_anchor`, `slow_motion`, `subject_isolation`. |
| Style QC | `yolo/style_qc.py` (new) | Measures the produced master with the SAME analyser and reports the distance to the reference profile + the hard output contract. |
| Bridge | `server/referenceStyleService.ts` (new) | Fault-tolerant TS bridge (never throws), with a profile cache. |
| Server | `server.ts` | `cinematicMode` + `referenceStyleProfile` + `referenceLocalPath` on `/api/analyze-video`; new `/api/reference-style/analyze`; `/api/analyze-reference` now upgrades to real measurement; new `/api/cinematic-director/plan`; `toLegacyStyleProfile()`. Fallback to `buildLocalEditPlan()` kept intact. |
| Renderer | `server/ffmpegEngine.ts` | Eased (smoothstep) dynamic reframing with an optional exit anchor; dissolve/fade implementation; background-only subject isolation (never full-frame blur, mask-blended, SAM -> YOLO -> motion); cinematic master grade (cooler shadows, warmer highlights, protected skin tones, de-neoned grass). Pipeline/contract preserved: 64s / 1080x1920 / 30fps / H.264 / yuv420p / AAC. |
| Frontend | `src/types/football.ts`, `src/App.tsx`, `src/components/ReferenceStyleModal.tsx` | `CinematicMode` (STANDARD / PRO / REFERENCE), reference video upload + measured profile display, mode selector. |
| Config | `config/cinematic.json` (new) | Modes, quality rules, output contract. |

## Reference matching is style-only
The profile carries style parameters. It never carries the reference's timestamps, shot order, frames, logos or watermarks, and the renderer never imports reference pixels.

## Honest measurement limits
`subject_shot_ratio` is a skin-coverage proxy computed on the SOURCE frames. It reports how much human/subject signal the footage contains; it does NOT change when the director crops in tighter. Judge player-first framing by the Edit Plan's `shot_type` mix and `subject_anchor`, and by visual inspection of the master.
