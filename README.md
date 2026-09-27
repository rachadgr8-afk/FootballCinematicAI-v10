<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/dc01fd3e-8555-4028-a8b2-8a89ed564ca0

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`


## IMPORTANT — cinematic-v3 backend deployment

The Android app checks `/api/version` before rendering. The expected backend build is
`2026-09-26-cinematic-v3`. After deploying this source to Render, verify:
`GET https://fotbal-1.onrender.com/api/version`
and confirm the returned `buildVersion` is `2026-09-26-cinematic-v3`.

If the backend reports an older build, the app stops instead of silently producing the
old source-only render. The APK must point to the newly deployed backend.

## Football YOLO + ElevenLabs upgrade

This version adds:
- Custom `models/best.pt` YOLOv8 football detector/tracker.
- ByteTrack-based player/ball tracking with normalized coordinates and an annotated MP4.
- Gemini receives the tracking summary as supporting evidence while still being required to inspect the source video.
- ElevenLabs Arabic commentary generation and automatic audio muxing into the final MP4.
- Automatic rotation/cooldown for multiple Gemini and ElevenLabs keys.
- `/api/yolo/status` and `/api/elevenlabs/status` operational endpoints.
- GitHub Actions debug APK artifact.

### Secrets

Do **not** put Gemini or ElevenLabs keys in React/Vite source, APK assets, GitHub files, or this ZIP. Set them as server/Render secrets:
`GEMINI_API_KEYS` and `ELEVENLABS_API_KEYS`.

The keys pasted into the chat should be treated as exposed credentials and replaced before production use. ElevenLabs explicitly recommends keeping API keys server-side and rotating/revoking exposed keys. See the official documentation:
https://elevenlabs.io/docs/overview/administration/workspaces/api-keys

### Local YOLO

Install the backend Python dependencies from `yolo/requirements.txt`, then the Node server automatically invokes `yolo/track_football.py` during video analysis.


## Unified Football CLI — YOLO + Player Target + RIFE

The project now exposes a single Python entry point:

```bash
pip install -r requirements.txt
python main.py --video your_video.mp4 --player RAPHINHA
```

For the full cinematic path:

```bash
python main.py --video your_video.mp4 --player RAPHINHA --cinematic --rife 2x --export-json
```

The CLI runs the custom YOLO tracker, exports `tracking.json`, optionally creates a cinematic FFmpeg master, and optionally applies RIFE 2x/4x interpolation.

**Important:** `--player RAPHINHA` filters detections by detector class name. A generic person/football detector cannot identify a named player from appearance alone. For reliable named-player selection, train the supplied YOLO model with player-identity classes or add a jersey/face identity model.


## ULTRA v5 pipeline

The current pipeline is `2026-09-27-yolo-elevenlabs-rife-ultra-v5`.

### What changed

- Exact YOLO bounding boxes are exported into `tracking.json`.
- Player Re-ID accepts multiple reference images:
  `--player-reference front.jpg --player-reference side.jpg`
- Re-ID uses confidence and margin gates and can return `UNKNOWN` instead of forcing a player match.
- Highlight scoring combines detector confidence, player prominence, local detection density, motion and acceleration, while avoiding scene cuts.
- RIFE is scene-aware when highlights/cinematic mode is enabled: each selected shot is interpolated independently before concatenation, so interpolation does not cross hard cuts.
- `--no-scene-aware-rife` is available when whole-master interpolation is explicitly desired.
- Android CI now produces both Debug and unsigned Release APK artifacts.

Example:

```bash
python main.py \
  --video match.mp4 \
  --player RAPHINHA \
  --player-reference raphinha_front.jpg \
  --player-reference raphinha_side.jpg \
  --cinematic \
  --rife 2x \
  --highlight-count 10
```

A named player is not asserted solely from the string `--player RAPHINHA`. The detector class, reference appearance and confidence gate must provide evidence. If the evidence is insufficient, the pipeline keeps the result unresolved instead of silently assigning the wrong track.


## EXCEPTIONAL v7 — Production Pipeline

The pipeline is now designed as an explainable production system:

`Video → YOLO/ByteTrack → Identity Gate → Football Event Engine → Scene/Highlight Ranking → Scene-aware RIFE → Cinematic Master → Quality Gate → Manifest`

### New capabilities

- **Football Event Engine** detects ball/player proximity when a ball class is present and combines it with motion, acceleration and detector confidence.
- **Explainable highlights** include `reason`, `event_score`, `ball_proximity` and motion fields.
- **Quality Gate** uses `ffprobe` plus decode smoke tests before the pipeline can report success.
- **Pipeline Manifest** records stages, timestamps, cache fingerprint, errors and final artifacts.
- **Deterministic cache fingerprint** prevents unnecessary YOLO re-analysis when the same video/model/options are reused.
- **Failure-safe output**: a failed quality gate cannot be reported as a successful cinematic render.
- **Scene-aware RIFE** remains enabled by default when RIFE is requested with highlights/cinematic mode.

Example:

```bash
python main.py \
  --video match.mp4 \
  --player RAPHINHA \
  --player-reference raphinha_front.jpg \
  --player-reference raphinha_side.jpg \
  --cinematic \
  --rife 2x \
  --highlight-count 10
```

The pipeline deliberately does not claim a named player without sufficient evidence. If Re-ID is ambiguous, the identity remains unresolved.


## EXCEPTIONAL v8 — Football Director

The pipeline now includes a Football Director commentary layer. Instead of generating one long robotic voice-over, Gemini creates sparse Arabic commentary lines tied to verified timeline moments; each line receives adaptive delivery settings and is rendered independently with ElevenLabs, then positioned on the final 64-second timeline. Voice ducking automatically lowers the stadium/music bed while the commentator speaks and restores it between phrases.

This is deliberately football-specific: the director can leave silence before impact, intensify the climax, and avoid inventing goals, scores, player identities, or events not verified in the source timeline.

## v10 — Football Cinematic Intelligence

v10 upgrades the local intelligence layer from generic motion scoring to explainable football-event candidates:

- player-ball engagement continuity
- pressure candidates from nearby tracked players
- direction-change candidates
- explosive runs and ball-speed evidence
- evidence-weighted tension peaks and replay candidates
- explicit distinction between observed evidence and uncertain action candidates
- story roles: hook, setup, escalation, impact, reaction, climax, outro
- silence-before-impact and evidence-gated replay decisions

The system intentionally does **not** assert goals, shots, passes, identities, or scores without sufficient visual evidence.


## RENDER INTEGRITY FIX

The v10.0.3-MADNESS5 release requires the frontend and backend to use the same exceptional-v10 backend, adds a cache-busting final-video URL, and refuses to publish a final master if the renderer accidentally points at the uploaded source. The cinematic renderer performs real FFmpeg transforms per timeline clip before concatenation.
