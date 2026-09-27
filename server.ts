import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { GoogleGenAI, GenerateVideosOperation, Type } from '@google/genai';
import { createUserContent, createPartFromUri } from '@google/genai';
import cors from 'cors';
import { exec, spawn } from 'child_process';
import util from 'util';
import { ffmpegEngine, FFmpegProgress } from './server/ffmpegEngine';
import { storage } from './server/storage';
import { geminiRotator, isQuotaError, isOverloadError, isInvalidKeyError } from './server/geminiRotator';

const execPromise = util.promisify(exec);

dotenv.config();

// The rotator module is evaluated (and its constructor runs) during the hoisted
// imports, i.e. BEFORE dotenv loads .env. Reload now that the environment is set.
geminiRotator.reload();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const BUILD_VERSION = '2026-09-27-exceptional-v10.0.3-madness5';

// Enable CORS for frontend requests
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Range', 'Accept'],
}));

app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));

// Health Check Endpoint as specified in requirements
app.get('/health', (req, res) => {
  res.json({
    success: true,
    service: 'fotbal-backend',
    storage: storage.driver,
  });
});

app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    service: 'fotbal-backend',
    storage: storage.driver,
  });
});

app.get('/api/version', (req, res) => {
  res.json({
    success: true,
    service: 'fotbal-backend',
    buildVersion: BUILD_VERSION,
    pipeline: 'football-director-v10 + evidence-engine + Gemini-video-analysis + YOLO-ByteTrack + ReID + event-engine + beat-sync + scene-aware-RIFE + ffmpeg + ElevenLabs + optional-Veo',
    geminiKeys: geminiRotator.count(),
    activeGeminiKeys: geminiRotator.activeCount(),
  });
});

// Gemini key pool health — used by the UI to show rotation status and by ops to
// confirm every configured key is valid/active.
app.get('/api/keys/status', (req, res) => {
  res.json({ success: true, ...geminiRotator.status() });
});

// Force a reload of the key pool from the environment (no restart needed).
app.post('/api/keys/reload', (req, res) => {
  geminiRotator.reload();
  res.json({ success: true, ...geminiRotator.status() });
});

// Serve static videos directory with CORS and Range headers for mobile streaming.
// The directory comes from the storage layer so it points at the Render Disk
// (or a plain local folder in dev) instead of a hardcoded ephemeral path.
const videosDir = storage.mediaDir;
if (!fs.existsSync(videosDir)) {
  fs.mkdirSync(videosDir, { recursive: true });
}
app.use('/videos', (req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');
  next();
}, express.static(videosDir));

// Multer storage for real user uploaded football videos
const uploadStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, videosDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || '.mp4';
    cb(null, `uploaded_match_${Date.now()}${ext}`);
  },
});
const upload = multer({ storage: uploadStorage, limits: { fileSize: 500 * 1024 * 1024 } });

// Shared server-side Gemini client.
// NOTE: the pipeline no longer uses a single global client. It rotates across
// every configured GEMINI_API_KEYS entry via `geminiRotator` so one exhausted
// key cannot block a whole render. `apiKey` is kept only for backward-compatible
// checks (e.g. "is any key configured?").
const apiKey = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '').split(/[\s,;]+/).filter(Boolean)[0] || '';
const ai = new GoogleGenAI({
  apiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

function anyGeminiKeyConfigured(): boolean {
  return geminiRotator.hasKeys();
}


// ---------------------------------------------------------------------------
// Server-side Veo 3.1 enhancement pipeline.
// The old frontend Veo manager called endpoints that did not exist on the
// backend, so AI-enhanced renders silently fell back to source-only FFmpeg.
// This helper makes Veo a real part of the render pipeline and stores generated
// clips locally so FFmpeg can actually splice them into the final master.
// ---------------------------------------------------------------------------
const veoJobs = new Map<string, any>();

function veoModelName() {
  return process.env.VEO_MODEL || 'veo-3.1-generate-preview';
}

function isVeoConfigured() {
  return anyGeminiKeyConfigured();
}

async function extractReferenceFrame(inputPath: string, atSeconds: number, outPath: string) {
  const t = Math.max(0, Number(atSeconds) || 0);
  await execPromise(`ffmpeg -y -ss ${t.toFixed(3)} -i "${inputPath}" -frames:v 1 -vf "scale=720:-2:force_original_aspect_ratio=decrease" -q:v 2 "${outPath}"`);
  return fs.readFileSync(outPath).toString('base64');
}

async function generateVeoShotServer(prompt: string, imagePath: string, outPath: string, onProgress?: (stage: string) => void) {
  if (!isVeoConfigured()) throw new Error('No Gemini API key configured; Veo AI enhancement is unavailable.');
  onProgress?.('Submitting Veo 3.1 cinematic shot...');

  const imageBytes = fs.readFileSync(imagePath).toString('base64');

  // One atomic Veo attempt: submit + poll + download. If the key hits a quota
  // or overload error, the rotator replays the whole attempt on the next key.
  return geminiRotator.run('veo-generate', async (client, state) => {
    let operation: any = await (client.models as any).generateVideos({
      model: veoModelName(),
      prompt,
      image: { imageBytes, mimeType: 'image/jpeg' },
      config: {
        aspectRatio: '9:16',
        resolution: process.env.VEO_RESOLUTION || '720p',
        numberOfVideos: 1,
      },
    });

    const maxPolls = Math.max(6, Number(process.env.VEO_MAX_POLLS || 30));
    for (let poll = 0; poll < maxPolls; poll++) {
      if (!operation?.done) {
        onProgress?.(`Veo 3.1 rendering neural frames (${Math.min(96, 30 + poll * 2)}%)...`);
        await new Promise((resolve) => setTimeout(resolve, 10000));
        operation = await (client.operations as any).getVideosOperation({ operation });
      }
      if (operation?.done) break;
    }

    if (!operation?.done) throw new Error('Veo generation timed out.');
    const generated = operation?.response?.generatedVideos?.[0]?.video;
    if (!generated?.uri) throw new Error('Veo completed without a generated video URI.');

    onProgress?.('Downloading generated Veo clip...');
    let uri = String(generated.uri);
    const key = geminiRotator.keyForIndex(state.index);
    if (key && !/[?&]key=/.test(uri)) uri += `${uri.includes('?') ? '&' : '?'}key=${encodeURIComponent(key)}`;
    const response = await fetch(uri);
    if (!response.ok) throw new Error(`Veo video download failed (${response.status}).`);
    const bytes = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(outPath, bytes);
    if (!fs.existsSync(outPath) || fs.statSync(outPath).size < 10000) throw new Error('Downloaded Veo clip is empty.');
    onProgress?.('Veo cinematic shot ready.');
    return outPath;
  });
}

// Compatibility endpoints used by the original frontend VeoGenerationManager.
// They now talk to the same real Veo operation objects as the render pipeline.
app.post('/api/generate-veo-shot', async (req, res) => {
  try {
    if (!isVeoConfigured()) return res.status(503).json({ success: false, errorMessage: 'No Gemini API key configured for Veo.' });
    const prompt = String(req.body?.prompt || '').trim();
    if (!prompt) return res.status(400).json({ success: false, errorMessage: 'A Veo prompt is required.' });

    // Acquire a healthy key and remember its index: the operation and the
    // generated URI are both scoped to that key, so polling/downloading must
    // use the same one.
    const state = geminiRotator.acquire();
    const client = geminiRotator.clientFor(state.index);
    const operation: any = await (client.models as any).generateVideos({
      model: veoModelName(),
      prompt,
      config: {
        aspectRatio: String(req.body?.aspectRatio || '9:16'),
        resolution: String(req.body?.resolution || process.env.VEO_RESOLUTION || '720p'),
        numberOfVideos: 1,
      },
    });
    geminiRotator.reportSuccess(state);
    const operationName = operation?.name || `veo-${Date.now()}`;
    veoJobs.set(operationName, { operation, keyIndex: state.index, createdAt: Date.now() });
    res.json({ success: true, operationName });
  } catch (err: any) {
    console.error('[VEO API] generate failed:', err);
    const status = isQuotaError(err) ? 429 : 500;
    res.status(status).json({ success: false, errorMessage: err?.message || 'Veo generation failed.' });
  }
});

app.post('/api/video-status', async (req, res) => {
  try {
    const operationName = String(req.body?.operationName || '').trim();
    if (!operationName) return res.status(400).json({ done: false, errorMessage: 'operationName is required.' });
    const job = veoJobs.get(operationName);
    if (!job) return res.status(404).json({ done: false, errorMessage: 'Veo operation not found or expired.' });
    const client = geminiRotator.clientFor(job.keyIndex ?? 0);
    const operation = await (client.operations as any).getVideosOperation({ operation: job.operation });
    job.operation = operation;
    veoJobs.set(operationName, job);
    res.json({ done: Boolean(operation?.done), hasVideo: Boolean(operation?.response?.generatedVideos?.length), errorMessage: operation?.error?.message || undefined });
  } catch (err: any) {
    res.status(500).json({ done: false, errorMessage: err?.message || 'Unable to poll Veo operation.' });
  }
});

app.post('/api/video-download', async (req, res) => {
  try {
    const operationName = String(req.body?.operationName || '').trim();
    const job = veoJobs.get(operationName);
    const video = job?.operation?.response?.generatedVideos?.[0]?.video;
    if (!video?.uri) return res.status(404).json({ success: false, errorMessage: 'Generated Veo video is not ready.' });
    let uri = String(video.uri);
    const key = geminiRotator.keyForIndex(job?.keyIndex ?? 0);
    if (key && !/[?&]key=/.test(uri)) uri += `${uri.includes('?') ? '&' : '?'}key=${encodeURIComponent(key)}`;
    const response = await fetch(uri);
    if (!response.ok) return res.status(response.status).json({ success: false, errorMessage: `Veo download failed (${response.status}).` });
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Cache-Control', 'no-store');
    if (response.body) {
      // Node 22 supports Readable.fromWeb for Fetch response bodies.
      const { Readable } = await import('stream');
      Readable.fromWeb(response.body as any).pipe(res);
    } else {
      res.end(Buffer.from(await response.arrayBuffer()));
    }
  } catch (err: any) {
    res.status(500).json({ success: false, errorMessage: err?.message || 'Unable to download Veo video.' });
  }
});

async function prepareVeoEnhancements(inputPath: string, editPlan: any, generationTier: string, onProgress?: (progress: FFmpegProgress) => void) {
  if (!['AI ENHANCED', 'AI CINEMATIC'].includes(generationTier)) return editPlan;
  if (!isVeoConfigured()) {
    console.warn('[VEO] AI tier selected but GEMINI_API_KEY is missing; continuing with real-footage-only render.');
    return editPlan;
  }

  const maxClips = generationTier === 'AI CINEMATIC' ? 5 : 2;
  const timeline = Array.isArray(editPlan?.timeline) ? editPlan.timeline : [];
  const candidates = timeline
    .map((clip: any, index: number) => ({ clip, index }))
    .filter(({ clip }: any) => clip.veo_needed || ['close_up', 'extreme_close_up', 'detail', 'reaction'].includes(clip.shot_type))
    .filter(({ clip }: any) => clip.beat_role !== 'climax')
    .sort((a: any, b: any) => Number(b.clip.importance || 0) - Number(a.clip.importance || 0));

  const chosen = candidates.slice(0, maxClips);
  if (!chosen.length) return editPlan;

  const sessionDir = path.join('/tmp/football_engine/work', `veo_${Date.now()}`);
  fs.mkdirSync(sessionDir, { recursive: true });
  const updated = JSON.parse(JSON.stringify(editPlan));

  try {
    for (let n = 0; n < chosen.length; n++) {
      const { clip, index } = chosen[n];
      const framePath = path.join(sessionDir, `ref_${n}.jpg`);
      const veoPath = path.join(sessionDir, `veo_${n}.mp4`);
      const sourceAt = Number(clip.source_start) || 0;
      onProgress?.({ percent: Math.min(35, 18 + n * 5), stage: `Preparing AI cinematic insert ${n + 1}/${chosen.length}...` });
      await extractReferenceFrame(inputPath, sourceAt, framePath);

      const prompt = clip.veo_prompt || [
        'Create a premium vertical football cinematic insert based on the supplied reference frame.',
        'Preserve the same player appearance, kit colors, stadium context and overall visual identity as closely as possible.',
        `Shot type: ${clip.shot_type || 'cinematic action'}.`,
        'Use realistic professional sports cinematography, shallow depth of field, natural motion, dramatic stadium lighting, subtle handheld/tracking movement, crisp details, realistic skin and fabric, no logos or added text.',
        'This is an editorial bridge shot for a 9:16 football reel; do not invent a score, celebration, or specific event that is not implied by the reference frame.',
        `Observed source moment: ${String(clip.action || 'football action').slice(0, 180)}.`,
      ].join(' ');

      try {
        await generateVeoShotServer(prompt, framePath, veoPath, (stage) => {
          onProgress?.({ percent: Math.min(70, 38 + Math.round((n / chosen.length) * 30)), stage: `AI shot ${n + 1}/${chosen.length}: ${stage}` });
        });
        updated.timeline[index].veo_needed = true;
        updated.timeline[index].veo_status = 'ready';
        updated.timeline[index].veo_prompt = prompt;
        updated.timeline[index].veo_local_path = veoPath;
      } catch (err: any) {
        console.warn(`[VEO] shot ${n + 1} failed:`, err?.message || err);
        updated.timeline[index].veo_status = 'fallback';
        updated.timeline[index].veo_local_path = undefined;
      }
    }
    return updated;
  } catch (err) {
    try { fs.rmSync(sessionDir, { recursive: true, force: true }); } catch {}
    throw err;
  }
}

// Live render progress tracking
let currentRenderProgress: FFmpegProgress = { percent: 0, stage: 'Idle' };

// Real local football video presets for instant testability
const SAMPLE_FOOTBALL_CLIPS = [
  {
    id: 'sample-1',
    title: 'El Clasico Final (Live Match Footage)',
    description: 'Real 75s match footage: striker sprint, dazzling solo dribble run, 45s strike on goal, net bulging, 90+4 goal and corner flag celebration.',
    duration: 75.0,
    sourceUrl: '/videos/football_match.mp4',
    posterUrl: '/videos/poster_10s.jpg',
    localPath: 'public/videos/football_match.mp4',
    tags: ['Climax Goal', 'Solo Dribble', 'Corner Celebration', 'Sprint'],
    defaultSubject: 'Striker #10',
  },
  {
    id: 'sample-2',
    title: 'Match Highlights (60s Reel)',
    description: 'High stakes 60s dynamic sequence featuring midfield duels, turns and counter-attacks.',
    duration: 60.1,
    sourceUrl: '/videos/sample_match.mp4',
    posterUrl: '/videos/poster_02s.jpg',
    localPath: 'public/videos/sample_match.mp4',
    tags: ['Midfield Sprint', 'Turnover', 'Counter Attack'],
    defaultSubject: 'Winger / Playmaker',
  },
];

// Motivational captions pool for the celebration close-up (3-6 words, all caps, second-person bold tone)
const MOTIVATIONAL_CELEBRATION_CAPTIONS = [
  'MAKE THEM REMEMBER YOU',
  'THEY CANNOT STOP YOU NOW',
  'OUTWORK THEM IN SILENCE',
  'EARN WHAT IS YOURS',
  'DEMAND YOUR OWN GREATNESS',
  'NEVER DOUBT YOUR MOMENT',
  'LET YOUR GAME SPEAK',
  'PROVE THEM WRONG EVERY TIME',
];

// Prevent a long Gemini-selected interval from becoming an almost-original
// export. This only subdivides timestamps Gemini already verified.
function densifyReferenceTimeline(timeline: any[], targetMin = 14, targetMax = 22): any[] {
  const chunks: any[] = [];
  const maxSourceChunk = 4.0;
  for (const clip of timeline) {
    const start = Number(clip.source_start);
    const end = Number(clip.source_end);
    const span = end - start;
    if (!Number.isFinite(start) || !Number.isFinite(end) || span <= 0.08) continue;
    const parts = Math.max(1, Math.ceil(span / maxSourceChunk));
    for (let p = 0; p < parts; p++) {
      const a = start + span * p / parts;
      const b = start + span * (p + 1) / parts;
      if (b - a < 0.08) continue;
      const copy = { ...clip, source_start: a, source_end: b };
      if (p > 0) {
        copy.text = '';
        copy.action = `${String(clip.action || 'football moment').slice(0, 100)} — detail ${p + 1}`;
        copy.beat_role = 'escalation';
      }
      if (p % 3 === 1) copy.zoom_start = Math.max(Number(copy.zoom_start || 1.02), 1.10);
      if (p % 3 === 2) copy.zoom_start = Math.max(Number(copy.zoom_start || 1.02), 1.18);
      chunks.push(copy);
    }
  }
  if (chunks.length >= targetMin) return chunks.slice(0, targetMax);

  let guard = 0;
  while (chunks.length < targetMin && guard++ < 32) {
    let longest = -1, longestSpan = 0;
    chunks.forEach((c: any, i: number) => {
      const span = Number(c.source_end) - Number(c.source_start);
      if (span > longestSpan && span > 1.2) { longest = i; longestSpan = span; }
    });
    if (longest < 0) break;
    const c = chunks[longest];
    const mid = Number(c.source_start) + longestSpan / 2;
    chunks.splice(longest, 1,
      { ...c, source_end: mid },
      { ...c, source_start: mid, text: '', action: `${String(c.action || 'football moment').slice(0, 100)} — detail`, zoom_start: Math.max(Number(c.zoom_start || 1.02), 1.12) }
    );
  }
  return chunks.slice(0, targetMax);
}

// ---------------------------------------------------------------------------
// Edit-plan validation. IMPORTANT: this function never invents football events.
// Gemini must supply the real source timestamps. If the plan is empty/invalid,
// the request fails instead of silently falling back to a fake timeline.
// ---------------------------------------------------------------------------
function validateAndEnforce64sEditPlan(data: any, videoDuration: number, styleName: string = ''): any {
  const duration = 64;
  if (!data || !Array.isArray(data.timeline) || data.timeline.length === 0) {
    throw new Error('Gemini returned an empty edit timeline. No synthetic fallback is allowed.');
  }
  if (!Number.isFinite(videoDuration) || videoDuration < 1) {
    throw new Error('Invalid source video duration.');
  }

  const subject = {
    name: typeof data.subject?.name === 'string' && data.subject.name.trim() ? data.subject.name.trim() : 'Main player',
    confidence: Math.max(0, Math.min(1, Number(data.subject?.confidence) || 0.5)),
  };

  // The reference reel uses a dense, emotional vertical montage. Keep the real
  // source moments selected by Gemini and reject anything outside the uploaded file.
  const raw = data.timeline
    .map((clip: any, idx: number) => {
      const s = Number(clip.source_start);
      const e = Number(clip.source_end);
      if (!Number.isFinite(s) || !Number.isFinite(e) || e <= s) return null;
      const start = Math.max(0, Math.min(videoDuration - 0.05, s));
      const end = Math.max(start + 0.08, Math.min(videoDuration, e));
      if (end <= start) return null;
      const speed = Math.max(0.45, Math.min(2.2, Number(clip.speed) || 1));
      const zStart = Math.max(1, Math.min(1.55, Number(clip.zoom_start) || 1.02));
      const zEnd = Math.max(zStart, Math.min(1.65, Number(clip.zoom_end) || zStart + 0.08));
      const transitionRaw = String(clip.transition || 'hard_cut').toLowerCase();
      const transition = ['hard_cut', 'directional_blur', 'flash'].includes(transitionRaw) ? transitionRaw : 'hard_cut';
      const text = typeof clip.text === 'string' ? clip.text.replace(/\s+/g, ' ').trim().slice(0, 72).toUpperCase() : '';
      return {
        timeline_index: idx,
        source_start: Number(start.toFixed(3)),
        source_end: Number(end.toFixed(3)),
        action: typeof clip.action === 'string' ? clip.action.trim() : `Real football moment ${idx + 1}`,
        importance: Math.max(1, Math.min(10, Number(clip.importance) || 5)),
        speed,
        zoom_start: Number(zStart.toFixed(2)),
        zoom_end: Number(zEnd.toFixed(2)),
        crop_x: Math.max(0.05, Math.min(0.95, Number(clip.crop_x ?? 0.5))),
        crop_y: Math.max(0.05, Math.min(0.95, Number(clip.crop_y ?? 0.5))),
        transition,
        text,
        narration: typeof clip.narration === 'string' ? clip.narration.replace(/\s+/g, ' ').trim().slice(0, 90) : '',
        shot_type: ['wide','medium','close_up','extreme_close_up','eye_close_up','action','reaction','crowd','detail'].includes(String(clip.shot_type)) ? String(clip.shot_type) : 'action',
        beat_role: ['hook','setup','escalation','impact','reaction','climax','outro'].includes(String(clip.beat_role)) ? String(clip.beat_role) : undefined,
        veo_needed: false,
        veo_prompt: '',
      };
    })
    .filter(Boolean) as any[];

  if (raw.length < 4) throw new Error('Gemini returned too few usable real moments for a cinematic montage.');

  const denseRaw = densifyReferenceTimeline(raw, 14, 22);
  raw.length = 0;
  raw.push(...denseRaw.map((c: any, idx: number) => ({ ...c, timeline_index: idx })));

  // Preserve Gemini's editorial timing when valid; otherwise derive a cinematic rhythm
  // from real source moments. Never invent a football event or source timestamp.
  const requestedDurations = raw.map((c) => {
    return Math.max(0.8, Math.min(4.5, (c.source_end - c.source_start) / c.speed));
  });
  const totalRequested = requestedDurations.reduce((a, b) => a + b, 0);
  let cursor = 0;
  const timeline = raw.map((c, i) => {
    const normalized = i === raw.length - 1 ? 64 - cursor : Math.max(0.35, 64 * requestedDurations[i] / totalRequested);
    const start = cursor;
    const end = i === raw.length - 1 ? 64 : Math.min(64, cursor + normalized);
    cursor = end;
    return { ...c, output_start: Number(start.toFixed(3)), output_end: Number(end.toFixed(3)) };
  });
  timeline[timeline.length - 1].output_end = 64;

  // AI tiers now mark suitable inserts for the real server-side Veo pipeline.
  // ORIGINAL FOOTAGE ONLY remains strictly source-only.
  const requestedTier = String(data.generationTier || 'ORIGINAL FOOTAGE ONLY');
  if (requestedTier !== 'ORIGINAL FOOTAGE ONLY') {
    const maxVeo = requestedTier === 'AI CINEMATIC' ? 5 : 2;
    const eligible = timeline.filter((c: any) => c.beat_role !== 'climax' && ['close_up','extreme_close_up','detail','reaction'].includes(c.shot_type));
    eligible.slice(0, maxVeo).forEach((c: any) => { c.veo_needed = true; });
  }

  // Reference-style captions are short, sparse, white, centered and editorial.
  // If Gemini supplied none, generate only non-factual editorial micro-copy from the
  // verified action label; this does not claim an event that wasn't observed.
  // PSYCHOLOGICAL DRAMA uses first-person inner-monologue wording rendered in a
  // cinematic SERIF font (the FFmpeg engine applies the font/size per style).
  const isDrama = String(styleName || data.styleName || '').toUpperCase().includes('PSYCHOLOGICAL');
  const captionPool = ['WATCH THIS', 'ONE MORE STEP', 'NO WAY BACK', 'LOCKED IN', 'TOO CLOSE', 'THE MOMENT', 'RIGHT NOW', 'ICE COLD', 'GAME ON', 'THAT TOUCH'];
  const monologuePool = ['COME CLOSER', 'ONE MORE STEP', 'NOW YOU ARE MINE', 'I SAW IT COMING', 'TOO SLOW', 'WATCH THE EYES', 'THIS IS MY MOMENT', 'DO NOT BLINK'];
  timeline.forEach((c, i) => {
    if (!c.text) {
      const action = String(c.action || '').toLowerCase();
      if (i === 0) c.text = isDrama ? 'COME CLOSER' : 'WATCH THIS';
      else if (/goal|score|finish|shot|strike|net/.test(action)) c.text = isDrama ? 'NOW' : 'THE MOMENT';
      else if (/skill|drib|nutmeg|turn|feint|touch/.test(action)) c.text = isDrama ? 'TOO SLOW' : 'THAT TOUCH';
      else if (/reaction|celebr|crowd/.test(action)) c.text = isDrama ? 'THIS IS MY MOMENT' : 'LIVE THE MOMENT';
      else c.text = isDrama ? monologuePool[i % monologuePool.length] : captionPool[i % captionPool.length];
    }
    // Both styles render uppercase; the difference is the wording and the serif
    // font applied by the FFmpeg engine.
    c.text = c.text.replace(/[^\p{L}\p{N}À-ÖØ-Þ\s'!?.,-]/gu, '').replace(/\s+/g, ' ').trim().slice(0, 46).toUpperCase();
    c.narration = (c.narration && c.narration.trim()) ? c.narration.trim().slice(0, 90) : (isDrama ? c.text : '');
  });

  let flashSeen = false;
  for (const c of timeline) {
    if (c.transition === 'flash') {
      if (flashSeen) c.transition = 'hard_cut';
      else flashSeen = true;
    }
  }

  const referenceGrade = data.color_grade || {};
  return {
    duration,
    aspect_ratio: '9:16',
    subject,
    timeline,
    music: {
      style: data.music?.style || 'Emotional cinematic football / dark trap pulse',
      bpm: Math.max(90, Math.min(170, Number(data.music?.bpm) || 126)),
      energy_curve: Array.isArray(data.music?.energy_curve) ? data.music.energy_curve : [0.72,0.58,0.65,0.78,0.9,1,0.82,0.38],
    },
    color_grade: {
      contrast: Math.max(1, Math.min(1.45, Number(referenceGrade.contrast) || 1.20)),
      saturation: Math.max(0.85, Math.min(1.35, Number(referenceGrade.saturation) || 1.04)),
      highlights: Math.max(-0.25, Math.min(0.15, Number(referenceGrade.highlights) || -0.04)),
      shadows: Math.max(-0.1, Math.min(0.2, Number(referenceGrade.shadows) || 0.02)),
      grain: Math.max(0, Math.min(0.22, Number(referenceGrade.grain) || 0.035)),
    },
    style_profile: {
      name: isDrama ? 'PSYCHOLOGICAL DRAMA' : 'REFERENCE CINEMATIC REEL',
      average_shot_duration: Number((64 / timeline.length).toFixed(2)),
      zoom_intensity: isDrama ? 0.85 : 0.78,
      transition_frequency: 0.16,
      slow_motion_frequency: isDrama ? 0.55 : 0.30,
      text_frequency: 0.9,
      caption_style: isDrama
        ? 'cinematic white SERIF inner-monologue, ALL CAPS, centered lower third with soft shadow'
        : 'small white uppercase centered near lower third with soft shadow',
      visual_language: isDrama
        ? 'extreme eye close-ups, footwork and sweat details, slow-motion duels, speed-ramp impact, hard cuts, high contrast, deep shadows'
        : 'tight vertical crops, intimate player close-ups, football details, hard cuts, restrained flash impacts, dramatic pacing',
    },
    style_name: isDrama ? 'PSYCHOLOGICAL DRAMA' : 'REFERENCE CINEMATIC REEL',
  };
}

function extractJsonObject(text: string): any {
  const cleaned = String(text || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '');
  try { return JSON.parse(cleaned); } catch {}
  const first = cleaned.indexOf('{');
  const last = cleaned.lastIndexOf('}');
  if (first >= 0 && last > first) return JSON.parse(cleaned.slice(first, last + 1));
  throw new Error('Gemini response was not valid JSON.');
}

function isRetryableGeminiError(err: any): boolean {
  const message = String(err?.message || err || '');
  return /503|high demand|overloaded|temporar|429|resource exhausted|timeout|DEADLINE_EXCEEDED/i.test(message);
}

/**
 * Uploads the REAL video to the Gemini Files API, waits for processing, and asks
 * Gemini to return strict JSON — all under a single rotating API key.
 *
 * The Files API is key-scoped: a file uploaded with key A can only be analysed
 * with key A. `geminiRotator.run` therefore wraps the ENTIRE unit (upload +
 * poll + generateContent) so a fresh attempt re-uploads with the next key.
 */
async function generateGeminiJsonWithVideo(
  localPath: string,
  prompt: string,
  systemInstruction: string,
  model: string,
  mimeType: string = 'video/mp4',
  onProgress?: (p: { percent: number; stage: string }) => void
) {
  if (!anyGeminiKeyConfigured()) throw new Error('No Gemini API key is configured on the backend.');
  if (!fs.existsSync(localPath)) throw new Error(`Uploaded video is missing on server: ${localPath}`);

  const maxAttempts = Math.min(geminiRotator.count(), Number(process.env.GEMINI_MAX_KEY_ATTEMPTS || 4));

  return geminiRotator.run(`analyze-video(${model})`, async (client, state) => {
    let videoFile: any;
    try {
      onProgress?.({ percent: 8, stage: `Uploading source video to Gemini (key ${state.index + 1}/${geminiRotator.count()})...` });
      videoFile = await client.files.upload({ file: localPath, config: { mimeType } });
      onProgress?.({ percent: 12, stage: 'Gemini is processing the uploaded footage...' });
      const deadline = Date.now() + 180000;
      let progressTick = 0;
      while (String(videoFile.state || '') !== 'ACTIVE') {
        if (String(videoFile.state) === 'FAILED') throw new Error('Gemini failed to process the uploaded video.');
        if (Date.now() > deadline) throw new Error('Timed out while Gemini was processing the uploaded video.');
        await new Promise((r) => setTimeout(r, 3000));
        videoFile = await client.files.get({ name: videoFile.name });
        progressTick += 1;
        const pct = Math.min(14, 12 + progressTick);
        onProgress?.({ percent: pct, stage: `Gemini is processing the footage... (${progressTick * 3}s)` });
      }

      onProgress?.({ percent: 15, stage: 'Gemini is analyzing the match footage and building the edit plan...' });
      let lastErr: any;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const response = await client.models.generateContent({
            model,
            contents: createUserContent([
              createPartFromUri(videoFile.uri, videoFile.mimeType || 'video/mp4'),
              prompt,
            ]),
            config: {
              systemInstruction,
              responseMimeType: 'application/json',
            },
          });
          return extractJsonObject(response.text || '');
        } catch (err: any) {
          lastErr = err;
          // A quota error must bubble up so the rotator can switch to the next key.
          if (isQuotaError(err)) throw err;
          if (!isRetryableGeminiError(err) || attempt === 2) throw err;
          await new Promise((r) => setTimeout(r, 3000 * Math.pow(2, attempt)));
        }
      }
      throw lastErr || new Error('Gemini request failed.');
    } finally {
      // Files are temporary analysis assets; best-effort cleanup with the SAME key.
      try {
        if (videoFile?.name && (client.files as any).delete) await (client.files as any).delete({ name: videoFile.name });
      } catch {}
    }
  }, maxAttempts);
}

// 1. GET /api/presets
app.get('/api/presets', (req, res) => {
  res.json({
    presets: SAMPLE_FOOTBALL_CLIPS,
    styles: [
      { id: 'CINEMATIC SPORTS', label: 'Cinematic Sports', description: 'Dynamic slow-mo push-ins, high contrast, crisp stadium lighting and punchy transitions.' },
      { id: 'DARK FOOTBALL DOCUMENTARY', label: 'Dark Football Documentary', description: 'Moody anamorphic grade, subtle grain, introspective pacing, and thunderous beat drops.' },
      { id: 'HYPE / VIRAL FOOTBALL', label: 'Hype / Viral Football', description: 'Fast speed ramps, kinetic typography, flash impact cuts, and maximum bass-drop energy.' },
      { id: 'EMOTIONAL FOOTBALL STORY', label: 'Emotional Football Story', description: 'Slow-burn tension, orchestral strings, player close-up focus and triumphant hero climax.' },
      { id: 'REFERENCE CINEMATIC REEL', label: 'Reference Cinematic Reel', description: 'Premium vertical football montage: tight close-ups, dramatic details, emotional white captions, punch-ins, slow-motion accents and hard-cut pacing.' },
      { id: 'PSYCHOLOGICAL DRAMA', label: 'Psychological Drama', description: 'Anime-style football thriller: extreme close-ups on eyes, heavy slow motion, speed ramps on the decisive move, cinematic serif inner-monologue captions, deep contrast and epic mood.' },
    ],
  });
});

app.get('/presets', (req, res) => {
  res.redirect('/api/presets');
});

// 2. Video Upload endpoints (with safe JSON error handling and flexible field names)
const uploadMiddleware = (req: any, res: any, next: any) => {
  upload.any()(req, res, (err: any) => {
    if (err) {
      console.error('[MULTER UPLOAD ERROR]', err);
      return res.status(400).json({
        success: false,
        error: {
          code: err.code || 'UPLOAD_FAILED',
          message: err.message || 'File upload failed. Ensure the file is a valid video under 500MB.',
        },
      });
    }
    next();
  });
};

const handleVideoUpload = async (req: any, res: any) => {
  // Support 'video' field, 'file' field, or first uploaded file
  const file = req.file || (Array.isArray(req.files) && req.files.length > 0 ? req.files[0] : null);

  if (!file) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'NO_FILE',
        message: 'No video file provided in upload request.',
      },
    });
  }

  const allowedVideoExt = new Set(['.mp4', '.mov', '.webm', '.mkv', '.m4v', '.avi', '.ts']);
  const ext = path.extname(file.originalname || '').toLowerCase();
  if (!(file.mimetype || '').startsWith('video/') && !allowedVideoExt.has(ext)) {
    try { fs.rmSync(file.path, { force: true }); } catch {}
    return res.status(415).json({ success: false, error: { code: 'UNSUPPORTED_VIDEO', message: 'Please upload a supported video file.' } });
  }

  const filename = file.filename;
  const localPath = file.path;
  const posterFilename = `${path.parse(filename).name}_poster.jpg`;
  const posterPath = path.join(path.dirname(localPath), posterFilename);

  let duration = 60.0;
  let width = 1920;
  let height = 1080;
  let fps = 25.0;

  // Extract true video metadata using ffprobe on the uploaded file
  try {
    const probeCmd = `ffprobe -v error -show_entries format=duration -show_entries stream=width,height,r_frame_rate -of json "${localPath}"`;
    const { stdout } = await execPromise(probeCmd);
    const probeData = JSON.parse(stdout);
    if (probeData.format?.duration) {
      duration = Math.round(parseFloat(probeData.format.duration) * 100) / 100;
    }
    const videoStream = probeData.streams?.find((s: any) => s.width && s.height);
    if (videoStream) {
      width = videoStream.width;
      height = videoStream.height;
      if (videoStream.r_frame_rate && videoStream.r_frame_rate.includes('/')) {
        const [num, den] = videoStream.r_frame_rate.split('/').map(Number);
        if (den > 0) fps = Math.round((num / den) * 100) / 100;
      }
    }
  } catch (probeErr) {
    console.warn('ffprobe metadata probe failed, using defaults:', probeErr);
  }

  // Generate poster thumbnail from uploaded match footage
  try {
    const posterCmd = `ffmpeg -y -ss ${Math.min(0.5, Math.max(0.1, duration * 0.1))} -i "${localPath}" -vframes 1 -q:v 2 "${posterPath}"`;
    await execPromise(posterCmd);
  } catch (e) {
    console.warn('Poster generation for uploaded video skipped:', e);
  }

  // Persist to the storage backend (no-op locally; uploads to S3/R2 when configured)
  const storedVideo = await storage.publish(localPath);
  let posterUrl: string | undefined;
  if (fs.existsSync(posterPath)) {
    const storedPoster = await storage.publish(posterPath);
    posterUrl = storedPoster.url;
  }

  res.json({
    success: true,
    videoId: filename,
    videoUrl: storedVideo.url,
    posterUrl,
    localPath,
    filename,
    title: file.originalname,
    mimeType: file.mimetype || 'video/mp4',
    size: file.size,
    duration,
    width,
    height,
    fps,
  });
};

app.post('/api/upload-video', uploadMiddleware, handleVideoUpload);
app.post('/api/upload', uploadMiddleware, handleVideoUpload);
app.post('/upload', uploadMiddleware, handleVideoUpload);

// 3. POST /api/test-render-1
// Extracts first 5 seconds, converts to 9:16 (1080x1920)
app.post('/api/test-render-1', async (req, res) => {
  try {
    const inputPath = req.body.localPath;
    if (!inputPath || !fs.existsSync(inputPath)) return res.status(400).json({ success: false, error: 'Valid uploaded video localPath is required.' });

    const result = await ffmpegEngine.runTest1(inputPath);
    res.json({
      success: true,
      videoUrl: result.videoUrl,
      posterUrl: result.posterUrl,
      testName: 'TEST 1: 5s 9:16 Crop',
    });
  } catch (err: any) {
    console.error('Test 1 failed:', err);
    res.status(500).json({ error: err.message });
  }
});

// 4. POST /api/test-render-2
// Extracts 10s -> 15s, applies 0.7x speed + 10% zoom
app.post('/api/test-render-2', async (req, res) => {
  try {
    const inputPath = req.body.localPath;
    if (!inputPath || !fs.existsSync(inputPath)) return res.status(400).json({ success: false, error: 'Valid uploaded video localPath is required.' });

    const result = await ffmpegEngine.runTest2(inputPath);
    res.json({
      success: true,
      videoUrl: result.videoUrl,
      posterUrl: result.posterUrl,
      testName: 'TEST 2: 0.7x Speed + 10% Zoom',
    });
  } catch (err: any) {
    console.error('Test 2 failed:', err);
    res.status(500).json({ error: err.message });
  }
});

// 5. POST /api/test-render-3
// Adds burned-in text "TEST CINEMATIC" from 2s -> 4s
app.post('/api/test-render-3', async (req, res) => {
  try {
    const inputPath = req.body.localPath;
    if (!inputPath || !fs.existsSync(inputPath)) return res.status(400).json({ success: false, error: 'Valid uploaded video localPath is required.' });

    const result = await ffmpegEngine.runTest3(inputPath);
    res.json({
      success: true,
      videoUrl: result.videoUrl,
      posterUrl: result.posterUrl,
      testName: 'TEST 3: Burned-in Text Overlay',
    });
  } catch (err: any) {
    console.error('Test 3 failed:', err);
    res.status(500).json({ error: err.message });
  }
});

// RIFE post-processing: optional 2x/4x frame interpolation after the real render.
async function runRifeInterpolation(inputPath: string, outputPath: string, exp: number = 1) {
  if (process.env.RIFE_ENABLED === 'false') return { applied: false, outputPath: inputPath };
  const pythonBin = process.env.PYTHON_BIN || 'python3';
  const script = path.join(__dirname, 'yolo', 'rife_interpolate.py');
  if (!fs.existsSync(script)) throw new Error(`RIFE helper not found: ${script}`);
  await new Promise<void>((resolve, reject) => {
    const child = spawn(pythonBin, [script, '--input', inputPath, '--output', outputPath, '--exp', String(exp)], {
      env: { ...process.env }, stdio: ['ignore', 'pipe', 'pipe']
    });
    let stderr = '';
    child.stderr.on('data', d => { stderr += d.toString(); });
    child.on('error', reject);
    child.on('close', code => code === 0 ? resolve() : reject(new Error(stderr || `RIFE exited with ${code}`)));
  });
  if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size < 10000) {
    throw new Error('RIFE produced an invalid output video.');
  }
  return { applied: true, outputPath };
}

// 6. POST /api/render-full-cinematic
// Executes complete real FFmpeg video processing pipeline
app.post('/api/render-full-cinematic', async (req, res) => {
  try {
    const { localPath, editPlan, musicVolume = 0.8, originalVolume = 0.9, generationTier = 'ORIGINAL FOOTAGE ONLY' } = req.body;
    if (!localPath || !fs.existsSync(localPath)) {
      return res.status(400).json({ success: false, error: 'Valid uploaded video localPath is required.' });
    }

    currentRenderProgress = { percent: 0, stage: 'Initializing real FFmpeg render engine...' };

    const preparedPlan = await prepareVeoEnhancements(localPath, editPlan, generationTier, (p) => { currentRenderProgress = p; });
    const generatedVeoPaths = preparedPlan.timeline.map((c: any) => c.veo_local_path).filter((p: any) => typeof p === 'string');

    try {
      const result = await ffmpegEngine.renderFullCinematic(
        localPath,
        preparedPlan,
        musicVolume,
        originalVolume,
        (p) => {
          currentRenderProgress = p;
        }
      );

      let publishedPath = result.localPath;
      let rifeApplied = false;
      const rifeMultiplier = Number(req.body?.rifeMultiplier || process.env.RIFE_FPS_MULTIPLIER || 2);
      if (process.env.RIFE_ENABLED !== 'false' && (rifeMultiplier === 2 || rifeMultiplier === 4)) {
        currentRenderProgress = { percent: 94, stage: `RIFE frame interpolation ${rifeMultiplier}x...` };
        const rifeOut = path.join(path.dirname(publishedPath), `${path.parse(publishedPath).name}_${rifeMultiplier}x.mp4`);
        try {
          const rife = await runRifeInterpolation(publishedPath, rifeOut, rifeMultiplier === 4 ? 2 : 1);
          if (rife.applied) { publishedPath = rife.outputPath; rifeApplied = true; }
        } catch (rifeErr: any) {
          console.warn('[RIFE] interpolation failed; keeping FFmpeg master:', rifeErr?.message || rifeErr);
        }
      }
      currentRenderProgress = { percent: 100, stage: rifeApplied ? 'Cinematic render + RIFE complete.' : 'Cinematic render complete.' };
      res.json({
        success: true,
        ...result,
        localPath: publishedPath,
        videoUrl: publishedPath ? publishedPath.replace(videosDir, '/videos') : result.videoUrl,
        generationTier,
        aiEnhanced: generationTier !== 'ORIGINAL FOOTAGE ONLY' && generatedVeoPaths.length > 0,
        rifeApplied,
        rifeMultiplier: rifeApplied ? rifeMultiplier : 0,
      });
    } finally {
      // Generated bridge shots are temporary render assets; the published master is persistent.
      for (const filePath of generatedVeoPaths) {
        try { fs.rmSync(filePath, { force: true }); } catch {}
        try { fs.rmSync(path.dirname(filePath), { recursive: true, force: true }); } catch {}
      }
    }
  } catch (err: any) {
    console.error('Full cinematic render failed:', err);
    currentRenderProgress = { percent: 0, stage: `Render Error: ${err.message}` };
    res.status(500).json({ error: err.message });
  }
});

// 7. GET /api/render-progress
app.get('/api/render-progress', (req, res) => {
  res.json(currentRenderProgress);
});


// ---------------------------------------------------------------------------
// YOLOv8 football tracking + ElevenLabs commentary
// Provider secrets remain server-side; the Android APK never contains them.
// ---------------------------------------------------------------------------
function elevenKeys(): string[] {
  const raw = process.env.ELEVENLABS_API_KEYS || process.env.ELEVENLABS_API_KEY || '';
  return [...new Set(raw.split(/[\s,;]+/).map(s => s.trim()).filter(Boolean))];
}
let elevenIndex = 0;
const elevenCooldown = new Map<string, number>();

function elevenStatus() {
  const keys = elevenKeys(), now = Date.now();
  return {
    totalKeys: keys.length,
    activeKeys: keys.filter(k => (elevenCooldown.get(k) || 0) <= now).length,
    coolingKeys: keys.filter(k => (elevenCooldown.get(k) || 0) > now).length,
    configured: keys.length > 0,
    voiceId: process.env.ELEVENLABS_VOICE_ID || '',
  };
}

type VoiceDelivery = { stability: number; similarity_boost: number; style: number; use_speaker_boost: boolean; speed: number };

async function elevenTTS(text: string, voiceId?: string, delivery?: Partial<VoiceDelivery>): Promise<string> {
  const keys = elevenKeys();
  if (!keys.length) throw new Error('ElevenLabs is not configured. Set ELEVENLABS_API_KEYS.');
  const voice = voiceId || process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb';
  const modelId = process.env.ELEVENLABS_MODEL_ID || 'eleven_multilingual_v2';
  const voiceSettings: VoiceDelivery = {
    stability: Math.max(0, Math.min(1, Number(delivery?.stability ?? process.env.ELEVENLABS_STABILITY ?? 0.34))),
    similarity_boost: Math.max(0, Math.min(1, Number(delivery?.similarity_boost ?? process.env.ELEVENLABS_SIMILARITY ?? 0.82))),
    style: Math.max(0, Math.min(1, Number(delivery?.style ?? process.env.ELEVENLABS_STYLE ?? 0.32))),
    use_speaker_boost: delivery?.use_speaker_boost ?? true,
    speed: Math.max(0.7, Math.min(1.2, Number(delivery?.speed ?? 1))),
  };
  let lastErr: any;
  for (let attempt = 0; attempt < keys.length; attempt++) {
    const now = Date.now();
    const ordered = [...keys.slice(elevenIndex), ...keys.slice(0, elevenIndex)];
    const key = ordered.find(k => (elevenCooldown.get(k) || 0) <= now);
    if (!key) { await new Promise(r => setTimeout(r, 1000)); continue; }
    elevenIndex = (keys.indexOf(key) + 1) % keys.length;
    try {
      const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voice)}?output_format=mp3_44100_128`, {
        method: 'POST',
        headers: { 'xi-api-key': key, 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_id: modelId, voice_settings: voiceSettings }),
      });
      if (r.ok) {
        const audio = Buffer.from(await r.arrayBuffer());
        const filename = `commentary_${Date.now()}_${Math.random().toString(36).slice(2,8)}.mp3`;
        const out = path.join(videosDir, filename);
        fs.writeFileSync(out, audio);
        return out;
      }
      lastErr = new Error(`ElevenLabs ${r.status}: ${(await r.text()).slice(0, 300)}`);
      if (r.status === 401 || r.status === 403) elevenCooldown.set(key, Date.now() + 300000);
      else if (r.status === 429 || r.status >= 500) elevenCooldown.set(key, Date.now() + 15000);
    } catch (e) {
      lastErr = e; elevenCooldown.set(key, Date.now() + 5000);
    }
  }
  throw lastErr || new Error('No healthy ElevenLabs key available');
}

function runYoloTracking(inputPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const script = path.join(__dirname, 'yolo', 'track_football.py');
    const modelPath = process.env.YOLO_MODEL_PATH || path.join(__dirname, 'models', 'best.pt');
    const outputDir = path.join(videosDir, `yolo_${Date.now()}`);
    fs.mkdirSync(outputDir, { recursive: true });
    const child = spawn(process.env.PYTHON_BIN || 'python3',
      [script, '--source', inputPath, '--model', modelPath, '--output-dir', outputDir, '--json'],
      { env: { ...process.env }, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    child.stdout.on('data', d => { stdout += d.toString(); });
    child.stderr.on('data', d => { stderr += d.toString(); });
    child.on('error', reject);
    child.on('close', code => {
      if (code !== 0) return reject(new Error(`YOLO tracker exited ${code}: ${stderr.slice(-1500)}`));
      try {
        const result = JSON.parse(stdout.trim().split('\n').filter(Boolean).pop() || '{}');
        if (!result.success) return reject(new Error(result.error || 'YOLO tracking failed'));
        resolve(result);
      } catch { reject(new Error(`YOLO tracker returned invalid JSON: ${stdout.slice(-1000)}`)); }
    });
  });
}

async function generateCommentaryScript(plan: any, style: string): Promise<any[]> {
  const timeline = Array.isArray(plan?.timeline) ? plan.timeline : [];
  const fallback = timeline.filter((t: any) => t.narration).map((t: any, i: number) => ({
    timeline_index: Number(t.timeline_index ?? i),
    text: String(t.narration).trim(),
    emotion: ['hook','impact','climax'].includes(t.beat_role) ? 'intense' : 'focused',
    intensity: ['climax'].includes(t.beat_role) ? 0.95 : ['impact','reaction'].includes(t.beat_role) ? 0.78 : 0.55,
    pause_after_ms: 220,
  }));
  const prompt = `أنت مخرج تعليق رياضي عربي محترف، وليس روبوت قراءة نص.
اكتب تعليقًا صوتيًا طبيعيًا لمقطع كرة قدم سينمائي مدته 64 ثانية.
المطلوب أداء حي يشبه معلّقًا محترفًا: جمل قصيرة ومتوسطة، تنويع في الطول، توقفات مقصودة، انفعالات تتصاعد مع القصة، وعدم الكلام فوق كل ثانية.
استخدم العربية الفصحى الرياضية المرنة مع تعبيرات طبيعية مثل: "يا سلام!"، "انظر إلى هذه اللمسة"، "هنا تبدأ الحكاية" عندما تكون مناسبة، لكن لا تكررها.
لا تستخدم عبارات عامة آلية مثل "هذه لحظة رائعة" في كل لقطة.
لا تخترع هدفًا أو تمريرة أو اسم لاعب أو نتيجة أو بطولة. استخدم فقط ما يثبته action في الخط الزمني.
اترك بعض اللقطات بلا تعليق إذا كان الصمت يخدم التشويق.
في لحظة climax: ارفع الطاقة، قصّر الكلمات، ثم اترك وقفة قصيرة بعد الضربة.
أخرج JSON فقط بهذا الشكل:
[{"timeline_index":0,"text":"...","emotion":"calm|focused|excited|intense|shock|triumphant|emotional","intensity":0.0,"pause_after_ms":0}]
الحد الأقصى 14 سطرًا. كل سطر من 3 إلى 16 كلمة.
الأسلوب: ${style}
الخط الزمني الموثق: ${JSON.stringify(timeline.map((t:any)=>({timeline_index:t.timeline_index,source_start:t.source_start,source_end:t.source_end,output_start:t.output_start,output_end:t.output_end,action:t.action,beat_role:t.beat_role,importance:t.importance})))}.
دليل Football Director المتاح: ${JSON.stringify(plan?.footballDirectorEvidence || null)}
لا تنسب إلى اللقطة حدثًا أقوى من الدليل؛ إذا كان الدليل ضعيفًا فاجعل اللغة وصفية، واستعمل الصمت بدل ملء الفراغ.`;
  try {
    const response = await geminiRotator.run('commentary-director', client => client.models.generateContent({
      model: process.env.GEMINI_VIDEO_MODEL || 'gemini-3.8-flash',
      contents: prompt,
      config: { responseMimeType: 'application/json' },
    }));
    const parsed = JSON.parse((response.text || '[]').trim());
    if (Array.isArray(parsed) && parsed.length) return parsed.slice(0, 14);
  } catch (err) {
    console.warn('[COMMENTARY] Gemini director failed; using verified narration:', err);
  }
  return fallback.slice(0, 14);
}

async function probeAudioDuration(filePath: string): Promise<number> {
  const p = await execPromise(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${JSON.stringify(filePath)}`);
  return Math.max(0, Number.parseFloat(p.stdout.trim()) || 0);
}

async function buildTimedCommentary(plan: any, style: string): Promise<{ audioPath: string; script: any[] }> {
  const script = await generateCommentaryScript(plan, style);
  const timeline = Array.isArray(plan?.timeline) ? plan.timeline : [];
  const work = path.join('/tmp/football_engine/work', `commentary_${Date.now()}`);
  fs.mkdirSync(work, { recursive: true });
  const clips: Array<{ path: string; start: number; duration: number }> = [];
  try {
    for (const [i, line] of script.entries()) {
      const idx = Number(line.timeline_index);
      const target = timeline.find((t: any) => Number(t.timeline_index) === idx);
      if (!target || !String(line.text || '').trim()) continue;
      const start = Math.max(0, Number(target.output_start) || 0);
      const slot = Math.max(0.45, (Number(target.output_end) || start + 1) - start);
      const intensity = Math.max(0, Math.min(1, Number(line.intensity) || 0.55));
      const emotion = String(line.emotion || 'focused');
      const speed = Math.max(0.88, Math.min(1.12, 0.94 + intensity * 0.13));
      const stability = emotion === 'calm' ? 0.46 : emotion === 'intense' || emotion === 'shock' ? 0.24 : 0.34;
      const styleExaggeration = emotion === 'intense' || emotion === 'shock' ? 0.48 : 0.30;
      const audio = await elevenTTS(String(line.text).replace(/\s+/g, ' ').trim(), undefined, {
        stability, style: styleExaggeration, speed, similarity_boost: 0.82, use_speaker_boost: true,
      });
      const rawDuration = await probeAudioDuration(audio);
      const fit = rawDuration > slot * 0.96 ? Math.max(0.82, Math.min(1.2, rawDuration / (slot * 0.92))) : 1;
      const fitted = path.join(work, `line_${String(i).padStart(2,'0')}.mp3`);
      const atempo = fit > 1.001 ? `atempo=${Math.min(2, fit).toFixed(4)}` : fit < 0.999 ? `atempo=${Math.max(0.5, fit).toFixed(4)}` : 'anull';
      await execPromise(`ffmpeg -y -i ${JSON.stringify(audio)} -af ${JSON.stringify(atempo)} -c:a aac -b:a 160k ${JSON.stringify(fitted)}`);
      const fittedDuration = await probeAudioDuration(fitted);
      clips.push({ path: fitted, start, duration: fittedDuration });
    }
    if (!clips.length) throw new Error('No commentary lines were generated.');
    const inputs = clips.map(c => `-i ${JSON.stringify(c.path)}`).join(' ');
    const filters = clips.map((c, i) => `[${i}:a]adelay=${Math.round(c.start*1000)}|${Math.round(c.start*1000)},volume=1.0[a${i}]`).join(';');
    const mix = clips.map((_, i) => `[a${i}]`).join('');
    const out = path.join(videosDir, `commentary_track_${Date.now()}.m4a`);
    await execPromise(`ffmpeg -y ${inputs} -filter_complex ${JSON.stringify(`${filters};${mix}amix=inputs=${clips.length}:duration=longest:dropout_transition=0,alimiter=limit=0.94`)} -t 64 -c:a aac -b:a 192k ${JSON.stringify(out)}`);
    return { audioPath: out, script };
  } finally {
    try { fs.rmSync(work, { recursive: true, force: true }); } catch {}
  }
}

async function muxCommentary(videoPath: string, audioPath: string): Promise<string> {
  const out = path.join(videosDir, `final_commentary_${Date.now()}.mp4`);
  // Sidechain ducking makes the commentator sit in front of the stadium/music bed
  // instead of fighting it; this is the main difference from the old robotic mux.
  const filter = '[0:a]volume=0.72[base];[1:a]volume=1.0[vo];[vo]asplit=2[vo_sc][vo_mix];[base][vo_sc]sidechaincompress=threshold=0.055:ratio=6:attack=12:release=260:makeup=1[ducked];[ducked][vo_mix]amix=inputs=2:duration=first:dropout_transition=0.18,alimiter=limit=0.94[a]';
  await execPromise(`ffmpeg -y -i ${JSON.stringify(videoPath)} -i ${JSON.stringify(audioPath)} -filter_complex ${JSON.stringify(filter)} -map 0:v:0 -map "[a]" -c:v copy -c:a aac -b:a 192k -shortest ${JSON.stringify(out)}`);
  return out;
}


app.get('/api/yolo/status', (req, res) => {
  res.json({ success: true, enabled: process.env.YOLO_ENABLED !== 'false', model: process.env.YOLO_MODEL_PATH || 'models/best.pt', tracker: 'ByteTrack' });
});


async function runFootballEvidence(inputPath: string, trackingResult: any): Promise<any> {
  try {
    const trackingPath = trackingResult?.jsonUrl
      ? path.join(videosDir, String(trackingResult.jsonUrl).replace(/^\/videos\//,''))
      : null;
    if (!trackingPath || !fs.existsSync(trackingPath)) return null;
    const work = path.join(path.dirname(trackingPath), `director_${Date.now()}`);
    fs.mkdirSync(work,{recursive:true});
    const eventsPath=path.join(work,'events.json');
    const directorPath=path.join(work,'director.json');
    const py=process.env.PYTHON_BIN||'python3';
    await execPromise(`${JSON.stringify(py)} ${JSON.stringify(path.join(__dirname,'yolo/event_engine.py'))} --tracking ${JSON.stringify(trackingPath)} --output ${JSON.stringify(eventsPath)}`);
    const summary=trackingResult.summary||{};
    const duration=(Number(summary.frames)||0)/Math.max(.1,Number(summary.fps)||25);
    await execPromise(`${JSON.stringify(py)} ${JSON.stringify(path.join(__dirname,'yolo/football_director.py'))} --events ${JSON.stringify(eventsPath)} --output ${JSON.stringify(directorPath)} --duration ${duration.toFixed(3)}`);
    const director = JSON.parse(fs.readFileSync(directorPath,'utf8'));
    director.eventsPath = eventsPath;
    director.directorPath = directorPath;
    return director;
  } catch (e:any) {
    console.warn('[DIRECTOR] evidence pass skipped:',e?.message||e);
    return null;
  }
}


async function runMadnessEngine(eventsPath: string, timeline: any[]) {
  try {
    const py = process.env.PYTHON_BIN || 'python3';
    const work = path.dirname(eventsPath);
    const timelineInput = path.join(work, 'madness_timeline_input.json');
    const madnessPath = path.join(work, 'madness.json');
    fs.writeFileSync(timelineInput, JSON.stringify({ timeline }, null, 2), 'utf8');
    const script = path.join(__dirname, 'yolo/madness_engine.py');
    if (!fs.existsSync(script) || !fs.existsSync(eventsPath)) return null;
    await execPromise(`${JSON.stringify(py)} ${JSON.stringify(script)} --events ${JSON.stringify(eventsPath)} --timeline ${JSON.stringify(timelineInput)} --output ${JSON.stringify(madnessPath)}`);
    if (!fs.existsSync(madnessPath)) return null;
    return JSON.parse(fs.readFileSync(madnessPath, 'utf8'));
  } catch (e:any) {
    console.warn('[MADNESS] evidence pass skipped:', e?.message || e);
    return null;
  }
}

// 8. POST /api/analyze-video
// Sends the ACTUAL uploaded video to Gemini Files API. No metadata-only analysis.
app.post('/api/analyze-video', async (req, res) => {
  try {
    const { videoMetadata, style = 'CINEMATIC SPORTS', generationTier = 'ORIGINAL FOOTAGE ONLY', referenceStyle = null, trackingEnabled = true } = req.body;
    const localPath = videoMetadata?.localPath;
    const duration = Number(videoMetadata?.duration);
    if (!localPath || !fs.existsSync(localPath)) {
      return res.status(400).json({ success: false, error: 'Uploaded video file is not available on the backend.' });
    }
    if (!Number.isFinite(duration) || duration <= 0) {
      return res.status(400).json({ success: false, error: 'Invalid uploaded video duration.' });
    }

    const referenceReelRules = `REFERENCE CINEMATIC REEL STYLE: 9:16 emotional football social edit. Build a dense montage of roughly 12-22 real shots. Open immediately with a visually strong moment, then alternate wide/action/detail/close-up/reaction shots. Typical output shots are about 1-5 seconds, with shorter impact inserts around major actions. Use tight crops and controlled punch-ins, not constant zoom. Use 0.55-0.85x slow motion only when the source motion benefits from it; use 1.15-1.45x for low-information travel/setup. Hard cuts should dominate. Use a single restrained white flash only for a major verified impact. Captions should appear on most shots as short 2-6 word ALL-CAPS editorial lines, white, centered around the lower third, never a boxed subtitle. Build a story arc: HOOK -> SETUP -> ESCALATION -> IMPACT -> REACTION -> CLIMAX -> OUTRO. The final shot should be a real verified reaction/celebration/detail or strongest available closing frame. Never invent a goal, player identity, emotion, score, or event. Copy only the editing language, not logos, watermarks, exact frames, or copyrighted footage.
Style requested: ${style}. Generation tier: ${generationTier}. Optional style profile: ${JSON.stringify(referenceStyle || null)}.`;

    // PSYCHOLOGICAL DRAMA rules: an "anime / mind-game" football thriller built
    // ONLY from the uploaded footage. The editing language is defined here; the
    // runtime still rejects invented events/identities.
    const psychologicalDramaRules = `PSYCHOLOGICAL DRAMA STYLE (anime-mind-game football thriller): Build a tense 9:16 vertical edit of roughly 14-20 real shots from the SAME uploaded footage. This style is about the DUEL and the inner monologue, not a highlight reel. Editorial rules:
- Alternate extreme close-ups on eyes, feet, boots touching the ball, shirts/numbers, sweat details and faces with mid shots of the duel between the two nearest players. Mark these eyes/face detail shots with shot_type "eye_close_up" or "extreme_close_up".
- Heavy slow motion on the buildup and the decisive move: 0.4-0.65x on the feint / body-weight shift / ball touch. A sudden speed ramp back to 1.0-1.2x at the impact (shot / pass / tackle). Use 1.15-1.4x only for low-information travel.
- High contrast, deep shadows, high saturation, slight vignette. Set color_grade contrast 1.3-1.45, saturation 1.1-1.3, highlights negative, grain 0.03-0.1.
- Text is a FIRST-PERSON INNER MONOLOGUE ("COME CLOSER", "ONE MORE STEP AND YOU ARE MINE", "NOW"). Keep it 2-7 words, ALL CAPS, and put it on most shots centered in the LOWER THIRD. Also fill the "narration" field on each shot with the same short spoken line (the voice-over).
- Story arc: the approach -> the trap/lure -> eye-contact standoff -> the feint -> the fall -> the impact -> the reaction/celebration -> a closing line. The single strongest verified moment is the climax.
- Copy only the editing language. Never invent a goal, player name, score, kit, logo, watermark, or event that is not visible. If the footage does not clearly contain a duel, build the tense montage from the closest available real action and close-up details.`;
    // Reset the live progress for this new run so a previous "Idle"/error state
    // never sticks while the (slow) analysis is genuinely in progress.
    currentRenderProgress = { percent: 2, stage: 'Starting the master render pipeline...' };

    let trackingResult: any = null;
    if (trackingEnabled && process.env.YOLO_ENABLED !== 'false') {
      currentRenderProgress = { percent: 4, stage: 'YOLOv8 tracking real players and the ball...' };
      try { trackingResult = await runYoloTracking(localPath); }
      catch (trackErr: any) { console.warn('[YOLO] Tracking skipped:', trackErr.message); }
    }
    if (trackingResult) {
      currentRenderProgress = { percent: 7, stage: 'Building the football evidence timeline...' };
    }
    const footballEvidence = trackingResult ? await runFootballEvidence(localPath, trackingResult) : null;
    const compactTracking = trackingResult ? {
      fps: trackingResult.summary?.fps,
      detections: trackingResult.summary?.detections,
      tracks: Object.keys(trackingResult.summary?.tracks || {}).length,
    } : null;
    const trackingContext = trackingResult ? `\nYOLOv8 TRACKING EVIDENCE: ${JSON.stringify(compactTracking)}\nFOOTBALL DIRECTOR EVIDENCE: ${JSON.stringify(footballEvidence || {evidence_level:'unavailable'})}\n` : '';
    const activeStyleRules = style === 'PSYCHOLOGICAL DRAMA' ? psychologicalDramaRules : referenceReelRules;
    const systemInstruction = `You are an elite football short-form editor and cinematographer.${trackingContext} You MUST watch the attached source video itself. Every source timestamp and action description MUST be grounded in visible frames from that exact video. Never invent a goal, dribble, celebration, player identity, score, camera movement, or timestamp. Return strict JSON only.
${activeStyleRules}`;

    const prompt = `${trackingContext}\nWATCH THE ATTACHED FOOTBALL VIDEO BEFORE WRITING ANY TIMESTAMPS. Analyze the full video, identify real salient moments, and then design a premium 64-second vertical montage in the requested style. Prefer 12-22 distinct real moments when the source contains enough material. Avoid long generic gameplay unless it is necessary for story continuity.
Required JSON shape:
{
  "duration":64,
  "aspect_ratio":"9:16",
  "subject":{"name":string,"confidence":number},
  "timeline":[{"source_start":number,"source_end":number,"output_start":number,"output_end":number,"action":string,"importance":number,"speed":number,"zoom_start":number,"zoom_end":number,"crop_x":number,"crop_y":number,"transition":"hard_cut"|"directional_blur"|"flash","text":string,"narration":string,"shot_type":"wide"|"medium"|"close_up"|"extreme_close_up"|"eye_close_up"|"action"|"reaction"|"crowd"|"detail","beat_role":"hook"|"setup"|"escalation"|"impact"|"reaction"|"climax"|"outro"}],
  "music":{"style":string,"bpm":number,"energy_curve":number[]},
  "color_grade":{"contrast":number,"saturation":number,"highlights":number,"shadows":number,"grain":number}
}
For each text field, write a short editorial caption that does not assert an unverified fact. The "narration" field is the spoken inner-monologue line for that shot (may be empty for non-narrative styles). Source timestamps are the truth; do not use metadata as evidence. Make the first 3 seconds highly arresting and reserve the strongest verified moment for the climax.`;

    const model = process.env.GEMINI_VIDEO_MODEL || 'gemini-3.8-flash';
    const parsed = await generateGeminiJsonWithVideo(
      localPath,
      prompt,
      systemInstruction,
      model,
      videoMetadata?.mimeType || 'video/mp4',
      (p) => { currentRenderProgress = p; }
    );
    currentRenderProgress = { percent: 22, stage: 'Validating the 64-second edit plan...' };
    parsed.generationTier = generationTier;
    parsed.styleName = style;
    const validatedPlan = validateAndEnforce64sEditPlan(parsed, duration, style);
    if (footballEvidence) (validatedPlan as any).footballDirectorEvidence = footballEvidence;

    // MADNESS-5 runs only after the real-video evidence pass and validated timeline.
    // It escalates verified moments; it never creates football facts.
    let madnessResult: any = null;
    if (footballEvidence?.eventsPath && Array.isArray(validatedPlan.timeline)) {
      madnessResult = await runMadnessEngine(footballEvidence.eventsPath, validatedPlan.timeline);
      if (madnessResult?.timeline) {
        validatedPlan.timeline = madnessResult.timeline;
        (validatedPlan as any).madness = {
          version: madnessResult.version,
          policy: madnessResult.policy,
          counts: madnessResult.counts,
        };
      }
    }
    res.json({
      success: true,
      editPlan: validatedPlan,
      fallbackUsed: false,
      videoAnalyzed: true,
      model,
      madness: madnessResult ? { version: madnessResult.version, counts: madnessResult.counts } : { enabled: false },
      tracking: trackingResult ? { success: true, videoUrl: trackingResult.videoUrl, jsonUrl: trackingResult.jsonUrl, summary: trackingResult.summary } : { success: false }
    });
  } catch (err: any) {
    console.error('Video analysis failed:', err);
    const status = isRetryableGeminiError(err) ? 503 : 500;
    res.status(status).json({
      success: false,
      videoAnalyzed: false,
      error: err?.message || 'Gemini video analysis failed.',
      code: isRetryableGeminiError(err) ? 'GEMINI_TEMPORARILY_UNAVAILABLE' : 'VIDEO_ANALYSIS_FAILED',
    });
  }
});


// 8.5. POST /api/commentary/generate
app.post('/api/commentary/generate', async (req, res) => {
  try {
    const { editPlan, style = 'CINEMATIC SPORTS', videoLocalPath } = req.body;
    const generated = await buildTimedCommentary(editPlan, style);
    const script = generated.script;
    const audioPath = generated.audioPath;
    const finalVideoPath = videoLocalPath && fs.existsSync(videoLocalPath) ? await muxCommentary(videoLocalPath, audioPath) : undefined;
    res.json({
      success: true, provider: 'ElevenLabs', script, commentaryAudioUrl: `/videos/${path.basename(audioPath)}`,
      audioUrl: `/videos/${path.basename(audioPath)}`,
      finalVideoUrl: finalVideoPath ? `/videos/${path.basename(finalVideoPath)}` : undefined,
      voiceId: process.env.ELEVENLABS_VOICE_ID || '',
    });
  } catch (err: any) {
    console.error('[ELEVENLABS] Commentary failed:', err);
    res.status(503).json({ success: false, provider: 'ElevenLabs', error: err.message });
  }
});
app.get('/api/elevenlabs/status', (req, res) => res.json({ success: true, provider: 'ElevenLabs', ...elevenStatus() }));

// 9. POST /api/qc-review
// QC now inspects the actual rendered MP4 instead of reviewing only JSON metadata.
app.post('/api/qc-review', async (req, res) => {
  try {
    const { outputLocalPath, editPlan, style = 'CINEMATIC SPORTS' } = req.body;
    if (!outputLocalPath || !fs.existsSync(outputLocalPath)) {
      return res.status(400).json({ success: false, error: 'Rendered video localPath is required for visual QC.' });
    }

    const systemInstruction = `You are a professional football short-form video QC editor. WATCH THE ATTACHED RENDERED VIDEO. Do not infer visual quality from metadata alone. Review the actual frames and audio/video pacing. Return strict JSON only with factual, actionable observations. Do not invent events that are not visible.`;
    const prompt = `Inspect this rendered 64-second football short. Style: ${style}.
The expected edit plan is: ${JSON.stringify(editPlan?.timeline?.map((t: any) => ({
      idx: t.timeline_index, source: [t.source_start, t.source_end], output: [t.output_start, t.output_end], action: t.action, speed: t.speed, text: t.text
    })) || [])}
Check: opening impact, real-footage continuity, crop/framing, excessive zoom, motion quality, pacing, transitions, caption readability, audio continuity, climax, ending, and whether the final output is exactly 64 seconds.
Return:
{"qc_verdict":"APPROVED_WITH_TWEAKS"|"REVISE_PACING"|"EXCELLENT","overall_critique":string,"pacing_score":number,"cinematic_score":number,"corrections":[{"timeline_index":number,"change":"adjust_speed"|"adjust_crop"|"replace_text"|"refine_transition"|"trim_duration","reason":string,"recommended_speed":number,"recommended_crop_x":number,"recommended_crop_y":number,"recommended_text":string,"recommended_transition":string}]}`;

    const model = process.env.GEMINI_VIDEO_MODEL || 'gemini-3.8-flash';
    const parsedQC = await generateGeminiJsonWithVideo(outputLocalPath, prompt, systemInstruction, model);
    res.json({ success: true, review: parsedQC, videoAnalyzed: true });
  } catch (err: any) {
    console.error('Error during visual QC:', err);
    const status = isRetryableGeminiError(err) ? 503 : 500;
    res.status(status).json({ success: false, videoAnalyzed: false, error: err?.message || 'Visual QC failed.' });
  }
});

// 10. POST /api/analyze-reference
app.post('/api/analyze-reference', async (req, res) => {
  try {
    const { referenceDescription, referenceTitle } = req.body;

    const systemInstruction = `You are a senior football short-form editor. Extract only editorial characteristics: pacing, shot length, framing, caption treatment, zoom behavior, transition density, color mood and energy curve. Never copy logos, watermarks, exact frames or footage. Return STRICT JSON.`;

    const prompt = `Create a style profile for the user's requested reference reel.
Title: ${referenceTitle || 'Reference Cinematic Football Reel'}
Notes: ${referenceDescription || 'Vertical 9:16 football montage with intimate player close-ups, match-action details, hard cuts, controlled punch-ins, selective slow motion, small white captions near the lower third, dramatic dark-green stadium grade, and a strong emotional climax.'}
Return: {"average_shot_duration":1.8,"zoom_intensity":0.78,"transition_frequency":0.16,"slow_motion_frequency":0.30,"text_frequency":0.88,"color_style":"dark cinematic stadium green with controlled contrast and warm skin highlights","energy_curve":"strong-hook / tension / escalation / climax / emotional outro","recommended_bpm":126,"cinematography_notes":"Tight 9:16 crops, close-ups, football details, restrained flash impacts, small white editorial captions, hard-cut rhythm."}`;

    const response = await geminiRotator.run('analyze-reference', (client) =>
      client.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
        },
      })
    );

    const profile = JSON.parse(response.text || '{}');
    res.json({
      success: true,
      styleProfile: profile,
    });
  } catch (err: any) {
    console.error('Error analyzing reference video:', err);
    res.json({
      success: true,
      styleProfile: {
        average_shot_duration: 1.45,
        zoom_intensity: 0.75,
        transition_frequency: 0.65,
        slow_motion_frequency: 0.45,
        text_frequency: 0.35,
        color_style: 'Dark anamorphic cinematic with gold highlights',
        energy_curve: 'slow-build-explosive-climax',
        recommended_bpm: 130,
        cinematography_notes: 'Tight 9:16 vertical tracking with aggressive punch-ins on ball impact.',
      },
    });
  }
});

// CRITICAL FIX: Any unhandled /api/* route must return STRICT JSON 404, NEVER fall through to Vite index.html
app.all('/api/*', (req, res) => {
  res.status(404).json({
    success: false,
    error: {
      code: 'ENDPOINT_NOT_FOUND',
      message: `API endpoint ${req.method} ${req.originalUrl} not found on this server.`,
    },
  });
});

// Global Express error handler returning STRICT JSON (prevents default HTML error templates)
app.use((err: any, req: any, res: any, next: any) => {
  console.error('[SERVER GLOBAL ERROR]', err);
  if (res.headersSent) {
    return next(err);
  }
  res.status(err.status || 500).json({
    success: false,
    error: {
      code: err.code || 'INTERNAL_ERROR',
      message: err.message || 'Internal server error occurred.',
    },
  });
});

// Verify that the FFmpeg/FFprobe binaries are reachable at boot.
// On Render's native Node runtime these are missing — the Docker image installs
// them. Logging this loudly turns a vague "spawn ffmpeg ENOENT" runtime error
// (500 on every render route) into an obvious, actionable startup message.
async function verifyFFmpegBinaries(): Promise<boolean> {
  try {
    const { stdout } = await execPromise('ffmpeg -version');
    const version = stdout.split('\n')[0];
    console.log(`[BOOT] FFmpeg detected -> ${version}`);
    await execPromise('ffprobe -version');
    console.log('[BOOT] FFprobe detected. Video pipeline is ready.');
    return true;
  } catch (err: any) {
    console.error('[BOOT] FFmpeg/FFprobe NOT found. Video rendering will fail.');
    console.error('[BOOT] Deploy with the provided Dockerfile so ffmpeg is installed.');
    console.error(`[BOOT] Details: ${err?.message || err}`);
    return false;
  }
}

// Serve frontend in development via Vite middleware or production build
async function startServer() {
  await verifyFFmpegBinaries();

  // Storage diagnostics: makes the persistence mode obvious in Render logs.
  console.log(`[BOOT] Storage driver: ${storage.driver} | media dir: ${storage.mediaDir}`);
  if (storage.driver === 'local' && storage.mediaDir === path.resolve('public/videos')) {
    console.warn('[BOOT] Rendering to the default ephemeral folder. Attach a Render Disk (set PUBLIC_DIR) or use STORAGE_DRIVER=s3 for persistence.');
  }

  // Gemini key pool diagnostics.
  const keyStatus = geminiRotator.status();
  console.log(`[BOOT] Gemini key pool: ${keyStatus.totalKeys} key(s) | ${keyStatus.activeKeys} active | ${keyStatus.coolingKeys} cooling | ${keyStatus.invalidKeys} invalid`);
  if (keyStatus.totalKeys === 0) {
    console.error('[BOOT] NO GEMINI API KEYS configured. Set GEMINI_API_KEYS (comma separated) in the environment.');
  }

  // Periodic cleanup of old generated media (only when MEDIA_RETENTION_HOURS > 0)
  if (storage.retentionHours > 0) {
    storage.cleanupOldFiles();
    const intervalMs = Math.min(storage.retentionHours * 3600 * 1000, 60 * 60 * 1000);
    setInterval(() => storage.cleanupOldFiles(), intervalMs).unref?.();
  }

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    if (fs.existsSync(distPath)) {
      // Hashed assets (index-<hash>.js/css) are immutable: cache them hard so the
      // browser never re-downloads the same build.
      app.use(
        express.static(distPath, {
          // Let the catch-all below serve index.html (with no-store headers)
          // for "/" instead of express.static's default cached copy.
          index: false,
          setHeaders: (res, filePath) => {
            if (/\/assets\/.*-[A-Za-z0-9_-]{8,}\.(js|css)$/.test(filePath)) {
              res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
            }
          },
        })
      );
      // CRITICAL: the SPA entry (index.html) MUST NEVER be cached. Otherwise a
      // device/WebView keeps loading the OLD hashed bundle (which pointed at a
      // dead backend URL) even after a redeploy, producing phantom 404s such as
      // "The server did not find the required API route (/api/version)".
      app.get('*', (req, res) => {
        res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
        res.setHeader('Pragma', 'no-cache');
        res.setHeader('Expires', '0');
        res.sendFile(path.join(distPath, 'index.html'));
      });
    }
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[FOOTBALL CINEMATIC AI] Server running on port ${PORT} with real FFmpeg video processing`);
  });
}

startServer();
