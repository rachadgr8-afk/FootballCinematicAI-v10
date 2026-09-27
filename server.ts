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
import os from 'os';
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
      i