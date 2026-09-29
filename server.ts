import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import multer from 'multer';
import { fileURLToPath } from 'url';
import cors from 'cors';
import { exec, spawn } from 'child_process';
import util from 'util';
import { ffmpegEngine, FFmpegProgress } from './server/ffmpegEngine';
import { storage } from './server/storage';
import { geminiRotator } from './server/geminiRotator';
import { cinematicEngine } from './server/cinematicEngine';
import { samService } from './server/samService';
import { referenceStyleService, CinematicMode } from './server/referenceStyleService';
import { storytellerService } from './server/storytellerService';
import {
  openRouterText,
  openRouterVision,
  deepseekText,
  providerHealth,
  providerConfigured,
  sanitize,
  OPENROUTER_TEXT_MODEL,
  OPENROUTER_VISION_MODEL,
  DEEPSEEK_MODEL,
} from './server/aiProviders';

const execPromise = util.promisify(exec);

dotenv.config();

// The rotator module is evaluated (and its constructor runs) during the hoisted
// imports, i.e. BEFORE dotenv loads .env. Reload now that the environment is set.
geminiRotator.reload();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const BUILD_VERSION = '2026-09-27-exceptional-v10.2.0-openrouter-deepseek';

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
    pipeline: 'football-director-v10 + LOCAL-motion-analysis + Optional-OpenRouter(VLM/text) + Optional-DeepSeek + evidence-engine + YOLO-ByteTrack + ReID + event-engine + optional-SAM-segmentation + beat-sync + scene-aware-RIFE + ffmpeg',
    videoAnalysis: 'local-opencv-motion',
    externalVideoApi: false,
    // OPTIONAL providers: capability flags only — never the keys themselves.
    optionalAiProviders: {
      openrouter: providerConfigured('openrouter'),
      deepseek: providerConfigured('deepseek'),
    },
    // OPTIONAL SAM layer: reported as a capability flag only — the heavy engine
    // lives in its own isolated interpreter and is OFF unless SAM_ENABLED=true.
    sam: { enabled: samService.enabled, adapter: process.env.SAM_ADAPTER || 'auto', isolated: Boolean(process.env.SAM_PYTHON_BIN) },
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

// ---------------------------------------------------------------------------
// LOCAL video analysis (NO external API / NO API keys).
//
// The pipeline previously asked Gemini ("analyze-video") to watch the uploaded
// footage. When every Gemini key hit a 503 "high demand" spike, the whole
// render died with a red Pipeline error. This helper replaces that dependency
// with a pure local OpenCV pass (yolo/local_motion_analysis.py): low-motion
// samples become "slow moments" (dribble / build-up), high-motion samples
// become action/impact moments. Nothing is invented — the analysis only reports
// where the real motion in the uploaded file actually is.
// ---------------------------------------------------------------------------
interface LocalMotionAnalysis {
  success: boolean;
  model: string;
  duration: number;
  fps: number;
  motion_profile: Array<{ t: number; energy: number }>;
  slow_moments: number[];
  action_moments: Array<{ t: number; energy: number }>;
  peaks: Array<{ t: number; energy: number }>;
  thresholds: { low: number; high: number };
  mean_energy: number;
  max_energy: number;
  fallback_reason?: string;
}

async function runLocalMotionAnalysis(videoPath: string): Promise<LocalMotionAnalysis> {
  const pythonBin = process.env.PYTHON_BIN || 'python3';
  const script = path.join(__dirname, 'yolo', 'local_motion_analysis.py');
  const sampleFps = Number(process.env.LOCAL_MOTION_SAMPLE_FPS || 4);
  const maxSeconds = Number(process.env.LOCAL_MOTION_MAX_SECONDS || 300);
  if (!fs.existsSync(script)) {
    throw new Error(`Local motion analysis script is missing: ${script}`);
  }
  const cmd = `${JSON.stringify(pythonBin)} ${JSON.stringify(script)} --video ${JSON.stringify(videoPath)} --sample-fps ${sampleFps} --max-seconds ${maxSeconds}`;
  let stdout = '';
  try {
    const res = await execPromise(cmd, { timeout: Number(process.env.LOCAL_MOTION_TIMEOUT_MS || 240000), maxBuffer: 32 * 1024 * 1024 });
    stdout = res.stdout;
  } catch (err: any) {
    // The script prints valid JSON even on internal failure; a non-zero exit with
    // usable stdout is still parsed below. Anything else propagates.
    if (err?.stdout) stdout = err.stdout;
    else throw new Error(`Local motion analysis failed: ${err?.message || err}`);
  }
  const parsed = JSON.parse(String(stdout || '').trim());
  if (!parsed || !Array.isArray(parsed.motion_profile)) {
    throw new Error('Local motion analysis returned an invalid profile.');
  }
  return parsed as LocalMotionAnalysis;
}

// ---------------------------------------------------------------------------
// OPTIONAL AI PROVIDER INTEGRATION (OpenRouter VLM + DeepSeek text)
//
// These helpers are the REAL wiring into the render path the project actually
// uses:
//   * `enhancePlanWithOptionalProviders` is called by POST /api/analyze-video and
//     sends REAL JPEG frames extracted from the uploaded footage (as base64 data
//     URLs) to the OpenRouter VLM, so the model sees ACTUAL pixels — never just a
//     text prompt.
//   * `generateCommentaryScript` lets OpenRouter / DeepSeek write the Arabic
//     commentary from the VERIFIED timeline, with the local narration as fallback.
//   * `applyProviderEnvironment` maps OPENROUTER_API_KEY onto the pre-existing
//     generic VLM layer (yolo/vlm_analyzer.py) so the psychological "Predator vs
//     Prey" layer also uses REAL sampled frames.
//
// Every call is OPTIONAL and FAIL-SAFE: a missing key or a failed request returns
// `null` / the local script and the render continues unchanged. Each call is
// recorded (secret-free) for GET /api/ai/status so success is PROVABLE.
// ---------------------------------------------------------------------------

/** Frame scale for the VLM. Small = fast + cheap, still legible. */
const VLM_FRAME_WIDTH = Number(process.env.VLM_FRAME_WIDTH || 480);

/**
 * Map the OPTIONAL provider keys onto the EXISTING optional layers WITHOUT adding
 * new secrets or new providers. Only DEFAULTS are filled in: an explicitly set
 * VLM_* variable always wins, so existing deployments are unaffected.
 *
 *   OPENROUTER_API_KEY -> the OpenAI-compatible VLM layer consumed by
 *        yolo/vlm_analyzer.py (VLM_PROVIDER=openai, VLM_BASE_URL=OpenRouter).
 *   DEEPSEEK_API_KEY   -> read directly by yolo/deepseek_storyteller.py.
 *
 * No key is ever logged.
 */
function applyProviderEnvironment(): void {
  const orKey = (process.env.OPENROUTER_API_KEY || '').trim();
  if (orKey && !process.env.VLM_API_KEY) {
    process.env.VLM_API_KEY = orKey;
    if (!process.env.VLM_PROVIDER) process.env.VLM_PROVIDER = 'openai';
    if (!process.env.VLM_BASE_URL) {
      process.env.VLM_BASE_URL = (process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1').replace(/\/+$/, '');
    }
    if (!process.env.VLM_MODEL) {
      process.env.VLM_MODEL = process.env.OPENROUTER_VISION_MODEL || OPENROUTER_VISION_MODEL;
    }
  }
}

/**
 * Extract real JPEG frames from the uploaded footage at the given source
 * timestamps and return them as `data:image/jpeg;base64,...` URLs so the ACTUAL
 * pixels (not a metadata description) reach the vision model.
 */
async function extractFrameDataUrls(videoPath: string, timestamps: number[], maxFrames = 8): Promise<string[]> {
  const urls: string[] = [];
  const tmpDir = path.join('/tmp/football_engine/work', `vlm_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`);
  fs.mkdirSync(tmpDir, { recursive: true });
  try {
    const unique = Array.from(new Set(timestamps.map((t) => Math.max(0, Number(t) || 0)))).slice(0, maxFrames);
    for (let i = 0; i < unique.length; i++) {
      const out = path.join(tmpDir, `f_${i}.jpg`);
      try {
        await execPromise(`ffmpeg -y -ss ${unique[i].toFixed(3)} -i ${JSON.stringify(videoPath)} -frames:v 1 -vf "scale=${VLM_FRAME_WIDTH}:-2" -q:v 4 ${JSON.stringify(out)}`);
        if (fs.existsSync(out) && fs.statSync(out).size > 500) {
          urls.push(`data:image/jpeg;base64,${fs.readFileSync(out).toString('base64')}`);
        }
      } catch { /* skip this frame, keep the rest */ }
    }
  } finally {
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  }
  return urls;
}

interface AiPlanTelemetry {
  openrouterVision: string;
  openrouterText: string;
  deepseek: string;
  usedVision: boolean;
  usedDeepseek: boolean;
}

function telemetryLine(r: { status: string; model: string; httpStatus: number; latencyMs: number; error?: string }): string {
  return `${r.status} (model=${r.model}, http=${r.httpStatus || 'n/a'}, ${r.latencyMs}ms${r.error ? `, ${r.error}` : ''})`;
}

/** Validate + normalise an AI-provider plan into the editor's timeline shape. */
function parseAiPlan(content: any): any | null {
  try {
    const parsed = JSON.parse(String(content || '').replace(/^```(?:json)?/i, '').replace(/```$/, '').trim());
    if (Array.isArray(parsed?.timeline) && parsed.timeline.length >= 4) return parsed;
  } catch { /* not usable */ }
  return null;
}

/**
 * Ask the optional providers for an ENHANCED edit plan (real vision first).
 *   1. OpenRouter VLM looks at REAL extracted frames (primary — true vision).
 *   2. If the VLM is unavailable/failed, DeepSeek refines from the local motion.
 *   3. Any failure returns null and the caller keeps the pure-local plan.
 */
async function enhancePlanWithOptionalProviders(
  videoPath: string,
  videoDuration: number,
  localPlan: any,
  motion: LocalMotionAnalysis,
  styleName: string
): Promise<{ plan: any | null; telemetry: AiPlanTelemetry }> {
  const telemetry: AiPlanTelemetry = {
    openrouterVision: 'not_attempted',
    openrouterText: 'not_attempted',
    deepseek: 'not_attempted',
    usedVision: false,
    usedDeepseek: false,
  };

  const motionDigest = {
    duration: Number(motion.duration || videoDuration),
    mean_energy: motion.mean_energy,
    thresholds: motion.thresholds,
    action_moments: (motion.action_moments || []).slice(0, 12),
    slow_moments: (motion.slow_moments || []).slice(0, 12),
    peaks: (motion.peaks || []).slice(0, 12),
  };
  const requiredShape = `{"timeline":[{"source_start":number,"source_end":number,"action":string,"importance":1-10,"speed":0.5-2,"zoom_start":number,"zoom_end":number,"crop_x":0-1,"crop_y":0-1,"transition":"hard_cut"|"directional_blur"|"flash","text":string,"shot_type":"wide"|"medium"|"close_up"|"extreme_close_up"|"action"|"reaction"|"crowd"|"detail","beat_role":"hook"|"setup"|"escalation"|"impact"|"reaction"|"climax"|"outro"}]}`;
  const system = `You are an elite football short-form editor. You MUST ground every source_start/source_end strictly inside the real footage (0..${Number(motion.duration || videoDuration).toFixed(2)}s). Never invent a goal, score, player name or event that is not clearly visible. Return STRICT JSON only.`;

  // ---- 1) OpenRouter VLM on REAL frames ----
  const candidateTimes = [
    ...(localPlan?.timeline || []).map((c: any) => Number(c.source_start)),
    ...(motion.peaks || []).map((p: any) => Number(p.t)),
  ].filter((t: number) => Number.isFinite(t));
  let frames: string[] = [];
  try {
    frames = await extractFrameDataUrls(videoPath, candidateTimes, Number(process.env.VLM_MAX_FRAMES || 8));
  } catch { frames = []; }

  if (providerConfigured('openrouter') && frames.length) {
    const user = `You are shown ${frames.length} real frames sampled from a football video (in chronological order by source time).\nLocal motion evidence: ${JSON.stringify(motionDigest)}.\nDesign a premium 64-second 9:16 vertical montage in the style "${styleName}" using 14-22 shots.\nUse ONLY source times within 0..${Number(motion.duration || videoDuration).toFixed(2)}s.\nReturn JSON exactly: ${requiredShape}`;
    const r = await openRouterVision(frames, user, { system, model: OPENROUTER_VISION_MODEL, json: true, maxTokens: 2000 });
    providerHealth.record('openrouter', 'vision', r);
    telemetry.openrouterVision = telemetryLine(r);
    if (r.status === 'executed') {
      const parsed = parseAiPlan(r.data?.content);
      if (parsed) { telemetry.usedVision = true; return { plan: parsed, telemetry }; }
      telemetry.openrouterVision += ' | JSON had too few shots';
    }
  } else if (!providerConfigured('openrouter')) {
    telemetry.openrouterVision = 'not_configured (no OPENROUTER_API_KEY)';
  } else {
    telemetry.openrouterVision = 'failed (no frames could be extracted)';
  }

  // ---- 2) DeepSeek refines from the local motion evidence ----
  if (providerConfigured('deepseek')) {
    const user = `Refine this football montage plan. Write short editorial action labels and captions; do NOT invent events.\nLocal motion evidence: ${JSON.stringify(motionDigest)}\nCurrent plan: ${JSON.stringify({ timeline: (localPlan?.timeline || []).slice(0, 22) })}\nReturn JSON exactly: ${requiredShape}`;
    const r = await deepseekText(user, { system, model: DEEPSEEK_MODEL, json: true, maxTokens: 2000 });
    providerHealth.record('deepseek', 'text', r);
    telemetry.deepseek = telemetryLine(r);
    if (r.status === 'executed') {
      const parsed = parseAiPlan(r.data?.content);
      if (parsed) { telemetry.usedDeepseek = true; return { plan: parsed, telemetry }; }
      telemetry.deepseek += ' | JSON had too few shots';
    }
  } else {
    telemetry.deepseek = 'not_configured (no DEEPSEEK_API_KEY)';
  }

  return { plan: null, telemetry };
}

/**
 * Merge an optional AI-provider plan into the local plan: keep the local plan's
 * VERIFIED structure/timing as the backbone and adopt the provider's editorial
 * text/shot-type/zoom only where present. This guarantees an AI can NEVER move a
 * source timestamp outside the real footage.
 */
function mergeAiPlanIntoLocal(localPlan: any, aiPlan: any): any {
  const localTl = Array.isArray(localPlan?.timeline) ? localPlan.timeline : [];
  const aiTl = Array.isArray(aiPlan?.timeline) ? aiPlan.timeline : [];
  if (!localTl.length || !aiTl.length) return localPlan;
  const merged = localTl.map((c: any, i: number) => {
    const a = aiTl[Math.min(i, aiTl.length - 1)] || {};
    const text = typeof a.text === 'string' && a.text.trim() ? a.text : c.text;
    const action = typeof a.action === 'string' && a.action.trim() ? a.action.trim().slice(0, 160) : c.action;
    const shot = ['wide', 'medium', 'close_up', 'extreme_close_up', 'eye_close_up', 'action', 'reaction', 'crowd', 'detail'].includes(String(a.shot_type)) ? String(a.shot_type) : c.shot_type;
    const trans = ['hard_cut', 'directional_blur', 'flash'].includes(String(a.transition)) ? String(a.transition) : c.transition;
    return {
      ...c,
      text,
      action,
      shot_type: shot,
      transition: trans,
      narration: typeof a.narration === 'string' && a.narration.trim() ? a.narration.trim().slice(0, 90) : c.narration,
      zoom_start: Number.isFinite(Number(a.zoom_start)) ? Math.max(1, Math.min(1.55, Number(a.zoom_start))) : c.zoom_start,
      zoom_end: Number.isFinite(Number(a.zoom_end)) ? Math.max(1, Math.min(1.65, Number(a.zoom_end))) : c.zoom_end,
    };
  });
  return { ...localPlan, timeline: merged };
}

// Expose provider health + real-call proof (never returns keys).
app.get('/api/ai/status', (req, res) => {
  res.json({
    success: true,
    providers: {
      openrouter: { configured: providerConfigured('openrouter'), textModel: OPENROUTER_TEXT_MODEL, visionModel: OPENROUTER_VISION_MODEL },
      deepseek: { configured: providerConfigured('deepseek'), model: DEEPSEEK_MODEL },
    },
    // The same keys the subprocess VLM/DeepSeek layers would use (derived, never
    // the key value): proves the OpenRouter -> VLM mapping is active.
    vlmLayer: {
      enabled: Boolean(process.env.VLM_API_KEY),
      provider: process.env.VLM_PROVIDER || null,
      model: process.env.VLM_MODEL || null,
      base: process.env.VLM_BASE_URL || null,
    },
    lastCalls: providerHealth.snapshot(),
  });
});


/**
 * Build a real, source-grounded 64-second edit plan ENTIRELY from local motion.
 *
 * No external model is consulted. We partition the uploaded footage into a
 * montage of verified source windows, ordering the highest-motion windows
 * (impact) late so the edit climbs to a real climax, and pair them with
 * editorial, non-factual captions. Captions never assert a goal, score or name.
 */
function buildLocalEditPlan(videoDuration: number, analysis: LocalMotionAnalysis, styleName: string): any {
  const targetDuration = 64;
  const sourceDur = Math.max(1, Number(videoDuration) || Number(analysis?.duration) || targetDuration);
  const profile = (analysis?.motion_profile || []).filter((p) => Number.isFinite(p?.t) && Number.isFinite(p?.energy));
  const highThr = Number(analysis?.thresholds?.high ?? 0.6);

  // Pick candidate cut points. Prefer motion peaks (real impact moments); pad
  // with evenly spread points so a low-motion source still yields a full montage.
  const candidates = (analysis?.peaks?.length ? analysis.peaks.map((p) => p.t) : [])
    .filter((t) => t >= 0 && t < sourceDur)
    .sort((a, b) => a - b);
  const slotCount = Math.max(14, Math.min(22, candidates.length || 16));
  const cuts: number[] = candidates.slice(0, slotCount);
  if (cuts.length < slotCount) {
    for (let i = 0; i < slotCount && cuts.length < slotCount; i++) {
      const t = (sourceDur * (i + 0.5)) / slotCount;
      if (!cuts.some((c) => Math.abs(c - t) < 0.35)) cuts.push(t);
    }
  }
  cuts.sort((a, b) => a - b);

  const energyAt = (t: number): number => {
    if (!profile.length) return 0.5;
    let best = profile[0];
    let bestD = Math.abs(profile[0].t - t);
    for (const p of profile) {
      const d = Math.abs(p.t - t);
      if (d < bestD) { best = p; bestD = d; }
    }
    return Number(best.energy) || 0;
  };

  const pickShotType = (energy: number): string =>
    energy >= highThr ? 'action' : energy <= Number(analysis?.thresholds?.low ?? 0.4) ? 'close_up' : 'medium';

  const clips = cuts.map((t) => {
    const energy = energyAt(t);
    const start = Math.max(0, Math.min(sourceDur - 0.12, t));
    const end = Math.max(start + 0.4, Math.min(sourceDur, start + 2.4));
    const shotType = pickShotType(energy);
    const speed = shotType === 'medium' ? 1.1 : shotType === 'close_up' ? 0.78 : 1.0;
    return {
      source_start: Number(start.toFixed(3)),
      source_end: Number(end.toFixed(3)),
      action: `Real source moment @ ${start.toFixed(1)}s (local motion ${energy.toFixed(2)})`,
      importance: Math.round(3 + energy * 7),
      speed,
      zoom_start: 1.02,
      zoom_end: 1.08,
      crop_x: 0.5,
      crop_y: 0.5,
      transition: 'hard_cut',
      text: '',
      narration: '',
      shot_type: shotType,
      _energy: energy,
    };
  });

  // Arc ordering: keep the opening clip early, place the strongest verified
  // motion as the climax near the end instead of inventing a dramatic event.
  const indexed = clips.map((c, i) => ({ c, i }));
  indexed.sort((a, b) => Number(a.c._energy) - Number(b.c._energy));
  const ordered: any[] = new Array(clips.length);
  ordered[0] = clips[0];
  ordered[ordered.length - 1] = indexed[indexed.length - 1].c;
  const middle = indexed.slice(0, indexed.length - 1).map((x) => x.c);
  for (let i = 0; i < middle.length; i++) {
    if (ordered[i + 1] === undefined) ordered[i + 1] = middle[i];
  }
  const finalTimeline = ordered.filter(Boolean).slice(0, clips.length);

  // Editorial captions only — never a goal/score/name claim.
  const captionPool = ['WATCH THIS', 'THE BUILD UP', 'ONE MORE STEP', 'LOCKED IN', 'THE TOUCH', 'THE MOMENT', 'RIGHT NOW', 'GAME ON', 'NO WAY BACK', 'THAT MOVE'];
  const monologuePool = ['COME CLOSER', 'ONE MORE STEP', 'NOW YOU ARE MINE', 'I SAW IT COMING', 'TOO SLOW', 'WATCH THE EYES', 'THIS IS MY MOMENT', 'DO NOT BLINK'];
  const isDrama = String(styleName || '').toUpperCase().includes('PSYCHOLOGICAL');

  const tags = ['hook', 'setup', 'escalation', 'escalation', 'impact', 'reaction', 'climax', 'outro'];
  const timeline = finalTimeline.map((c: any, i: number) => ({
    timeline_index: i,
    source_start: c.source_start,
    source_end: c.source_end,
    output_start: Number(((targetDuration * i) / finalTimeline.length).toFixed(3)),
    output_end: Number(((targetDuration * (i + 1)) / finalTimeline.length).toFixed(3)),
    action: c.action,
    importance: c.importance,
    speed: c.speed,
    zoom_start: c.zoom_start,
    zoom_end: c.zoom_end,
    crop_x: c.crop_x,
    crop_y: c.crop_y,
    transition: c.transition,
    text: isDrama ? monologuePool[i % monologuePool.length] : captionPool[i % captionPool.length],
    narration: isDrama ? monologuePool[i % monologuePool.length] : '',
    shot_type: c.shot_type,
    beat_role: tags[Math.min(tags.length - 1, Math.floor((i / finalTimeline.length) * tags.length))],
    veo_needed: false,
    veo_prompt: '',
  }));
  timeline[timeline.length - 1].output_end = targetDuration;

  return {
    duration: targetDuration,
    aspect_ratio: '9:16',
    subject: { name: 'Main player', confidence: 0.6 },
    timeline,
    music: {
      style: 'Emotional cinematic football / dark trap pulse',
      bpm: 126,
      energy_curve: [0.72, 0.58, 0.65, 0.78, 0.9, 1, 0.82, 0.38],
    },
    circle_grade: undefined,
    color_grade: {
      contrast: isDrama ? 1.34 : 1.2,
      saturation: isDrama ? 1.16 : 1.04,
      highlights: -0.04,
      shadows: 0.02,
      grain: 0.035,
    },
    style_profile: {
      name: isDrama ? 'PSYCHOLOGICAL DRAMA' : 'REFERENCE CINEMATIC REEL',
      average_shot_duration: Number((targetDuration / timeline.length).toFixed(2)),
      zoom_intensity: isDrama ? 0.85 : 0.78,
      transition_frequency: 0.16,
      slow_motion_frequency: isDrama ? 0.55 : 0.3,
      text_frequency: 0.9,
      caption_style: isDrama
        ? 'cinematic white SERIF inner-monologue, ALL CAPS, centered lower third'
        : 'small white uppercase centered near lower third',
      visual_language: 'real source moments selected by LOCAL motion analysis; hard cuts, controlled punch-ins',
    },
    style_name: isDrama ? 'PSYCHOLOGICAL DRAMA' : 'REFERENCE CINEMATIC REEL',
  };
}


// ---------------------------------------------------------------------------
// Legacy Veo compatibility endpoints.
//
// The external Veo video-generation tier is no longer part of the pipeline
// (removing the last external dependency so a render can never fail on an API
// outage). These endpoints are kept so the old frontend manager receives a
// clean, JSON answer instead of a 404, and they always report a "fallback" so
// the client keeps rendering from the real uploaded footage.
// ---------------------------------------------------------------------------
app.post('/api/generate-veo-shot', (req, res) => {
  res.json({
    success: false,
    disabled: true,
    fallback: true,
    errorMessage: 'External AI shot generation is disabled. Rendering from the real uploaded footage.',
  });
});

app.post('/api/video-status', (req, res) => {
  res.json({ done: false, disabled: true, hasVideo: false, errorMessage: 'External AI shot generation is disabled.' });
});

app.post('/api/video-download', (req, res) => {
  res.status(410).json({ success: false, disabled: true, errorMessage: 'External AI shot generation is disabled.' });
});

/**
 * No-op now that the AI tier is source-only: it guarantees every clip is
 * rendered from the real uploaded footage and strips any stale Veo paths.
 */
// ---------------------------------------------------------------------------
// CINEMATIC ENGINE (OpenCV + NumPy) — OPTIONAL, per-clip pre-processing.
//
// This reuses the SAME per-clip override seam the renderer already understands
// (`clip.veo_local_path` => the renderer reads from this file instead of the
// source). No new render system is introduced and no route is added.
//
// DEFAULT BEHAVIOUR IS UNCHANGED: everything here is inert unless
// CINEMATIC_ENGINE_ENABLED=true. On any failure the clip keeps its original
// source (applied:false) so the existing pipeline is never broken.
//
// A clip is a smart-slow-motion candidate only when the edit plan already marks
// a genuinely slow shot (speed < 0.9). We then render THAT single clip through
// CinematicEngine.slow_motion_optical_flow, so only the intended moments are
// slowed — never the whole video.
// ---------------------------------------------------------------------------
async function applyCinematicEnginePrePass(
  inputPath: string,
  editPlan: any,
  onProgress?: (progress: FFmpegProgress) => void
): Promise<{ generatedPaths: string[]; processed: number }> {
  const timeline = Array.isArray(editPlan?.timeline) ? editPlan.timeline : [];
  const generatedPaths: string[] = [];
  if (!editPlan || !timeline.length) return { generatedPaths, processed: 0 };

  const engineTempDir = path.resolve(process.env.TMP_WORK_DIR || '/tmp/football_engine/work', 'cinematic_engine');
  try { fs.mkdirSync(engineTempDir, { recursive: true }); } catch {}

  let processed = 0;
  for (let i = 0; i < timeline.length; i++) {
    const clip = timeline[i];
    // Skip clips that already carry an override (e.g. an AI insert).
    if (typeof clip.veo_local_path === 'string' && fs.existsSync(clip.veo_local_path)) continue;

    const sourceStart = Math.max(0, Number(clip.source_start) || 0);
    const sourceEnd = Math.max(sourceStart + 0.08, Number(clip.source_end) || sourceStart + 1);
    const span = sourceEnd - sourceStart;
    const speed = Number(clip.speed) || 1;
    const isSlow = clip.beat_role === 'climax' || speed < 0.9;
    // Guard: only apply to short, genuinely-slow windows (cost + RAM safety).
    if (!isSlow || span <= 0.1 || span > 6.0) continue;

    const outPath = path.join(engineTempDir, `slow_${Date.now()}_${i}.mp4`);
    const slowFactor = Math.max(0.25, Math.min(1.0, speed));
    const res = await cinematicEngine.slowMotion(inputPath, outPath, slowFactor, sourceStart, sourceEnd);
    if (res.applied && res.outputPath) {
      clip.veo_local_path = res.outputPath;
      clip.veo_status = 'fallback';
      // The renderer fits a real [source_start, source_end] window into the
      // output slot; with the slowed file the source window immediately follows
      // the slowed playback start.
      clip.source_start = 0;
      clip.source_end = span * (1 / slowFactor);
      generatedPaths.push(res.outputPath);
      processed += 1;
      onProgress?.({
        percent: 6 + Math.floor((i / timeline.length) * 20),
        stage: `Cinematic engine: smart slow-motion on shot ${i + 1}/${timeline.length}...`,
      });
    }
  }
  return { generatedPaths, processed };
}

async function prepareVeoEnhancements(inputPath: string, editPlan: any, _generationTier: string, _onProgress?: (progress: FFmpegProgress) => void) {
  const timeline = Array.isArray(editPlan?.timeline) ? editPlan.timeline : [];
  for (const clip of timeline) {
    clip.veo_needed = false;
    clip.veo_status = 'not_requested';
    if (clip.veo_local_path) delete clip.veo_local_path;
  }

  // OPTIONAL cinematic-engine pre-pass (opt-in; inert by default). Failures are
  // swallowed and leave the plan exactly as-is. Any generated files are recorded
  // on clip.veo_local_path, so the renderer's existing cleanup handles them.
  if (cinematicEngine.enabled) {
    try {
      await applyCinematicEnginePrePass(inputPath, editPlan, _onProgress);
    } catch (err: any) {
      console.warn('[CinematicEngine] pre-pass skipped:', err?.message || err);
    }
  }

  return editPlan;
}

// ---------------------------------------------------------------------------
// OPTIONAL SAM SEGMENTATION / TRACKING — pre-pass integration.
//
// Runs BETWEEN the existing football analysis and the existing CinematicEngine,
// reusing the SAME per-clip 'sam' annotation seam the renderer understands.
// DEFAULT BEHAVIOUR IS UNCHANGED: inert unless SAM_ENABLED=true, and every
// failure degrades to { applied:false } so the YOLO/tracking path is untouched.
// It only ever touches the INTERESTING SEGMENTS the edit plan already selected.
// ---------------------------------------------------------------------------
function resolveSamArtifacts(editPlan: any, explicit?: { trackingPath?: string; eventsPath?: string }) {
  const trackingPath = explicit?.trackingPath && fs.existsSync(explicit.trackingPath) ? explicit.trackingPath : undefined;
  const eventsPath = explicit?.eventsPath && fs.existsSync(explicit.eventsPath) ? explicit.eventsPath : undefined;
  if (trackingPath && eventsPath) return { trackingPath, eventsPath };

  // The plan produced by /api/analyze-video carries the football-director
  // evidence, which references the events file. Derive the sibling tracking.json
  // from it (both live under the same yolo_<ts>/ directory) instead of
  // re-running any detection.
  const evidence = editPlan?.footballDirectorEvidence;
  let resolvedEvents = eventsPath;
  if (!resolvedEvents && typeof evidence?.eventsPath === 'string' && fs.existsSync(evidence.eventsPath)) {
    resolvedEvents = evidence.eventsPath;
  }
  let resolvedTracking = trackingPath;
  if (!resolvedTracking && resolvedEvents) {
    // <videosDir>/yolo_<ts>/director_<ts>/events.json  ->  <videosDir>/yolo_<ts>/tracking.json
    const yoloDir = path.dirname(path.dirname(resolvedEvents));
    const candidate = path.join(yoloDir, 'tracking.json');
    if (fs.existsSync(candidate)) resolvedTracking = candidate;
  }
  return { trackingPath: resolvedTracking, eventsPath: resolvedEvents };
}

async function applySamPrePass(
  inputPath: string,
  editPlan: any,
  onProgress?: (progress: FFmpegProgress) => void,
  explicit?: { trackingPath?: string; eventsPath?: string }
): Promise<{ applied: boolean }> {
  if (!samService.enabled) return { applied: false };
  try {
    const artifacts = resolveSamArtifacts(editPlan, explicit);
    const res = await samService.enhanceRenderPlan(inputPath, editPlan, artifacts, (p) =>
      onProgress?.({ percent: p.percent, stage: p.stage })
    );
    return { applied: res.applied };
  } catch (err: any) {
    console.warn('[SAM] pre-pass skipped:', err?.message || err);
    return { applied: false };
  }
}

// Live render progress tracking
let currentRenderProgress: FFmpegProgress = { percent: 0, stage: 'Idle' };

/**
 * Attach optional DEPTH masks (from the psychological depth pass) onto the plan's
 * clips as a `sam`-shaped block, so the EXISTING renderer applies its
 * subject-isolation path WITHOUT any new filter graph. Additive + opt-in: nothing
 * happens unless the depth pass produced a real mask.
 */
function storyApplyDepth(editPlan: any, depthSegments: any[]): void {
  const timeline = Array.isArray(editPlan?.timeline) ? editPlan.timeline : [];
  if (!timeline.length || !Array.isArray(depthSegments) || !depthSegments.length) return;
  for (const seg of depthSegments) {
    if (!seg || typeof seg.maskRef !== 'string' || !fs.existsSync(seg.maskRef)) continue;
    const t = Number(seg.time) || 0;
    let bestIdx = 0;
    let bestD = Number.POSITIVE_INFINITY;
    for (let i = 0; i < timeline.length; i++) {
      const s = Number(timeline[i].source_start) || 0;
      const e = Number(timeline[i].source_end) || s;
      const d = t >= s && t <= e ? 0 : Math.min(Math.abs(t - s), Math.abs(t - e));
      if (d < bestD) { bestD = d; bestIdx = i; }
    }
    const clip = timeline[bestIdx];
    clip.depth_effect = 'depth_bokeh';
    clip.sam = {
      applied: true,
      source: 'depth-anything-v2',
      maskRef: seg.maskRef,
      depth_strength: Number(seg.depth_strength) || 0.4,
      foreground_ratio: Number(seg.foreground_ratio) || 0,
    };
  }
}

/**
 * Map a measured ReferenceStyleProfile onto the EXISTING StyleProfile contract.
 * The full measured profile travels alongside it as `referenceStyleProfile`, so
 * the editor can use either shape without breaking legacy clients.
 */
function toLegacyStyleProfile(profile: any): any {
  const color = profile?.color || {};
  const trans = profile?.transition_weights || {};
  const avg = Number(profile?.avg_shot_duration) || 2.9;
  return {
    average_shot_duration: avg,
    zoom_intensity: Math.max(0, Math.min(1, (Number(profile?.zoom_intensity) || 0.014) / 0.03)),
    transition_frequency: Number(trans?.hard_cut ?? 0.9),
    slow_motion_frequency: Number(profile?.slow_motion_shot_ratio) || 0.15,
    text_frequency: Number(profile?.text_per_shot) || 0.1,
    color_style: `measured: contrast ${color?.contrast ?? 0}, saturation ${color?.saturation ?? 0}, skin ${color?.skin_ratio ?? 0}, neon-grass ${color?.neon_grass_ratio ?? 0}`,
    energy_curve: 'hook / setup / escalation / impact / reaction / climax / outro (measured cut density + audio impact sync)',
    recommended_bpm: Math.max(90, Math.min(150, Math.round((60 / Math.max(0.5, avg)) * 2))),
    cinematography_notes: `Measured from ${profile?.duration ?? 0}s / ${profile?.shot_count ?? 0} shots: cut density ${profile?.cut_density ?? 0}/s, subject-shot ratio ${profile?.subject_shot_ratio ?? 0}, hard-cut ratio ${trans?.hard_cut ?? 0}, cut-impact sync ${profile?.audio?.cut_impact_sync_ratio ?? 0}.`,
  };
}

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

// Prevent a long editor-selected interval from becoming an almost-original
// export. This only subdivides timestamps the local analysis already verified.
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
// The local analysis supplies the real source timestamps. If the plan is
// empty/invalid, the request fails instead of silently falling back to a fake
// timeline.
// ---------------------------------------------------------------------------
function validateAndEnforce64sEditPlan(data: any, videoDuration: number, styleName: string = ''): any {
  const duration = 64;
  if (!data || !Array.isArray(data.timeline) || data.timeline.length === 0) {
    throw new Error('Empty edit timeline. No synthetic fallback is allowed.');
  }
  if (!Number.isFinite(videoDuration) || videoDuration < 1) {
    throw new Error('Invalid source video duration.');
  }

  const subject = {
    name: typeof data.subject?.name === 'string' && data.subject.name.trim() ? data.subject.name.trim() : 'Main player',
    confidence: Math.max(0, Math.min(1, Number(data.subject?.confidence) || 0.5)),
  };

  // The reference reel uses a dense, emotional vertical montage. Keep the real
  // source moments selected by the local analysis and reject anything outside
  // the uploaded file.
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

  if (raw.length < 4) throw new Error('Too few usable real moments were found for a cinematic montage.');

  const denseRaw = densifyReferenceTimeline(raw, 14, 22);
  raw.length = 0;
  raw.push(...denseRaw.map((c: any, idx: number) => ({ ...c, timeline_index: idx })));

  // Preserve the editor's timing when valid; otherwise derive a cinematic rhythm
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

  // ORIGINAL FOOTAGE ONLY remains strictly source-only.

  // Reference-style captions are short, sparse, white, centered and editorial.
  // If no caption was supplied, generate only non-factual editorial micro-copy
  // from the verified action label; this does not claim an unobserved event.
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

// ---------------------------------------------------------------------------
// 6. FULL CINEMATIC RENDER — ASYNC JOB MODEL
// ---------------------------------------------------------------------------
// WHY: A full 64s montage legitimately takes 1.5-8 minutes of real FFmpeg work.
// An HTTP request held open that long is severed by hosting proxies (Render,
// nginx, Cloudflare) which cap origin response time — the browser then reports
// exactly the "Failed to fetch" seen in production, even though the server
// eventually finished. The fix is to DECOUPLE the render from the request:
//
//   POST /api/render-full-cinematic  -> starts a job, returns { jobId } in <1s
//   GET  /api/render-progress        -> live { percent, stage, status } (polled)
//   GET  /api/render-result          -> final payload once status === 'done'
//
// Progress is persisted to disk so a Render container restart (which briefly
// 502s the whole service) is reported as an explicit error instead of an
// endless "stuck at 73%" spinner.
// ---------------------------------------------------------------------------
type RenderJobStatus = 'running' | 'done' | 'error';
interface RenderJob {
  jobId: string;
  status: RenderJobStatus;
  percent: number;
  stage: string;
  startedAt: number;
  updatedAt: number;
  result?: any;
  error?: string;
}

const RENDER_STATE_DIR = path.resolve(process.env.TMP_WORK_DIR || '/tmp/football_engine/work');
try { fs.mkdirSync(RENDER_STATE_DIR, { recursive: true }); } catch {}
const RENDER_STATE_FILE = path.join(RENDER_STATE_DIR, 'render_state.json');
const RENDER_JOB_STALE_MS = Number(process.env.RENDER_JOB_STALE_MS || 20 * 60 * 1000);

const renderJobs = new Map<string, RenderJob>();
let activeRenderJobId: string | null = null;

function writeRenderState(job: RenderJob): void {
  try { fs.writeFileSync(RENDER_STATE_FILE, JSON.stringify(job), 'utf8'); } catch { /* non-fatal */ }
}
function clearRenderState(): void {
  try { if (fs.existsSync(RENDER_STATE_FILE)) fs.rmSync(RENDER_STATE_FILE, { force: true }); } catch { /* non-fatal */ }
}

/** Reads the job from memory, or the persisted snapshot if the process restarted. */
function readRenderJob(jobId?: string): RenderJob | null {
  const id = jobId || activeRenderJobId;
  if (id && renderJobs.has(id)) return renderJobs.get(id)!;
  if (!fs.existsSync(RENDER_STATE_FILE)) return null;
  try {
    const disk = JSON.parse(fs.readFileSync(RENDER_STATE_FILE, 'utf8')) as RenderJob;
    if (!disk || (jobId && disk.jobId !== jobId)) return null;
    // A 'running' job whose process died (server restart / OOM kill) is dead.
    if (disk.status === 'running' && Date.now() - (disk.updatedAt || 0) > RENDER_JOB_STALE_MS) {
      disk.status = 'error';
      disk.stage = 'Render was interrupted (the server restarted mid-render). Please retry.';
    }
    return disk;
  } catch { return null; }
}

function updateJob(job: RenderJob, patch: Partial<RenderJob>): void {
  Object.assign(job, patch, { updatedAt: Date.now() });
  currentRenderProgress = { percent: job.percent, stage: job.stage };
  writeRenderState(job);
}

/**
 * Runs the whole render pipeline OFF the request thread and records the outcome
 * on the job. Never throws to the caller — failures land on the job object where
 * the poller surfaces them.
 */
async function runFullRenderJob(
  jobId: string,
  params: { localPath: string; editPlan: any; musicVolume: number; originalVolume: number; generationTier: string; rifeMultiplier: number; precomputedArtifacts?: { trackingPath?: string; eventsPath?: string }; storyteller?: boolean }
): Promise<void> {
  const job = renderJobs.get(jobId)!;
  const { localPath, editPlan, musicVolume, originalVolume, generationTier, rifeMultiplier, precomputedArtifacts } = params;
  let generatedVeoPaths: string[] = [];
  let samApplied = false;
  let storytellingMetrics: Record<string, any> | null = null;
  let storyScript: any[] = [];
  try {
    const preparedPlan = await prepareVeoEnhancements(localPath, editPlan, generationTier, (p) => updateJob(job, { percent: p.percent, stage: p.stage }));

    // OPTIONAL psychological "Predator vs Prey" layer (opt-in; inert by default).
    // Runs AFTER the existing analysis produced the edit plan and BEFORE the
    // existing renderer. It only ADDS optional fields (story roles, story_script,
    // depth/low-angle/POV hints) and leaves every existing field untouched. Any
    // failure leaves the plan exactly as the GoalFlow cinematic pipeline built it.
    if (storytellerService.wants(String((preparedPlan as any)?.cinematicMode || (preparedPlan as any)?.style_name || ''), preparedPlan, params.storyteller)) {
      updateJob(job, { percent: 6, stage: 'Building the psychological story (duel detection)...' });
      try {
        const artifacts = storytellerService.resolveArtifacts(preparedPlan, precomputedArtifacts);
        let duration = Number((preparedPlan as any)?.duration) || 0;
        if (!Number.isFinite(duration) || duration <= 0) {
          try {
            const probe = await execPromise(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${JSON.stringify(localPath)}`);
            duration = Number.parseFloat(probe.stdout.trim()) || 64;
          } catch { duration = 64; }
        }
        const st = await storytellerService.enrich(localPath, duration, preparedPlan, artifacts);
        if (st.applied) {
          storytellingMetrics = st.metrics || null;
          storyScript = Array.isArray((preparedPlan as any).story_script) ? (preparedPlan as any).story_script : [];
          (preparedPlan as any).storyteller = {
            applied: true,
            duelDetected: Boolean(st.duelDetected),
            duelConfidence: Number(st.duelConfidence || 0),
            winner: st.winner ?? null,
            story_script_count: storyScript.length,
          };
          // OPTIONAL depth pass — only for the duel/psychological segments the
          // planner already marked. Inert unless DEPTH_ENABLED=true.
          const depthSegs = (((st.bundle || {}) as any).depth_segments || []) as Array<{ time: number; span?: number }>;
          if (depthSegs.length) {
            const maskDir = path.resolve(process.env.SAM_CACHE_DIR || '/tmp/football_engine/work', 'depth_masks');
            try { fs.mkdirSync(maskDir, { recursive: true }); } catch {}
            const depth = await storytellerService.depth(localPath, depthSegs, maskDir);
            if (depth.applied) {
              storyApplyDepth(preparedPlan, depth.segments || []);
              if (storytellingMetrics) storytellingMetrics.depth_effect_used = true;
            }
          }
        }
      } catch (stErr: any) {
        console.warn('[Storyteller] layer skipped:', stErr?.message || stErr);
      }
    }

    // OPTIONAL SAM segmentation/tracking pre-pass (opt-in; inert by default).
    // Runs AFTER the existing analysis produced the edit plan and BEFORE the
    // existing CinematicEngine renders it. Any failure leaves the plan untouched.
    if (samService.enabled) {
      updateJob(job, { percent: 10, stage: 'Preparing segmentation...' });
      let artifacts = precomputedArtifacts;
      if (!resolveSamArtifacts(preparedPlan, artifacts).trackingPath) {
        // SAM needs YOLO DETECTION prompts (SAM never detects by itself). If the
        // analyze step ran with YOLO off (the CPU default), reuse the EXISTING
        // bounded YOLO tracker ONCE here so the interesting segments can be
        // segmented. This reuses the same function the analysis stage uses.
        try {
          const tracking = await runYoloTracking(localPath, (p) => updateJob(job, { percent: p.percent, stage: p.stage }));
          const evidence = await runFootballEvidence(localPath, tracking);
          artifacts = {
            trackingPath: tracking?.jsonUrl ? path.join(videosDir, String(tracking.jsonUrl).replace(/^\/videos\//, '')) : undefined,
            eventsPath: evidence?.eventsPath,
          };
        } catch (trackErr: any) {
          console.warn('[SAM] detection pass skipped:', trackErr?.message || trackErr);
        }
      }
      await applySamPrePass(localPath, preparedPlan, (p) => updateJob(job, { percent: p.percent, stage: p.stage }), artifacts);
      samApplied = Boolean((preparedPlan as any).samSegmentation?.applied);
    }

    // NOTE: engine-generated slow-motion files are registered as clip.veo_local_path
    // by the pre-pass, so this list already covers them (cleaned up in finally).
    generatedVeoPaths = preparedPlan.timeline.map((c: any) => c.veo_local_path).filter((p: any) => typeof p === 'string');

    const result = await ffmpegEngine.renderFullCinematic(
      localPath,
      preparedPlan,
      musicVolume,
      originalVolume,
      (p) => updateJob(job, { percent: p.percent, stage: p.stage })
    );

    let publishedPath = result.localPath;
    let rifeApplied = false;
    if (process.env.RIFE_ENABLED !== 'false' && (rifeMultiplier === 2 || rifeMultiplier === 4)) {
      updateJob(job, { percent: 94, stage: `RIFE frame interpolation ${rifeMultiplier}x...` });
      const rifeOut = path.join(path.dirname(publishedPath), `${path.parse(publishedPath).name}_${rifeMultiplier}x.mp4`);
      try {
        const rife = await runRifeInterpolation(publishedPath, rifeOut, rifeMultiplier === 4 ? 2 : 1);
        if (rife.applied) { publishedPath = rife.outputPath; rifeApplied = true; }
      } catch (rifeErr: any) {
        console.warn('[RIFE] interpolation failed; keeping FFmpeg master:', rifeErr?.message || rifeErr);
      }
    }

    const payload = {
      success: true,
      ...result,
      localPath: publishedPath,
      videoUrl: publishedPath ? publishedPath.replace(videosDir, '/videos') : result.videoUrl,
      generationTier,
      aiEnhanced: generationTier !== 'ORIGINAL FOOTAGE ONLY' && generatedVeoPaths.length > 0,
      rifeApplied,
      rifeMultiplier: rifeApplied ? rifeMultiplier : 0,
      // Additive observability flag: whether the OPTIONAL SAM layer contributed
      // subject-isolation references to this render. The frontend contract is
      // unchanged (extra field only).
      samApplied,
      // Additive: the OPTIONAL psychological "Predator vs Prey" layer. `story_script`
      // is always present ([] when the layer produced nothing), so existing clients
      // keep working. All other existing fields (videoUrl/posterUrl/...) are intact.
      story_script: Array.isArray(storyScript) ? storyScript : [],
      storytelling: storytellingMetrics,
    };
    updateJob(job, { percent: 100, status: 'done', stage: rifeApplied ? 'Cinematic render + RIFE complete.' : 'Cinematic render complete.', result: payload });
    console.log(`[RENDER JOB ${jobId}] done in ${((Date.now() - job.startedAt) / 1000).toFixed(1)}s`);
  } catch (err: any) {
    console.error(`[RENDER JOB ${jobId}] failed:`, err);
    updateJob(job, { status: 'error', error: err?.message || 'Render failed.' });
  } finally {
    // Generated bridge shots are temporary render assets; the published master is persistent.
    for (const filePath of generatedVeoPaths) {
      try { fs.rmSync(filePath, { force: true }); } catch {}
      try { fs.rmSync(path.dirname(filePath), { recursive: true, force: true }); } catch {}
    }
    if (activeRenderJobId === jobId) activeRenderJobId = null;
  }
}

// 6. POST /api/render-full-cinematic — starts a render job, returns immediately.
app.post('/api/render-full-cinematic', async (req, res) => {
  const { localPath, editPlan, musicVolume = 0.8, originalVolume = 0.9, generationTier = 'ORIGINAL FOOTAGE ONLY', rifeMultiplier, artifacts, storyteller } = req.body || {};
  if (!localPath || !fs.existsSync(localPath)) {
    return res.status(400).json({ success: false, error: 'Valid uploaded video localPath is required.' });
  }

  // Single-render-at-a-time guard: FFmpeg renders are CPU/RAM heavy. Report a
  // conflict instead of launching a second one that would OOM the container.
  const existing = activeRenderJobId ? renderJobs.get(activeRenderJobId) : null;
  if (existing && existing.status === 'running') {
    return res.status(409).json({
      success: false,
      busy: true,
      jobId: existing.jobId,
      error: 'A render is already in progress on this server.',
    });
  }

  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const job: RenderJob = {
    jobId,
    status: 'running',
    percent: 0,
    stage: 'Initializing real FFmpeg render engine...',
    startedAt: Date.now(),
    updatedAt: Date.now(),
  };
  renderJobs.set(jobId, job);
  activeRenderJobId = jobId;
  writeRenderState(job);
  currentRenderProgress = { percent: 0, stage: job.stage };

  const multiplier = Number(rifeMultiplier || process.env.RIFE_FPS_MULTIPLIER || 2);
  // Optional SAM artefacts (existing tracking/events paths) may be forwarded by
  // the analyze step so SAM never re-runs detection needlessly.
  const precomputedArtifacts = artifacts && typeof artifacts === 'object'
    ? { trackingPath: artifacts.trackingPath, eventsPath: artifacts.eventsPath }
    : undefined;
  // Fire-and-forget: the heavy work continues after this response is sent.
  void runFullRenderJob(jobId, { localPath, editPlan, musicVolume, originalVolume, generationTier, rifeMultiplier: multiplier, precomputedArtifacts, storyteller });

  return res.status(202).json({
    success: true,
    jobId,
    status: 'running',
    // Legacy clients that expect a synchronous result can detect the async
    // contract via `async: true` and start polling /api/render-progress.
    async: true,
    message: 'Render started. Poll /api/render-progress until status is done, then read /api/render-result.',
  });
});

// 7. GET /api/render-progress — live progress for the active/most-recent job.
app.get('/api/render-progress', (req, res) => {
  const job = readRenderJob(typeof req.query.jobId === 'string' ? req.query.jobId : undefined);
  if (!job) {
    return res.json({ ...currentRenderProgress, status: 'idle' });
  }
  res.json({
    percent: job.percent,
    stage: job.stage,
    status: job.status,
    jobId: job.jobId,
    error: job.status === 'error' ? job.error : undefined,
  });
});

// 7.1 GET /api/render-result — final artifact once the job is done.
app.get('/api/render-result', (req, res) => {
  const jobId = typeof req.query.jobId === 'string' ? req.query.jobId : undefined;
  const job = readRenderJob(jobId);
  if (!job) return res.status(404).json({ success: false, status: 'idle', error: 'No render job found.' });
  if (job.status === 'running') {
    return res.status(202).json({ success: false, status: 'running', percent: job.percent, stage: job.stage, jobId: job.jobId });
  }
  if (job.status === 'error') {
    return res.status(500).json({ success: false, status: 'error', error: job.error || 'Render failed.', jobId: job.jobId });
  }
  return res.json(job.result || { success: true, jobId: job.jobId });
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

function runYoloTracking(
  inputPath: string,
  onProgress?: (p: { percent: number; stage: string }) => void
): Promise<any> {
  return new Promise((resolve, reject) => {
    const script = path.join(__dirname, 'yolo', 'track_football.py');
    const modelPath = process.env.YOLO_MODEL_PATH || path.join(__dirname, 'models', 'best.pt');
    const outputDir = path.join(videosDir, `yolo_${Date.now()}`);
    fs.mkdirSync(outputDir, { recursive: true });

    // Bounded CPU work + a hard wall-clock timeout so tracking can NEVER hang the
    // whole pipeline (previously it could run for hours on a long CPU-only match).
    const maxSeconds = Number(process.env.YOLO_MAX_SECONDS || 120);
    const stride = Number(process.env.YOLO_STRIDE || 3);
    // Wall-clock budget for the whole tracking pass. Kept SHORT because CPU-only
    // hosts can take tens of seconds per frame; a long budget = a frozen UI.
    const deadlineSeconds = Number(process.env.YOLO_DEADLINE_SECONDS || 90);
    const timeoutMs = Number(process.env.YOLO_TIMEOUT_MS || 180000); // 3 min hard cap

    const child = spawn(process.env.PYTHON_BIN || 'python3',
      [script, '--source', inputPath, '--model', modelPath, '--output-dir', outputDir, '--json',
       '--max-seconds', String(maxSeconds), '--stride', String(stride),
       '--deadline-seconds', String(deadlineSeconds)],
      { env: { ...process.env }, stdio: ['ignore', 'pipe', 'pipe'] });

    let stdout = '', stderr = '', settled = false;
    let lastLines: string[] = [];

    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn();
    };

    const timer = setTimeout(() => {
      try { child.kill('SIGKILL'); } catch {}
      finish(() => reject(new Error(
        `YOLO tracking timed out after ${Math.round(timeoutMs / 1000)}s. ` +
        `Lower YOLO_MAX_SECONDS or increase YOLO_TIMEOUT_MS.`
      )));
    }, timeoutMs);

    child.stdout.on('data', d => { stdout += d.toString(); });
    child.stderr.on('data', d => {
      stderr += d.toString();
      // Parse our "PROGRESS <pct> <msg>" heartbeat lines into live UI updates.
      for (const line of d.toString().split('\n')) {
        if (line.startsWith('PROGRESS ')) {
          const parts = line.replace('PROGRESS ', '').trim().split(' ');
          const pct = Number(parts.shift());
          if (Number.isFinite(pct)) {
            onProgress?.({ percent: Math.max(4, Math.min(20, pct)), stage: `YOLOv8 tracking: ${parts.join(' ')}` });
          }
        } else if (line.trim()) {
          lastLines.push(line.trim());
          if (lastLines.length > 20) lastLines.shift();
        }
      }
    });

    child.on('error', (err) => finish(() => reject(err)));
    child.on('close', code => {
      finish(() => {
        if (code !== 0) {
          return reject(new Error(`YOLO tracker exited ${code}: ${(stderr || lastLines.join(' ')).slice(-1500)}`));
        }
        try {
          const result = JSON.parse(stdout.trim().split('\n').filter(Boolean).pop() || '{}');
          if (!result.success) return reject(new Error(result.error || 'YOLO tracking failed'));
          resolve(result);
        } catch { reject(new Error(`YOLO tracker returned invalid JSON: ${stdout.slice(-1000)}`)); }
      });
    });
  });
}

/**
 * Build the Arabic voice-over script from the VERIFIED timeline only.
 *
 * This is fully LOCAL: no external model is consulted. It reuses the
 * per-shot editorial "narration" lines already produced by the editor and maps
 * them to delivery parameters (emotion/intensity/pause) from the shot's beat
 * role. Nothing about a goal, player, score or event is ever invented.
 */
async function generateCommentaryScript(plan: any, style: string): Promise<any[]> {
  const timeline = Array.isArray(plan?.timeline) ? plan.timeline : [];

  // LOCAL fallback: build the script from the per-shot editorial `narration`
  // lines already produced by the editor. Nothing is invented.
  const localLines = timeline
    .filter((t: any) => t && String(t.narration || '').trim())
    .map((t: any, i: number) => ({
      timeline_index: Number(t.timeline_index ?? i),
      text: String(t.narration).trim().slice(0, 120),
      emotion: ['hook', 'impact', 'climax'].includes(t.beat_role) ? 'intense' : 'focused',
      intensity: t.beat_role === 'climax' ? 0.95 : ['impact', 'reaction'].includes(t.beat_role) ? 0.78 : 0.55,
      pause_after_ms: t.beat_role === 'climax' ? 320 : 220,
    }))
    .slice(0, 14);

  // OPTIONAL: let OpenRouter (preferred) or DeepSeek WRITE natural Arabic
  // commentary from the VERIFIED timeline. The prompt forbids inventing events.
  if (!providerConfigured('openrouter') && !providerConfigured('deepseek')) {
    return localLines;
  }
  const verified = timeline.map((t: any) => ({
    timeline_index: t.timeline_index,
    source: [t.source_start, t.source_end],
    output: [t.output_start, t.output_end],
    action: String(t.action || '').slice(0, 90),
    beat_role: t.beat_role,
  }));
  const user = `اكتب تعليقًا صوتيًا عربيًا طبيعيًا (كمعلّق محترف) لمقطع كرة قدم سينمائي مدته 64 ثانية.
لا تخترع هدفًا أو تمريرة أو اسم لاعب أو نتيجة أو بطولة؛ استخدم فقط ما يثبته action في كل لقطة.
جمل قصيرة متفاوتة الطول، انفعالات تتصاعد نحو climax، واترك بعض اللقطات بلا كلام.
الأسلوب: ${style}
الخط الزمني الموثق: ${JSON.stringify(verified)}
أخرج JSON فقط بالشكل: [{"timeline_index":number,"text":string,"emotion":"calm|focused|excited|intense|shock|triumphant|emotional","intensity":0.0,"pause_after_ms":number}]
الحد الأقصى 14 سطرًا، وكل سطر من 3 إلى 16 كلمة.`;
  const system = `You are a professional Arabic football commentator. Return STRICT JSON only. Never invent a goal, score, player, or event that the timeline does not prove.`;

  const tryParse = (content: any): any[] | null => {
    try {
      const parsed = JSON.parse(String(content || '').replace(/^```(?:json)?/i, '').replace(/```$/, '').trim());
      if (Array.isArray(parsed) && parsed.length) {
        return parsed
          .filter((l: any) => l && String(l.text || '').trim())
          .map((l: any, i: number) => ({
            timeline_index: Number(l.timeline_index ?? i),
            text: String(l.text).trim().slice(0, 120),
            emotion: String(l.emotion || 'focused'),
            intensity: Math.max(0, Math.min(1, Number(l.intensity) || 0.55)),
            pause_after_ms: Math.max(0, Math.min(1200, Number(l.pause_after_ms) || 220)),
          }))
          .slice(0, 14);
      }
    } catch { /* fall through */ }
    return null;
  };

  if (providerConfigured('openrouter')) {
    const r = await openRouterText(user, { system, model: OPENROUTER_TEXT_MODEL, json: true, maxTokens: 1200 });
    providerHealth.record('openrouter', 'text', r);
    if (r.status === 'executed') {
      const parsed = tryParse(r.data?.content);
      if (parsed && parsed.length) return parsed;
    }
  }
  if (providerConfigured('deepseek')) {
    const r = await deepseekText(user, { system, model: DEEPSEEK_MODEL, json: true, maxTokens: 1200 });
    providerHealth.record('deepseek', 'text', r);
    if (r.status === 'executed') {
      const parsed = tryParse(r.data?.content);
      if (parsed && parsed.length) return parsed;
    }
  }
  return localLines;
}


async function probeAudioDuration(filePath: string): Promise<number> {
  const p = await execPromise(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${JSON.stringify(filePath)}`);
  return Math.max(0, Number.parseFloat(p.stdout.trim()) || 0);
}

async function buildTimedCommentary(plan: any, style: string, dryRun = false): Promise<{ audioPath: string; script: any[] }> {
  const script = await generateCommentaryScript(plan, style);
  // dryRun: return the (optionally AI-generated) script WITHOUT synthesising
  // audio — lets ops verify the commentary TEXT provider with no TTS configured.
  if (dryRun) return { audioPath: '', script };
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
  res.json({ success: true, enabled: process.env.YOLO_ENABLED === 'true', model: process.env.YOLO_MODEL_PATH || 'models/best.pt', tracker: 'ByteTrack' });
});

// OPTIONAL SAM layer status. This is a read-only readiness probe (no model load,
// no network) so it is safe to poll from the UI. It NEVER starts a render and it
// respects the existing frontend contract (always returns JSON).
app.get('/api/sam/status', async (req, res) => {
  try {
    const force = req.query.refresh === '1';
    const status = await samService.status(force);
    res.json({ success: true, ...status });
  } catch (err: any) {
    res.json({ success: true, enabled: samService.enabled, available: false, reason: err?.message || 'status unavailable' });
  }
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
    const pyTimeout = Number(process.env.EVIDENCE_TIMEOUT_MS || 120000);
    await execPromise(`${JSON.stringify(py)} ${JSON.stringify(path.join(__dirname,'yolo/event_engine.py'))} --tracking ${JSON.stringify(trackingPath)} --output ${JSON.stringify(eventsPath)}`, { timeout: pyTimeout });
    const summary=trackingResult.summary||{};
    // Prefer the real source duration; fall back to analyzed frames/fps.
    const analyzedFrames = Number(summary.sourceFrames) || Number(summary.frames) || 0;
    const duration = analyzedFrames > 0
      ? analyzedFrames / Math.max(.1, Number(summary.fps) || 25)
      : Number(process.env.YOLO_MAX_SECONDS || 120);
    await execPromise(`${JSON.stringify(py)} ${JSON.stringify(path.join(__dirname,'yolo/football_director.py'))} --events ${JSON.stringify(eventsPath)} --output ${JSON.stringify(directorPath)} --duration ${duration.toFixed(3)}`, { timeout: pyTimeout });
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
    await execPromise(`${JSON.stringify(py)} ${JSON.stringify(script)} --events ${JSON.stringify(eventsPath)} --timeline ${JSON.stringify(timelineInput)} --output ${JSON.stringify(madnessPath)}`, { timeout: Number(process.env.EVIDENCE_TIMEOUT_MS || 120000) });
    if (!fs.existsSync(madnessPath)) return null;
    return JSON.parse(fs.readFileSync(madnessPath, 'utf8'));
  } catch (e:any) {
    console.warn('[MADNESS] evidence pass skipped:', e?.message || e);
    return null;
  }
}

// 8. POST /api/analyze-video
// LOCAL analysis only: the uploaded video is sampled with OpenCV
// (yolo/local_motion_analysis.py). NO external video API and NO API keys are
// used, so this route can never fail because of a Gemini/Veo outage.
app.post('/api/analyze-video', async (req, res) => {
  try {
    const {
      videoMetadata,
      style = 'CINEMATIC SPORTS',
      generationTier = 'ORIGINAL FOOTAGE ONLY',
      trackingEnabled = true,
      // Cinematic Mode: STANDARD | PRO | REFERENCE. Defaults to STANDARD so the
      // existing behaviour is preserved for every legacy client.
      cinematicMode = 'STANDARD',
      referenceStyle = null,
      referenceLocalPath = null,
    } = req.body;
    const localPath = videoMetadata?.localPath;
    let duration = Number(videoMetadata?.duration);
    if (!localPath || !fs.existsSync(localPath)) {
      return res.status(400).json({ success: false, error: 'Uploaded video file is not available on the backend.' });
    }
    if (!Number.isFinite(duration) || duration <= 0) {
      // Never reject on a missing/bogus client duration: probe the real file.
      try {
        const { stdout } = await execPromise(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${JSON.stringify(localPath)}`);
        duration = Number.parseFloat(stdout.trim()) || 0;
      } catch { /* fall through to the local analysis duration below */ }
    }

    currentRenderProgress = { percent: 3, stage: 'Analyzing local motion of the uploaded footage...' };

    // 1) LOCAL motion analysis (no API). Never throws: it degrades to a valid
    //    uniform profile if OpenCV/decoding is unavailable.
    const motion = await runLocalMotionAnalysis(localPath);
    if (!Number.isFinite(duration) || duration <= 0) {
      duration = Number(motion.duration) || 64;
    }
    currentRenderProgress = { percent: 10, stage: 'Building the 64-second edit plan from local motion...' };

    // 2) OPTIONAL YOLO evidence enhancer. Off by default (CPU-heavy); its failure
    //    never blocks the render.
    let trackingResult: any = null;
    const yoloExplicitlyEnabled = process.env.YOLO_ENABLED === 'true';
    if (trackingEnabled && yoloExplicitlyEnabled) {
      currentRenderProgress = { percent: 4, stage: 'YOLOv8 tracking real players and the ball...' };
      try {
        trackingResult = await runYoloTracking(localPath, (p) => { currentRenderProgress = p; });
      } catch (trackErr: any) {
        console.warn('[YOLO] Tracking skipped:', trackErr?.message || trackErr);
        trackingResult = null;
      }
    }
    const footballEvidence = trackingResult ? await runFootballEvidence(localPath, trackingResult) : null;

    // 3) Edit Plan V2 via the Cinematic Director.
    //
    // The director is STYLE-DRIVEN (STANDARD / PRO / the measured reference
    // profile) and EVENT-DRIVEN (real motion + real YOLO tracking/events). It
    // never copies the reference video's timestamps, shot order or frames.
    //
    // FAIL-SAFE: if the director is unavailable, rejects its own validation, or
    // returns nothing usable, the existing buildLocalEditPlan() result is kept
    // untouched, so the render contract and behaviour never regress.
    const resolvedMode = (['STANDARD', 'PRO', 'REFERENCE'].includes(String(cinematicMode).toUpperCase())
      ? String(cinematicMode).toUpperCase()
      : 'STANDARD') as CinematicMode;

    // Accept BOTH profile shapes: the measured ReferenceStyleProfile from
    // /api/reference-style/analyze, and the legacy StyleProfile contract.
    let normalizedReference: any = null;
    if (referenceStyle && typeof referenceStyle === 'object') {
      normalizedReference = referenceStyle.avg_shot_duration !== undefined
        ? referenceStyle
        : {
            source: 'client-style-profile',
            avg_shot_duration: Number(referenceStyle.average_shot_duration) || 2.9,
            cut_density: referenceStyle.average_shot_duration ? 1 / Number(referenceStyle.average_shot_duration) : 0.34,
            zoom_intensity: Number(referenceStyle.zoom_intensity) || 0.014,
            slow_motion_shot_ratio: Number(referenceStyle.slow_motion_frequency) || 0.15,
            text_per_shot: Number(referenceStyle.text_frequency) || 0.1,
            transition_weights: { hard_cut: Number(referenceStyle.transition_frequency) || 0.84 },
            shot_type_weights: { close_up: 0.6, action: 0.25, wide: 0.15 },
            color: {},
          };
    }

    currentRenderProgress = { percent: 14, stage: `Cinematic Director (${resolvedMode}) is building Edit Plan V2...` };
    const director = await referenceStyleService.buildEditPlan({
      duration,
      motion,
      eventsPath: (footballEvidence as any)?.eventsPath,
      trackingPath: trackingResult?.jsonUrl
        ? path.join(videosDir, String(trackingResult.jsonUrl).replace(/^\/videos\//, ''))
        : undefined,
      referenceStyle: normalizedReference,
      referenceLocalPath: typeof referenceLocalPath === 'string' && referenceLocalPath ? referenceLocalPath : undefined,
      mode: resolvedMode,
      subject: videoMetadata?.defaultSubject,
    });

    let parsed = director.applied && director.plan
      ? director.plan
      : buildLocalEditPlan(duration, motion, style);
    parsed.generationTier = generationTier;
    parsed.styleName = style;
    (parsed as any).cinematicMode = resolvedMode;
    if (director.applied) {
      (parsed as any).cinematicDirectorSource = 'cinematic-director-v11';
    } else {
      (parsed as any).cinematicDirectorSource = 'local-motion-fallback';
      (parsed as any).cinematicDirectorMessage = director.message || 'director unavailable';
    }

    // 3b) OPTIONAL enrichment: OpenRouter VLM (REAL extracted frames) primary,
    //     DeepSeek (local motion evidence) secondary. When no key is configured,
    //     or the call fails, this returns null and the local/director plan is
    //     kept as-is — the render NEVER stops.
    currentRenderProgress = { percent: 18, stage: 'Optional AI enrichment (OpenRouter VLM / DeepSeek)...' };
    const enhancement = await enhancePlanWithOptionalProviders(localPath, duration, parsed, motion, style);
    const usedEnhancedPlan = Boolean(enhancement.plan);
    if (enhancement.plan) {
      parsed = mergeAiPlanIntoLocal(parsed, enhancement.plan);
    }

    currentRenderProgress = { percent: 22, stage: 'Validating the 64-second edit plan...' };
    // validateAndEnforce64sEditPlan re-normalises timing/captions WITHOUT touching
    // the verified source timestamps.
    const validatedPlan = validateAndEnforce64sEditPlan(parsed, duration, style);
    if (footballEvidence) (validatedPlan as any).footballDirectorEvidence = footballEvidence;

    // MADNESS-5 escalation runs only with real YOLO evidence.
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
      analysisMode: usedEnhancedPlan
        ? (director.applied ? 'cinematic-director-v11 + optional-ai' : 'local-motion + optional-ai')
        : (director.applied ? 'cinematic-director-v11' : 'local-motion'),
      model: motion.model || 'local-opencv-motion',
      // Secret-free provider telemetry: executed/failed/not_configured + http + ms.
      aiProviders: {
        openrouterVision: enhancement.telemetry.openrouterVision,
        openrouterText: enhancement.telemetry.openrouterText,
        deepseek: enhancement.telemetry.deepseek,
        enhancedPlanUsed: usedEnhancedPlan,
      },
      cinematicMode: resolvedMode,
      cinematicDirector: {
        applied: director.applied,
        source: (parsed as any).cinematicDirectorSource,
        clips: director.clips || 0,
        heroMoment: director.heroMoment || null,
        evidenceLevel: director.evidenceLevel || null,
        referenceStyleSource: normalizedReference?.source || null,
        message: director.message || null,
      },
      motion: {
        samples: motion.motion_profile.length,
        slowMoments: motion.slow_moments.length,
        actionMoments: motion.action_moments.length,
        meanEnergy: motion.mean_energy,
      },
      madness: madnessResult ? { version: madnessResult.version, counts: madnessResult.counts } : { enabled: false },
      tracking: trackingResult ? { success: true, videoUrl: trackingResult.videoUrl, jsonUrl: trackingResult.jsonUrl, summary: trackingResult.summary } : { success: false }
    });
  } catch (err: any) {
    console.error('Local video analysis failed:', err);
    res.status(500).json({
      success: false,
      videoAnalyzed: false,
      error: err?.message || 'Local video analysis failed.',
      code: 'LOCAL_VIDEO_ANALYSIS_FAILED',
    });
  }
});
// 8.5. POST /api/commentary/generate
app.post('/api/commentary/generate', async (req, res) => {
  try {
    const { editPlan, style = 'CINEMATIC SPORTS', videoLocalPath, dryRun = false } = req.body;
    const generated = await buildTimedCommentary(editPlan, style, Boolean(dryRun));
    const script = generated.script;
    const audioPath = generated.audioPath;
    if (dryRun) {
      return res.json({ success: true, dryRun: true, script, provider: 'text-only' });
    }
    const finalVideoPath = videoLocalPath && fs.existsSync(videoLocalPath) && audioPath ? await muxCommentary(videoLocalPath, audioPath) : undefined;
    res.json({
      success: true, provider: 'ElevenLabs', script, commentaryAudioUrl: audioPath ? `/videos/${path.basename(audioPath)}` : undefined,
      audioUrl: audioPath ? `/videos/${path.basename(audioPath)}` : undefined,
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
/**
 * LOCAL quality control. Inspects the REAL rendered MP4 with ffprobe + the same
 * local motion pass (no external model) and reports factual pacing feedback.
 */
async function runLocalQc(outputLocalPath: string, editPlan: any, style: string) {
  let outDuration = 0;
  let width = 0;
  let height = 0;
  try {
    const { stdout } = await execPromise(`ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of json ${JSON.stringify(outputLocalPath)}`);
    const probe = JSON.parse(stdout);
    outDuration = Number(probe?.format?.duration) || 0;
    const v = probe?.streams?.find((s: any) => s.width && s.height);
    if (v) { width = v.width; height = v.height; }
  } catch { /* keep defaults */ }

  let motion: LocalMotionAnalysis | null = null;
  try { motion = await runLocalMotionAnalysis(outputLocalPath); } catch { /* optional */ }

  const shots = Array.isArray(editPlan?.timeline) ? editPlan.timeline.length : 0;
  const avgShot = shots > 0 ? outDuration / shots : 0;

  // Pacing: a 64s vertical reel reads best with ~14-22 shots (2.9-4.6s each but
  // cut into sub-beats). Score how close the actual output is to that band.
  const pacingScore = shots === 0 ? 6 : Math.max(1, Math.min(10, Number((10 - Math.abs(16 - shots) * 0.45).toFixed(1))));
  const durationOk = Math.abs(outDuration - 64) < 1.5;
  const cinematicScore = Math.max(1, Math.min(10, Number((pacingScore * 0.6 + (durationOk ? 4 : 1.5)).toFixed(1))));

  const corrections: any[] = [];
  if (!durationOk && shots > 0) {
    corrections.push({ timeline_index: shots - 1, change: 'trim_duration', reason: `Rendered duration is ${outDuration.toFixed(1)}s, expected 64s.` });
  }
  if (shots > 0 && shots < 14) {
    corrections.push({ timeline_index: 0, change: 'adjust_speed', reason: `Only ${shots} shots detected; increase montage density for the reference reel.`, recommended_speed: 1.0 });
  }

  const verdict = cinematicScore >= 8.5 ? 'EXCELLENT' : corrections.length ? 'APPROVED_WITH_TWEAKS' : 'APPROVED_WITH_TWEAKS';

  return {
    qc_verdict: verdict,
    overall_critique: `Local analysis: ${shots} shots over ${outDuration.toFixed(1)}s (avg ${avgShot.toFixed(2)}s). ${
      motion ? `Measured mean motion energy ${Number(motion.mean_energy || 0).toFixed(2)} with ${motion.action_moments.length} high-motion beats.` : 'Motion profile unavailable.'
    } Output is 9:16 ${width || 1080}x${height || 1920}.`,
    pacing_score: pacingScore,
    cinematic_score: cinematicScore,
    corrections,
  };
}

// 9. POST /api/qc-review  (LOCAL QC — no external API)
app.post('/api/qc-review', async (req, res) => {
  try {
    const { outputLocalPath, editPlan, style = 'CINEMATIC SPORTS' } = req.body;
    if (!outputLocalPath || !fs.existsSync(outputLocalPath)) {
      return res.status(400).json({ success: false, error: 'Rendered video localPath is required for visual QC.' });
    }
    const review = await runLocalQc(outputLocalPath, editPlan, style);
    res.json({ success: true, review, videoAnalyzed: true, analysisMode: 'local' });
  } catch (err: any) {
    console.error('Error during local QC:', err);
    res.status(500).json({ success: false, videoAnalyzed: false, error: err?.message || 'Visual QC failed.' });
  }
});

// 10. POST /api/analyze-reference  (LOCAL — publishes a curated editorial profile)
// The reference reel is a STYLE preset, not a downloaded/copied video: we return
// a deterministic editorial profile so the editor applies the same rhythm without
// any external model call.
// ---------------------------------------------------------------------------
// REFERENCE STYLE + CINEMATIC DIRECTOR routes.
//
// The reference video is measured ONCE into a ReferenceStyleProfile (style
// parameters only: pacing, cut density, zoom intensity, speed handling, framing
// mix, typography, transitions, colour, audio and hero-shot structure). The
// profile then drives the Cinematic Director. No timestamps, shot order, frames,
// logos or watermarks from the reference are ever imported into the output.
//
// `/api/analyze-reference` keeps its exact legacy response contract (it is
// still callable with just a title) and now upgrades to a REAL local analysis
// when a reference file is supplied.
// ---------------------------------------------------------------------------
app.post('/api/reference-style/analyze', uploadMiddleware, async (req: any, res: any) => {
  try {
    const files = Array.isArray(req.files) ? req.files : [];
    const uploaded = files.find((f: any) => /video|mp4|quicktime|matroska|mpeg/.test(String(f?.mimetype || ''))) || files[0];
    const localPath =
      uploaded?.path ||
      (typeof req.body?.localPath === 'string' && req.body.localPath ? req.body.localPath : undefined) ||
      (typeof req.body?.referenceLocalPath === 'string' ? req.body.referenceLocalPath : undefined);

    if (!localPath || !fs.existsSync(String(localPath))) {
      return res.status(400).json({
        success: false,
        error: 'A reference video file (or a valid server-side localPath) is required.',
      });
    }

    currentRenderProgress = { percent: 4, stage: 'Measuring the reference video style...' };
    const profile = await referenceStyleService.analyseReference(String(localPath));
    if (!profile) {
      return res.status(503).json({
        success: false,
        error: 'Reference style analysis is unavailable (disabled or unreadable reference file).',
      });
    }
    return res.json({
      success: true,
      analysisMode: 'measured-local',
      referenceLocalPath: String(localPath),
      styleProfile: toLegacyStyleProfile(profile),
      referenceStyleProfile: profile,
    });
  } catch (err: any) {
    console.error('[REFERENCE STYLE] analysis failed:', err?.message || err);
    return res.status(500).json({ success: false, error: err?.message || 'Reference style analysis failed.' });
  }
});

app.post('/api/analyze-reference', uploadMiddleware, async (req: any, res: any) => {
  const { referenceTitle } = req.body || {};
  const files = Array.isArray(req.files) ? req.files : [];
  const uploaded = files.find((f: any) => /video|mp4|quicktime|matroska|mpeg/.test(String(f?.mimetype || ''))) || files[0];
  const localPath = uploaded?.path || (typeof req.body?.localPath === 'string' && req.body.localPath ? req.body.localPath : undefined);

  // Real, measured analysis when a reference file is available. Never throws.
  if (localPath && fs.existsSync(String(localPath))) {
    const profile = await referenceStyleService.analyseReference(String(localPath));
    if (profile) {
      return res.json({
        success: true,
        analysisMode: 'measured-local',
        referenceLocalPath: String(localPath),
        styleProfile: toLegacyStyleProfile(profile),
        referenceStyleProfile: profile,
      });
    }
  }

  // Legacy deterministic profile (unchanged contract for title-only callers).
  res.json({
    success: true,
    analysisMode: 'local',
    styleProfile: {
      average_shot_duration: 1.8,
      zoom_intensity: 0.78,
      transition_frequency: 0.16,
      slow_motion_frequency: 0.3,
      text_frequency: 0.88,
      color_style: 'dark cinematic stadium green with controlled contrast and warm skin highlights',
      energy_curve: 'strong-hook / tension / escalation / climax / emotional outro',
      recommended_bpm: 126,
      cinematography_notes: `Tight 9:16 crops, close-ups, football details, restrained flash impacts, small white editorial captions, hard-cut rhythm. (profile: ${String(referenceTitle || 'Reference Cinematic Football Reel').slice(0, 80)})`,
    },
  });
});

// Direct Cinematic Director endpoint: build an Edit Plan V2 without a render.
app.post('/api/cinematic-director/plan', async (req, res) => {
  try {
    const { duration, motion = null, trackingPath = null, eventsPath = null, referenceStyle = null, mode = 'STANDARD', subject = 'Main player' } = req.body || {};
    const dur = Number(duration);
    if (!Number.isFinite(dur) || dur < 1) {
      return res.status(400).json({ success: false, error: 'A valid source duration (seconds) is required.' });
    }
    const result = await referenceStyleService.buildEditPlan({
      duration: dur,
      motion,
      trackingPath: typeof trackingPath === 'string' ? trackingPath : undefined,
      eventsPath: typeof eventsPath === 'string' ? eventsPath : undefined,
      referenceStyle,
      mode,
      subject,
    });
    return res.json({ success: result.applied, ...result });
  } catch (err: any) {
    return res.status(500).json({ success: false, error: err?.message || 'Cinematic Director failed.' });
  }
});

// ---------------------------------------------------------------------------
// PSYCHOLOGICAL STORYTELLER — status + orchestration.
// The status probe is read-only (no model load, no network). The orchestrate
// route runs the OPTIONAL layer against a supplied plan + real evidence and
// returns the ADDITIVELY-enriched plan plus the story bundle. Both are safe to
// poll and never throw (they always return JSON).
// ---------------------------------------------------------------------------
app.get('/api/psychology/status', (req, res) => {
  res.json({ success: true, ...storytellerService.status() });
});

app.post('/api/psychology/orchestrate', async (req, res) => {
  try {
    const { localPath, editPlan, duration, artifacts = {} } = req.body || {};
    if (!editPlan || !Array.isArray(editPlan.timeline)) {
      return res.status(400).json({ success: false, error: 'An editPlan with a timeline is required.' });
    }
    let dur = Number(duration) || Number(editPlan.duration) || 64;
    if (localPath && fs.existsSync(String(localPath))) {
      try {
        const probe = await execPromise(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${JSON.stringify(localPath)}`);
        dur = Number.parseFloat(probe.stdout.trim()) || dur;
      } catch { /* keep dur */ }
    }
    const resolved = storytellerService.resolveArtifacts(editPlan, {
      trackingPath: typeof artifacts.trackingPath === 'string' ? artifacts.trackingPath : undefined,
      eventsPath: typeof artifacts.eventsPath === 'string' ? artifacts.eventsPath : undefined,
      directorPath: typeof artifacts.directorPath === 'string' ? artifacts.directorPath : undefined,
      motionPath: typeof artifacts.motionPath === 'string' ? artifacts.motionPath : undefined,
    });
    const started = Date.now();
    const result = await storytellerService.enrich(
      typeof localPath === 'string' && fs.existsSync(localPath) ? localPath : '',
      dur, editPlan, resolved);
    return res.json({
      success: true,
      applied: result.applied,
      message: result.message || null,
      durationMs: Date.now() - started,
      metrics: result.metrics || null,
      duel: result.bundle?.duel || null,
      story_arc: result.bundle?.story_arc || [],
      story_script: Array.isArray(editPlan?.story_script) ? editPlan.story_script : [],
      editPlan,
    });
  } catch (err: any) {
    return res.status(500).json({ success: false, error: err?.message || 'Psychological orchestration failed.' });
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
  // Map the OPTIONAL provider keys onto the pre-existing optional layers BEFORE
  // anything spawns a subprocess, then report CONFIGURATION only (never keys).
  applyProviderEnvironment();

  await verifyFFmpegBinaries();

  console.log(`[BOOT] Optional AI providers -> OpenRouter: ${providerConfigured('openrouter') ? 'configured' : 'not configured'} | DeepSeek: ${providerConfigured('deepseek') ? 'configured' : 'not configured'}`);
  if (process.env.VLM_API_KEY) {
    console.log(`[BOOT] Optional VLM layer -> provider=${process.env.VLM_PROVIDER || 'qwen2-vl'} model=${process.env.VLM_MODEL || 'n/a'} base=${process.env.VLM_BASE_URL || 'default'}`);
  }

  // Storage diagnostics: makes the persistence mode obvious in Render logs.
  console.log(`[BOOT] Storage driver: ${storage.driver} | media dir: ${storage.mediaDir}`);
  if (storage.driver === 'local' && storage.mediaDir === path.resolve('public/videos')) {
    console.warn('[BOOT] Rendering to the default ephemeral folder. Attach a Render Disk (set PUBLIC_DIR) or use STORAGE_DRIVER=s3 for persistence.');
  }

  // Video analysis is now 100% LOCAL (OpenCV motion). External API keys are
  // optional and no longer required for a render to succeed.
  const keyStatus = geminiRotator.status();
  console.log(`[BOOT] Video analysis: LOCAL motion (no external API required). Optional legacy key pool: ${keyStatus.totalKeys} key(s).`);

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
