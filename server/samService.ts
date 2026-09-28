// ============================================================================
// OPTIONAL SAM SEGMENTATION / TRACKING BRIDGE
// ----------------------------------------------------------------------------
// Thin, fault-tolerant TypeScript wrapper around the standalone Python engine
// `python/sam_engine.py` (Meta SAM 3.1 segmentation + subject tracking).
//
// This is an ENHANCEMENT layer that plugs into the EXISTING render pipeline:
//
//   Football Analysis -> Event Detection -> Interesting Segment
//       -> [ OPTIONAL SAM : segmentation + tracking ]   <-- this bridge
//       -> Edit Plan -> CinematicEngine -> QC -> done
//
// Design rules (mirrors server/cinematicEngine.ts and yolo/rife_interpolate.py):
//   * DEFAULT OFF — inert unless SAM_ENABLED=true. When disabled, unavailable,
//     or failing, `enhanceRenderPlan()` returns { applied:false } and the caller
//     keeps the existing YOLO/tracking + CinematicEngine path byte-for-byte.
//   * NEVER throws into the render pipeline.
//   * NO new job system and NO new route: the existing renderJob/progress/result
//     contract is reused verbatim.
//   * The heavy work runs in a SEPARATE interpreter (SAM_PYTHON_BIN) whose site
//     packages are isolated from the main stack (see PYTHONPATH handling below),
//     so the pinned torch 2.6.0+cpu / numpy<2 / opencv-headless 4.10 stack is
//     never modified for SAM.
//   * No API keys / secrets are read or logged. Model weights come from a local
//     path (SAM_MODEL_PATH), never from the network.
// ============================================================================

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// ---------------------------------------------------------------------------
// Normalised result contract (compact — masks are REFERENCES, never inlined)
// ---------------------------------------------------------------------------
export interface SamMaskRef {
  maskRef?: string | null;
  rle?: number[];
  shape?: [number, number];
  class?: string;
  bbox?: number[];
  score?: number;
}

export interface SamFrameRef {
  frame: number;
  bbox: number[];
  score: number;
  maskRef?: string | null;
}

/** Canonical per-object shape requested by the pipeline contract. */
export interface SamObject {
  id: string;
  class: string;
  kind: string;
  trackId: string;
  confidence: number;
  bbox: number[];          // normalised [x1,y1,x2,y2] of the object's keyframe
  startFrame: number;
  endFrame: number;
  frames: SamFrameRef[];
  maskRefs: string[];
  keyframeMaskRef?: string | null;
  maskQuality: number;
  framesSampled: number;
  backend?: string;
}

export interface SamSegment {
  index: number;
  start: number;
  end: number;
  startFrame: number;
  endFrame: number;
  reason: string;
  needsIsolation: boolean;
  shotType?: string;
  beatRole?: string;
  targetKinds: string[];
  objects: SamObject[];
  framesSampled: number;
  keyframeMaskRef?: string | null;
  warnings?: string[];
}

export interface SamRawResult {
  success: boolean;
  applied: boolean;
  version?: string;
  backend?: string | null;
  error?: string;
  message?: string;
  video?: Record<string, any>;
  segmentsProcessed?: number;
  objectsTracked?: number;
  framesSampled?: number;
  segments?: SamSegment[];
  warnings?: string[];
  cache?: { hits: number; misses: number };
  elapsedSeconds?: number;
}

/** Compact spec stored inside the Edit Plan (references only, no raw masks). */
export interface SamEnhancementSpec {
  applied: boolean;
  version: string;
  backend: string | null;
  reason?: string;
  duration?: number;
  masks: Array<{
    id: string;
    segmentIndex: number;
    class: string;
    kind: string;
    trackId: string;
    maskRef: string | null;
    shape?: [number, number];
    bbox?: number[];
    framesCount: number;
    quality: number;
  }>;
  segments: Array<{
    index: number;
    timelineIndex?: number;
    start: number;
    end: number;
    reason: string;
    effect: string;
    objectIds: string[];
    keyframeMaskRef?: string | null;
  }>;
}

export interface SamProgress {
  percent: number;
  stage: string;
}

export interface SamStatus {
  enabled: boolean;
  available: boolean;
  version: string;
  adapter: string;
  modelPathSet: boolean;
  metaSamAvailable: boolean;
  metaBackend: string | null;
  pythonBin: string;
  isolated: boolean;
  reason?: string | null;
}

/** Reason -> editor-safe isolation effect. Keep in sync with config/sam.json. */
const EFFECT_BY_REASON: Record<string, string> = {
  goal: 'background_dimming',
  shot: 'subject_isolation',
  save: 'background_blur',
  tackle: 'player_highlight',
  dribble: 'selective_sharpening',
  celebration: 'cinematic_vignette_focus',
};

const STOP_WORDS = /ghp_|sk-|AIza|xi-api-key|Bearer /g;

function redact(text: unknown, limit = 300): string {
  return String(text).replace(STOP_WORDS, '***').slice(0, limit);
}

class SamServiceBridge {
  private readonly scriptPath: string;
  /** Small bounded cache of full analysis results keyed by video + plan hash. */
  private readonly resultCache = new Map<string, SamEnhancementSpec>();
  private readonly MAX_RESULT_CACHE = 8;
  private lastStatus: { at: number; value: SamStatus } | null = null;

  constructor() {
    let baseDir = process.cwd();
    try {
      baseDir = path.dirname(fileURLToPath(import.meta.url));
    } catch {
      baseDir = process.cwd();
    }
    const candidates = [
      path.resolve(baseDir, '..', 'python', 'sam_engine.py'), // dev: <root>/server -> <root>/python
      path.resolve(baseDir, 'python', 'sam_engine.py'),        // bundled: /app/python
      path.resolve(process.cwd(), 'python', 'sam_engine.py'),
    ];
    this.scriptPath = candidates.find((p) => fs.existsSync(p)) || candidates[0];
  }

  /** True when the operator explicitly enabled SAM. Default: OFF. */
  public get enabled(): boolean {
    return process.env.SAM_ENABLED === 'true';
  }

  private get pythonBin(): string {
    return process.env.SAM_PYTHON_BIN || process.env.PYTHON_BIN || 'python3';
  }

  private get cacheDir(): string {
    return process.env.SAM_CACHE_DIR
      || path.resolve(process.env.TMP_WORK_DIR || '/tmp/football_engine/work', 'sam_cache');
  }

  /**
   * When SAM runs in its own venv, `-S` would break it; instead we simply do
   * NOT inject the main interpreter's site-packages. If SAM_PYTHON_BIN is unset
   * we fall back to the project interpreter and warn the operator (isolation is
   * recommended but never silently blocked).
   */
  private buildEnv(): NodeJS.ProcessEnv {
    const env: NodeJS.ProcessEnv = { ...process.env };
    const isolated = Boolean(process.env.SAM_PYTHON_BIN);
    if (isolated) {
      // Ensure the isolated interpreter does NOT pull the main venv's packages.
      delete env.PYTHONPATH;
    }
    return env;
  }

  /** Spawn the engine CLI, parse the final JSON line from stdout. */
  private run(args: string[], timeoutMs: number): Promise<SamRawResult> {
    return new Promise((resolve, reject) => {
      if (!fs.existsSync(this.scriptPath)) {
        return reject(new Error(`sam_engine.py not found at ${this.scriptPath}`));
      }
      const child = spawn(this.pythonBin, [this.scriptPath, ...args], {
        env: this.buildEnv(),
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      let settled = false;
      const finish = (fn: () => void) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        fn();
      };
      const timer = setTimeout(() => {
        try { child.kill('SIGKILL'); } catch { /* ignore */ }
        finish(() => reject(new Error(`sam engine timed out after ${Math.round(timeoutMs / 1000)}s`)));
      }, timeoutMs);

      child.stdout.on('data', (d) => { stdout += d.toString(); });
      child.stderr.on('data', (d) => { stderr += d.toString(); });
      child.on('error', (err) => finish(() => reject(err)));
      child.on('close', () => {
        finish(() => {
          const text = stdout.trim();
          try {
            const parsed = (text ? JSON.parse(text.split('\n').filter(Boolean).pop() as string) : null) as SamRawResult;
            if (parsed) resolve(parsed);
            else reject(new Error(`sam engine returned no JSON: ${redact(stderr)}`));
          } catch {
            reject(new Error(`sam engine returned invalid JSON: ${redact(stdout)}`));
          }
        });
      });
    });
  }

  /** Readiness probe (no model load, no network). Cached for 60s. */
  public async status(force = false): Promise<SamStatus> {
    const base: SamStatus = {
      enabled: this.enabled,
      available: false,
      version: 'unknown',
      adapter: process.env.SAM_ADAPTER || 'auto',
      modelPathSet: Boolean(process.env.SAM_MODEL_PATH),
      metaSamAvailable: false,
      metaBackend: null,
      pythonBin: this.pythonBin,
      isolated: Boolean(process.env.SAM_PYTHON_BIN),
      reason: null,
    };
    if (!force && this.lastStatus && Date.now() - this.lastStatus.at < 60_000) {
      return { ...base, ...this.lastStatus.value, enabled: this.enabled };
    }
    try {
      const res: any = await this.run(['check'], Number(process.env.SAM_STATUS_TIMEOUT_MS || 20000));
      const value: SamStatus = {
        ...base,
        available: true,
        version: res?.version || 'unknown',
        adapter: res?.adapter || base.adapter,
        modelPathSet: Boolean(res?.modelPathSet),
        metaSamAvailable: Boolean(res?.metaSamAvailable),
        metaBackend: res?.metaBackend ?? null,
        reason: res?.metaSamReason ?? null,
      };
      this.lastStatus = { at: Date.now(), value };
      return value;
    } catch (err: any) {
      const value: SamStatus = { ...base, reason: redact(err?.message || err, 160) };
      this.lastStatus = { at: Date.now(), value };
      return value;
    }
  }

  /**
   * Attach SAM isolation effects to the matching timeline clips (references
   * only). Returns the number of clips annotated with a 'sam' block.
   */
  private attachToPlan(editPlan: any, spec: SamEnhancementSpec): number {
    const timeline = Array.isArray(editPlan?.timeline) ? editPlan.timeline : [];
    let annotated = 0;
    for (const seg of spec.segments) {
      const clip = timeline.find((c: any) =>
        typeof seg.timelineIndex === 'number'
          ? Number(c.timeline_index) === Number(seg.timelineIndex)
          : (Number(c.source_start) >= seg.start - 0.6 && Number(c.source_end) <= seg.end + 0.6));
      if (!clip) continue;
      const objects = spec.masks.filter((m) => seg.objectIds.includes(m.id)).map((m) => ({
        id: m.id,
        class: m.class,
        kind: m.kind,
        trackId: m.trackId,
        bbox: m.bbox,
        startFrame: 0,
        endFrame: 0,
        frames: [],
      }));
      clip.sam = {
        applied: true,
        reason: seg.reason,
        effect: seg.effect,
        maskRef: seg.keyframeMaskRef || null,
        objectIds: seg.objectIds,
        objects,
        version: spec.version,
        backend: spec.backend,
      };
      annotated += 1;
    }
    return annotated;
  }

  /** Build the compact, mask-reference-only spec embedded in the Edit Plan. */
  private normalize(result: SamRawResult): SamEnhancementSpec {
    const masks: SamEnhancementSpec['masks'] = [];
    const segments: SamEnhancementSpec['segments'] = [];
    for (const seg of result.segments || []) {
      const objectIds: string[] = [];
      for (const obj of seg.objects || []) {
        const firstMask = obj.frames?.find((f) => f.maskRef)?.maskRef || obj.keyframeMaskRef || null;
        masks.push({
          id: obj.id,
          segmentIndex: seg.index,
          class: obj.class,
          kind: obj.kind,
          trackId: obj.trackId,
          maskRef: firstMask,
          shape: undefined,
          bbox: obj.frames?.[0]?.bbox || undefined,
          framesCount: obj.frames?.length || 0,
          quality: obj.maskQuality ?? 0,
        });
        objectIds.push(obj.id);
      }
      segments.push({
        index: seg.index,
        start: seg.start,
        end: seg.end,
        reason: seg.reason,
        effect: EFFECT_BY_REASON[seg.reason] || 'subject_isolation',
        objectIds,
        keyframeMaskRef: seg.keyframeMaskRef || null,
      });
    }
    return {
      applied: masks.length > 0,
      version: result.version || 'sam-1.0.0',
      backend: result.backend ?? null,
      reason: result.message,
      duration: result.video?.duration,
      masks,
      segments,
    };
  }

  /**
   * MAIN ENTRY. Feature-gated, never-throwing. Runs the optional SAM pass on the
   * INTERESTING SEGMENTS only and attaches the compact spec to the edit plan.
   */
  public async enhanceRenderPlan(
    inputPath: string,
    editPlan: any,
    artifacts: { trackingPath?: string; eventsPath?: string },
    onProgress?: (p: SamProgress) => void,
  ): Promise<{ applied: boolean; spec: SamEnhancementSpec }> {
    const empty: SamEnhancementSpec = { applied: false, version: 'sam-1.0.0', backend: null, masks: [], segments: [] };
    if (!this.enabled) return { applied: false, spec: empty };
    if (!inputPath || !fs.existsSync(inputPath)) return { applied: false, spec: empty };
    if (!fs.existsSync(this.scriptPath)) {
      console.warn(`[SAM] engine missing at ${this.scriptPath}; falling back to YOLO/tracking.`);
      return { applied: false, spec: empty };
    }

    const outPath = path.join(this.cacheDir, `sam_result_${Date.now()}.json`);
    try { fs.mkdirSync(this.cacheDir, { recursive: true }); } catch { /* ignore */ }

    const args = [
      'run',
      '--video', inputPath,
      '--plan', this.writeJson('sam_plan', editPlan),
      '--output', outPath,
      '--cache-dir', this.cacheDir,
      '--mask-mode', process.env.SAM_MASK_MODE || 'rle',
      '--stride', process.env.SAM_STRIDE || '4',
      '--max-segments', process.env.SAM_MAX_SEGMENTS || '8',
      '--max-frames-per-segment', process.env.SAM_MAX_FRAMES_PER_SEGMENT || '6',
      '--max-objects-per-segment', process.env.SAM_MAX_OBJECTS_PER_SEGMENT || '3',
      '--deadline-seconds', process.env.SAM_DEADLINE_SECONDS || '120',
      '--mask-preview-dir', path.join(this.cacheDir, 'previews'),
    ];
    if (artifacts.trackingPath && fs.existsSync(artifacts.trackingPath)) {
      args.push('--detections', artifacts.trackingPath);
    }
    if (artifacts.eventsPath && fs.existsSync(artifacts.eventsPath)) {
      args.push('--events', artifacts.eventsPath);
    }

    onProgress?.({ percent: 13, stage: 'Segmenting key players...' });
    try {
      const result = await this.run(args, Number(process.env.SAM_TIMEOUT_MS || 240000));
      if (!result?.success || !result.applied) {
        console.warn(`[SAM] pass not applied: ${redact(result?.error || result?.message || 'no reason')}; using YOLO/tracking.`);
        return { applied: false, spec: empty };
      }
      const spec = this.normalize(result);
      if (!spec.applied) {
        console.log('[SAM] no interesting segment required subject isolation.');
        return { applied: false, spec };
      }
      const annotated = this.attachToPlan(editPlan, spec);
      (editPlan as any).samSegmentation = spec; // compact, references only
      this.remember(spec, inputPath);
      onProgress?.({ percent: 18, stage: 'Tracking subjects...' });
      console.log(`[SAM] applied backend=${spec.backend} segments=${spec.segments.length} objects=${spec.masks.length} clipsAnnotated=${annotated}`);
      return { applied: true, spec };
    } catch (err: any) {
      console.warn(`[SAM] pass skipped: ${redact(err?.message || err)}`);
      return { applied: false, spec: empty };
    } finally {
      try { if (fs.existsSync(outPath)) fs.unlinkSync(outPath); } catch { /* ignore */ }
    }
  }

  private writeJson(prefix: string, data: any): string {
    try {
      fs.mkdirSync(this.cacheDir, { recursive: true });
      const p = path.join(this.cacheDir, `${prefix}_${Date.now()}.json`);
      fs.writeFileSync(p, JSON.stringify(data), 'utf8');
      return p;
    } catch {
      return '';
    }
  }

  private remember(spec: SamEnhancementSpec, inputPath: string): void {
    try {
      const key = `${path.basename(inputPath)}:${spec.masks.length}:${spec.segments.length}`;
      this.resultCache.set(key, spec);
      if (this.resultCache.size > this.MAX_RESULT_CACHE) {
        const oldest = this.resultCache.keys().next().value as string;
        this.resultCache.delete(oldest);
      }
    } catch { /* ignore */ }
  }
}

export const samService = new SamServiceBridge();
