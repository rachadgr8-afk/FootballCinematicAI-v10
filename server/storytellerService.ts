// ============================================================================
// PSYCHOLOGICAL STORYTELLER BRIDGE  (optional layer)
// ----------------------------------------------------------------------------
// Thin, fault-tolerant TypeScript wrappers over three standalone Python tools:
//
//   yolo/cinematic_storyteller.py  -> duel detection + story arc + plan enrich
//   yolo/deepseek_storyteller.py   -> exactly 6 inner-monologue lines (TEXT only)
//   python/depth_engine.py         -> optional Depth-Anything-V2 depth mask
//
// Design rules (identical philosophy to server/cinematicEngine.ts and
// server/referenceStyleService.ts):
//   * NEVER throws into the render pipeline. On any failure the caller gets
//     `{ applied:false }` and keeps its existing plan/artifact untouched.
//   * Strictly ADDITIVE: the enriched plan only gains OPTIONAL fields, so an old
//     renderer/frontend keeps working byte-for-byte when the layer is off.
//   * FEATURE-GATED: inert unless the render requests it (PSYCHOLOGICAL mode) or
//     PSYCHOLOGY_ENABLED=true. Default OFF.
//   * Heavy OpenCV/model work stays in a subprocess (never on the Node event loop).
//   * NO secrets in code/logs: keys come from the environment, never echoed.
// ============================================================================

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';

export interface StorytellerResult {
  applied: boolean;
  message?: string;
  bundle?: any;
  duelDetected?: boolean;
  duelConfidence?: number;
  winner?: string | null;
  storyScript?: any[];
  metrics?: Record<string, any>;
}

function runScript(script: string, args: string[], deadlineMs: number): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(script)) return reject(new Error(`${path.basename(script)} not found`));
    const pythonBin = process.env.PYTHON_BIN || 'python3';
    const child = spawn(pythonBin, [script, ...args], {
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, deadlineMs);
    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('error', (e) => { clearTimeout(timer); reject(e); });
    child.on('close', (code) => {
      clearTimeout(timer);
      const text = stdout.trim();
      let parsed: any = null;
      try { parsed = text ? JSON.parse(text.split('\n').filter(Boolean).pop() as string) : null; } catch { parsed = null; }
      if (code === 0 && parsed && parsed.success) return resolve(parsed);
      reject(new Error(parsed?.error || parsed?.reason || stderr.trim().split('\n').slice(-1)[0] || `exit ${code}`));
    });
  });
}

class StorytellerService {
  private readonly baseDir: string;

  constructor() {
    let base = process.cwd();
    try { base = path.dirname(fileURLToPath(import.meta.url)); } catch { base = process.cwd(); }
    this.baseDir = base;
  }

  private resolveYolo(name: string): string {
    const candidates = [
      path.resolve(this.baseDir, '..', 'yolo', name),
      path.resolve(this.baseDir, 'yolo', name),
      path.resolve(process.cwd(), 'yolo', name),
      path.resolve(process.cwd(), name),
    ];
    return candidates.find((p) => fs.existsSync(p)) || candidates[0];
  }

  private resolvePython(name: string): string {
    const candidates = [
      path.resolve(this.baseDir, '..', 'python', name),
      path.resolve(this.baseDir, 'python', name),
      path.resolve(process.cwd(), 'python', name),
      path.resolve(process.cwd(), name),
    ];
    return candidates.find((p) => fs.existsSync(p)) || candidates[0];
  }

  /** True when the layer may run for this render request. */
  public wants(mode?: string, plan?: any, explicit?: boolean): boolean {
    if (explicit === true) return true;
    if (explicit === false) return false;
    const forced = process.env.PSYCHOLOGY_ENABLED === 'true';
    if (forced) return true;
    const m = String(mode || '').toUpperCase();
    if (m.includes('PSYCHOLOGICAL')) return true;
    const styleName = String(plan?.style_name || plan?.styleName || '').toUpperCase();
    if (styleName.includes('PSYCHOLOGICAL')) return true;
    if (plan?.psychology === true) return true;
    return false;
  }

  /** Resolve the real evidence paths already produced by the analysis stage. */
  public resolveArtifacts(plan: any, explicit?: { trackingPath?: string; eventsPath?: string; directorPath?: string; motionPath?: string }) {
    const evidence = plan?.footballDirectorEvidence;
    let eventsPath = explicit?.eventsPath && fs.existsSync(explicit.eventsPath) ? explicit.eventsPath : undefined;
    if (!eventsPath && typeof evidence?.eventsPath === 'string' && fs.existsSync(evidence.eventsPath)) eventsPath = evidence.eventsPath;
    let directorPath = explicit?.directorPath && fs.existsSync(explicit.directorPath) ? explicit.directorPath : undefined;
    if (!directorPath && typeof evidence?.directorPath === 'string' && fs.existsSync(evidence.directorPath)) directorPath = evidence.directorPath;
    let trackingPath = explicit?.trackingPath && fs.existsSync(explicit.trackingPath) ? explicit.trackingPath : undefined;
    if (!trackingPath && eventsPath) {
      const candidate = path.join(path.dirname(path.dirname(eventsPath)), 'tracking.json');
      if (fs.existsSync(candidate)) trackingPath = candidate;
    }
    const motionPath = explicit?.motionPath && fs.existsSync(explicit.motionPath) ? explicit.motionPath : undefined;
    return { trackingPath, eventsPath, directorPath, motionPath };
  }

  /**
   * Run the storyteller: duel detection + optional VLM + optional DeepSeek script,
   * then enrich the plan IN PLACE (additive fields only). Returns the bundle.
   */
  public async enrich(
    inputPath: string,
    duration: number,
    editPlan: any,
    artifacts: { trackingPath?: string; eventsPath?: string; directorPath?: string; motionPath?: string }
  ): Promise<StorytellerResult> {
    try {
      const stamp = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      const outBundle = path.join(os.tmpdir(), `storyteller_${stamp}.json`);
      const planIn = path.join(os.tmpdir(), `storyteller_plan_in_${stamp}.json`);
      const planOut = path.join(os.tmpdir(), `storyteller_plan_out_${stamp}.json`);
      fs.writeFileSync(planIn, JSON.stringify(editPlan));

      const args = [
        '--duration', String(duration),
        '--output', outBundle,
        '--plan', planIn,
        '--plan-out', planOut,
      ];
      if (artifacts.trackingPath) args.push('--tracking', artifacts.trackingPath);
      if (artifacts.eventsPath) args.push('--events', artifacts.eventsPath);
      if (artifacts.directorPath) args.push('--director', artifacts.directorPath);
      if (artifacts.motionPath) args.push('--motion', artifacts.motionPath);
      if (inputPath && fs.existsSync(inputPath)) args.push('--video', inputPath);

      const timeoutMs = Number(process.env.STORYTELLER_TIMEOUT_MS || 180000);
      const res = await runScript(this.resolveYolo('cinematic_storyteller.py'), args, timeoutMs);

      // Copy the additive fields from the enriched plan back onto the live plan.
      if (fs.existsSync(planOut)) {
        try {
          const enriched = JSON.parse(fs.readFileSync(planOut, 'utf8'));
          if (Array.isArray(enriched?.timeline) && Array.isArray(editPlan?.timeline)
              && enriched.timeline.length === editPlan.timeline.length) {
            const ADDITIVE = ['story_role', 'story_arc', 'duel_moment', 'duel_moment_exact',
              'depth_effect', 'low_angle', 'pov_switch', 'psychological_story', 'story_script_position'];
            for (let i = 0; i < enriched.timeline.length; i++) {
              const src = enriched.timeline[i] || {};
              const dst = editPlan.timeline[i];
              for (const key of ADDITIVE) if (src[key] !== undefined) dst[key] = src[key];
              // Only the psychological script may REPLACE the on-screen text (and
              // only when it actually produced a line for this clip).
              if (typeof src.text === 'string' && src.text && src.psychological_story) dst.text = src.text;
            }
          }
          for (const key of ['psychological_story', 'story_arc', 'duel_moment']) {
            if (enriched?.[key] !== undefined) editPlan[key] = enriched[key];
          }
          if (Array.isArray(enriched?.story_script) && enriched.story_script.length) {
            editPlan.story_script = enriched.story_script;
          }
        } catch (e: any) {
          console.warn('[Storyteller] plan merge skipped:', e?.message || e);
        }
      }

      const bundle = fs.existsSync(outBundle) ? JSON.parse(fs.readFileSync(outBundle, 'utf8')) : null;
      return {
        applied: true,
        bundle,
        duelDetected: Boolean(bundle?.duel?.detected),
        duelConfidence: Number(bundle?.duel?.confidence || 0),
        winner: bundle?.duel?.winner ?? null,
        storyScript: Array.isArray(editPlan?.story_script) ? editPlan.story_script : [],
        metrics: buildMetrics(bundle),
      };
    } catch (err: any) {
      console.warn('[Storyteller] skipped:', err?.message || err);
      return { applied: false, message: err?.message || String(err) };
    } finally {
      // temp files are best-effort cleaned by the OS (/tmp); explicit cleans below
    }
  }

  /** Optional depth pass. Returns depth segments to attach onto clips. */
  public async depth(
    inputPath: string,
    segments: Array<{ time: number; span?: number }>,
    maskDir: string
  ): Promise<{ applied: boolean; source?: string; segments?: any[]; message?: string }> {
    if (process.env.DEPTH_ENABLED !== 'true') return { applied: false, message: 'DEPTH_ENABLED is not true' };
    if (!segments || !segments.length) return { applied: false, message: 'no depth segments' };
    try {
      const stamp = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      const segFile = path.join(os.tmpdir(), `depth_segments_${stamp}.json`);
      const outFile = path.join(os.tmpdir(), `depth_out_${stamp}.json`);
      fs.writeFileSync(segFile, JSON.stringify(segments));
      const args = ['run', '--video', inputPath, '--segments', segFile,
        '--output', outFile, '--mask-dir', maskDir];
      if (process.env.DEPTH_PROXY_FALLBACK === 'true') args.push('--proxy');
      const res = await runScript(this.resolvePython('depth_engine.py'), args,
        Number(process.env.DEPTH_TIMEOUT_MS || 180000));
      const parsed = fs.existsSync(outFile) ? JSON.parse(fs.readFileSync(outFile, 'utf8')) : null;
      return { applied: Boolean(parsed?.applied), source: parsed?.source, segments: parsed?.segments || [], message: res?.reason };
    } catch (err: any) {
      console.warn('[Depth] skipped:', err?.message || err);
      return { applied: false, message: err?.message || String(err) };
    }
  }

  public status() {
    return {
      enabled: process.env.PSYCHOLOGY_ENABLED === 'true' || true,
      vlm: {
        configured: Boolean(process.env.VLM_API_KEY),
        provider: process.env.VLM_PROVIDER || 'qwen2-vl',
        model: process.env.VLM_MODEL || null,
      },
      deepseek: {
        configured: Boolean(process.env.DEEPSEEK_API_KEY),
        model: process.env.DEEPSEEK_MODEL || 'deepseek-reasoner',
      },
      depth: {
        enabled: process.env.DEPTH_ENABLED === 'true',
        model: process.env.DEPTH_MODEL_ID || 'depth-anything/Depth-Anything-V2-Small-hf',
        proxyFallback: process.env.DEPTH_PROXY_FALLBACK === 'true',
      },
      rife: {
        // RIFE is a SEPARATE, pre-existing optional layer. /opt/rife is expected
        // to be absent in dev; no psychology feature depends on it.
        available: fs.existsSync(process.env.RIFE_REPO || '/opt/rife'),
      },
    };
  }
}

/** Flatten the bundle into render-result metrics + attach audio cues. */
function buildMetrics(bundle: any): Record<string, any> {
  const duel = bundle?.duel || {};
  return {
    duel_confidence: Number(duel.confidence || 0),
    story_confidence: Number(bundle?.story_confidence ?? duel.confidence ?? 0),
    depth_effect_used: Boolean(bundle?.depth_segments?.length),
    low_angle_used: Boolean(bundle?.low_angle_segments?.length),
    pov_switch_used: Boolean(bundle?.pov_switch_points?.length),
    story_script_count: Array.isArray(bundle?.story_script) ? bundle.story_script.length : 0,
    duel_detected: Boolean(duel.detected),
    duel_source: duel.source || null,
    winner: duel.winner ?? null,
    loser: duel.loser ?? null,
    predator: duel.predator ?? null,
    prey: duel.prey ?? null,
    evidence_level: bundle?.evidence_level || null,
  };
}

export const storytellerService = new StorytellerService();
