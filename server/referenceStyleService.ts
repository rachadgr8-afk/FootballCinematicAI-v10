// ============================================================================
// REFERENCE STYLE + CINEMATIC DIRECTOR BRIDGE
// ----------------------------------------------------------------------------
// Two thin, fault-tolerant TypeScript bridges over the standalone Python tools:
//
//   yolo/reference_style_analyzer.py -> ReferenceStyleProfile (style parameters)
//   yolo/cinematic_director.py       -> Edit Plan V2 (style-constrained)
//
// Design rules (identical philosophy to server/cinematicEngine.ts):
//   * NEVER throws into the render pipeline. On any failure the caller gets
//     `null` and keeps its existing artefact/plan, so production behaviour is
//     unchanged when the feature is off or unavailable.
//   * The reference video is used as a STYLE SOURCE ONLY. Nothing here copies
//     its frames, timestamps, shot order, logos or watermarks into the output.
//   * Heavy OpenCV work stays in a subprocess (never on the Node event loop).
//   * No new render system and no FFmpeg call here.
// ============================================================================

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';

export type CinematicMode = 'STANDARD' | 'PRO' | 'REFERENCE';

export interface DirectorResult {
  applied: boolean;
  mode: CinematicMode;
  plan?: any;
  message?: string;
  clips?: number;
  heroMoment?: any;
  evidenceLevel?: string;
}

interface RunOptions {
  deadlineMs?: number;
}

class ReferenceStyleService {
  private readonly baseDir: string;
  /** ReferenceStyleProfile cache, keyed by file path + size + mtime. */
  private readonly profileCache = new Map<string, any>();

  constructor() {
    let base = process.cwd();
    try {
      base = path.dirname(fileURLToPath(import.meta.url));
    } catch {
      base = process.cwd();
    }
    this.baseDir = base;
  }

  private resolveScript(name: string): string {
    const candidates = [
      path.resolve(this.baseDir, '..', 'yolo', name), // dev: <root>/server -> <root>/yolo
      path.resolve(this.baseDir, 'yolo', name),       // bundled: /app/yolo
      path.resolve(process.cwd(), 'yolo', name),
      path.resolve(process.cwd(), name),
    ];
    return candidates.find((p) => fs.existsSync(p)) || candidates[0];
  }

  public get enabled(): boolean {
    return process.env.REFERENCE_STYLE_ENABLED !== 'false';
  }

  public get directorEnabled(): boolean {
    return process.env.CINEMATIC_DIRECTOR_ENABLED !== 'false';
  }

  private get pythonBin(): string {
    return process.env.PYTHON_BIN || 'python3';
  }

  /** Run a Python CLI and parse its strict-JSON stdout (last line). */
  private run(script: string, args: string[], opts: RunOptions = {}): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!fs.existsSync(script)) return reject(new Error(`${path.basename(script)} not found`));
      const child = spawn(this.pythonBin, [script, ...args], {
        env: { ...process.env },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      const timer = opts.deadlineMs
        ? setTimeout(() => {
            try { child.kill('SIGKILL'); } catch {}
          }, opts.deadlineMs)
        : null;
      child.stdout.on('data', (d) => { stdout += d.toString(); });
      child.stderr.on('data', (d) => { stderr += d.toString(); });
      child.on('error', (e) => { if (timer) clearTimeout(timer); reject(e); });
      child.on('close', (code) => {
        if (timer) clearTimeout(timer);
        const text = stdout.trim();
        let parsed: any = null;
        try {
          parsed = text ? JSON.parse(text.split('\n').pop() as string) : null;
        } catch {
          parsed = null;
        }
        if (parsed && parsed.success) return resolve(parsed);
        reject(new Error(parsed?.error || stderr.trim().split('\n').slice(-1)[0] || `exit ${code}`));
      });
    });
  }

  private cacheKey(file: string): string {
    try {
      const st = fs.statSync(file);
      return `${file}|${st.size}|${Math.round(st.mtimeMs)}`;
    } catch {
      return file;
    }
  }

  /**
   * Measure a reference video and return its ReferenceStyleProfile.
   * Returns null (never throws) when disabled/unavailable/unreadable.
   */
  public async analyseReference(inputPath: string): Promise<any | null> {
    if (!this.enabled) return null;
    if (!inputPath || !fs.existsSync(inputPath)) return null;
    const key = this.cacheKey(inputPath);
    const cached = this.profileCache.get(key);
    if (cached) return cached;
    const out = path.join(os.tmpdir(), `refstyle_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.json`);
    try {
      await this.run(this.resolveScript('reference_style_analyzer.py'),
        ['--input', inputPath, '--output', out],
        { deadlineMs: Number(process.env.REFERENCE_STYLE_DEADLINE_MS || 600000) });
      const profile = JSON.parse(fs.readFileSync(out, 'utf8'));
      if (this.profileCache.size > 24) this.profileCache.clear();
      this.profileCache.set(key, profile);
      return profile;
    } catch (err: any) {
      console.warn('[ReferenceStyle] analysis skipped:', err?.message || err);
      return null;
    } finally {
      try { fs.rmSync(out, { force: true }); } catch {}
    }
  }

  /**
   * Build an Edit Plan V2 via the Cinematic Director.
   *
   * Every input except `duration` is optional: with no motion analysis the
   * director still produces a valid, source-grounded 64s plan; with real
   * tracking/events it becomes fully player- and event-driven.
   */
  public async buildEditPlan(params: {
    duration: number;
    motion?: any;
    eventsPath?: string;
    trackingPath?: string;
    referenceStyle?: any;
    referenceLocalPath?: string;
    mode?: CinematicMode | string;
    subject?: string;
  }): Promise<DirectorResult> {
    const mode = (String(params.mode || 'STANDARD').toUpperCase() as CinematicMode);
    const resolvedMode: CinematicMode = ['STANDARD', 'PRO', 'REFERENCE'].includes(mode) ? mode : 'STANDARD';
    if (!this.directorEnabled) return { applied: false, mode: resolvedMode, message: 'CINEMATIC_DIRECTOR_ENABLED is false' };
    try {
      const stamp = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      let motionPath: string | undefined;
      if (params.motion && typeof params.motion === 'object') {
        motionPath = path.join(os.tmpdir(), `motion_${stamp}.json`);
        fs.writeFileSync(motionPath, JSON.stringify(params.motion));
      }

      // REFERENCE mode: measure the reference clip when the caller only handed
      // us a file (a client-supplied profile still wins).
      let reference = params.referenceStyle && typeof params.referenceStyle === 'object' ? params.referenceStyle : null;
      if (!reference && resolvedMode === 'REFERENCE' && params.referenceLocalPath) {
        reference = await this.analyseReference(params.referenceLocalPath);
      }
      let referencePath: string | undefined;
      if (reference) {
        referencePath = path.join(os.tmpdir(), `refstyle_in_${stamp}.json`);
        fs.writeFileSync(referencePath, JSON.stringify(reference));
      }

      const out = path.join(os.tmpdir(), `plan_${stamp}.json`);
      const args = [
        '--duration', String(params.duration),
        '--output', out,
        '--mode', resolvedMode,
      ];
      if (motionPath) args.push('--motion', motionPath);
      if (referencePath) args.push('--reference-style', referencePath);
      if (params.eventsPath && fs.existsSync(params.eventsPath)) args.push('--events', params.eventsPath);
      if (params.trackingPath && fs.existsSync(params.trackingPath)) args.push('--tracking', params.trackingPath);
      if (params.subject) args.push('--subject', String(params.subject));

      try {
        await this.run(this.resolveScript('cinematic_director.py'), args, { deadlineMs: 120000 });
      } finally {
        for (const f of [motionPath, referencePath]) {
          if (f) { try { fs.rmSync(f, { force: true }); } catch {} }
        }
      }
      const plan = JSON.parse(fs.readFileSync(out, 'utf8'));
      try { fs.rmSync(out, { force: true }); } catch {}
      const ok = Boolean(plan?.validation?.ok);
      return {
        applied: ok,
        mode: resolvedMode,
        plan,
        clips: Array.isArray(plan?.timeline) ? plan.timeline.length : 0,
        heroMoment: plan?.cinematic_director?.hero_moment,
        evidenceLevel: plan?.cinematic_director?.evidence_level,
        message: ok ? undefined : `director validation failed: ${JSON.stringify(plan?.validation?.problems || [])}`,
      };
    } catch (err: any) {
      return { applied: false, mode: resolvedMode, message: err?.message || String(err) };
    }
  }
}

export const referenceStyleService = new ReferenceStyleService();
