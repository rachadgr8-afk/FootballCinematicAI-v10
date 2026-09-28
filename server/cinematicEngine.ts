// ============================================================================
// CINEMATIC ENGINE BRIDGE (OpenCV + NumPy)
// ----------------------------------------------------------------------------
// Thin, fault-tolerant TypeScript wrapper around the standalone Python engine
// `video_engine.py` (class CinematicEngine).
//
// Design rules (mirrors the existing yolo/rife_interpolate.py bridge):
//   * NEVER throws into the render pipeline — failures degrade to
//     { applied: false } and the caller keeps the existing artifact.
//   * Fully feature-gated by the CINEMATIC_ENGINE_ENABLED env var so the
//     production behaviour is byte-for-byte identical unless explicitly turned
//     on. Nothing here runs by default.
//   * Spawns the engine as a subprocess and reads a strict JSON result on
//     stdout, so the heavy OpenCV work stays out of the Node event loop.
//   * No FFmpeg calls here: the engine is OpenCV + NumPy only.
// ============================================================================

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

export interface EngineResult {
  applied: boolean;
  outputPath?: string;
  frames?: number;
  fps?: number;
  width?: number;
  height?: number;
  message?: string;
}

class CinematicEngineBridge {
  /** Absolute path to the Python engine, resolved across dev + bundled layouts. */
  private readonly scriptPath: string;

  constructor() {
    // The engine lives at the repo root as `video_engine.py`.
    //  * ESM dev (tsx): import.meta.url points at <root>/server/cinematicEngine.ts
    //  * bundled (esbuild server.ts -> /app/server.js with imports inlined):
    //    import.meta.url points at /app/server.js
    // We probe the plausible locations and keep the first that exists.
    let baseDir = process.cwd();
    try {
      baseDir = path.dirname(fileURLToPath(import.meta.url));
    } catch {
      baseDir = process.cwd();
    }
    const candidates = [
      path.resolve(baseDir, '..', 'video_engine.py'), // dev: <root>/server -> <root>
      path.resolve(baseDir, 'video_engine.py'),        // bundled: /app/video_engine.py
      path.resolve(process.cwd(), 'video_engine.py'),  // cwd fallback
    ];
    this.scriptPath = candidates.find((p) => fs.existsSync(p)) || candidates[0];
  }

  /** True when the operator explicitly enabled the engine. Default: OFF. */
  public get enabled(): boolean {
    return process.env.CINEMATIC_ENGINE_ENABLED === 'true';
  }

  private get pythonBin(): string {
    return process.env.PYTHON_BIN || 'python3';
  }

  /**
   * Run the engine CLI and parse its JSON stdout. Resolves with the raw parsed
   * object on success; rejects only on spawn/protocol errors (callers wrap this
   * in try/catch and degrade gracefully).
   */
  private run(args: string[]): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!fs.existsSync(this.scriptPath)) {
        return reject(new Error(`video_engine.py not found at ${this.scriptPath}`));
      }
      const child = spawn(this.pythonBin, [this.scriptPath, ...args], {
        env: { ...process.env },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (d) => { stdout += d.toString(); });
      child.stderr.on('data', (d) => { stderr += d.toString(); });
      child.on('error', reject);
      child.on('close', (code) => {
        const text = stdout.trim();
        // The CLI always prints valid JSON (even on internal failure).
        let parsed: any = null;
        try {
          parsed = text ? JSON.parse(text.split('\n').pop() as string) : null;
        } catch {
          parsed = null;
        }
        if (code === 0 && parsed && parsed.success) return resolve(parsed);
        const err = parsed?.error || stderr.trim() || `video_engine exited with ${code}`;
        reject(new Error(err));
      });
    });
  }

  /** Validate that a produced MP4 exists and is not empty. */
  private isValidOutput(outputPath: string): boolean {
    try {
      return fs.existsSync(outputPath) && fs.statSync(outputPath).size > 10000;
    } catch {
      return false;
    }
  }

  /**
   * Smart, segment-scoped slow motion via optical-flow frame interpolation.
   * Only the [segStart, segEnd] window is slowed; the rest is untouched.
   */
  public async slowMotion(
    inputPath: string,
    outputPath: string,
    slowFactor = 0.25,
    segStart = 2.0,
    segEnd = 4.0
  ): Promise<EngineResult> {
    if (!this.enabled) return { applied: false, message: 'CINEMATIC_ENGINE_ENABLED is not true' };
    try {
      const res = await this.run([
        'slow-motion',
        '--input', inputPath,
        '--output', outputPath,
        '--slow-factor', String(slowFactor),
        '--seg-start', String(segStart),
        '--seg-end', String(segEnd),
      ]);
      if (!this.isValidOutput(outputPath)) {
        return { applied: false, message: 'slow-motion produced no valid output' };
      }
      return {
        applied: true,
        outputPath,
        frames: res.frames_written,
        fps: res.fps,
        width: res.width,
        height: res.height,
      };
    } catch (err: any) {
      return { applied: false, message: err?.message || String(err) };
    }
  }

  /**
   * Cinematic LUT (CLAHE + Teal&Orange) applied frame-by-frame. `intensity`
   * is the blend weight in [0, 1].
   */
  public async cinematicLut(
    inputPath: string,
    outputPath: string,
    intensity = 0.7
  ): Promise<EngineResult> {
    if (!this.enabled) return { applied: false, message: 'CINEMATIC_ENGINE_ENABLED is not true' };
    try {
      const res = await this.run([
        'lut',
        '--input', inputPath,
        '--output', outputPath,
        '--intensity', String(intensity),
      ]);
      if (!this.isValidOutput(outputPath)) {
        return { applied: false, message: 'lut produced no valid output' };
      }
      return {
        applied: true,
        outputPath,
        frames: res.frames_written,
        fps: res.fps,
        width: res.width,
        height: res.height,
      };
    } catch (err: any) {
      return { applied: false, message: err?.message || String(err) };
    }
  }
}

export const cinematicEngine = new CinematicEngineBridge();
