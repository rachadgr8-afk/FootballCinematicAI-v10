/**
 * ASYNCHRONOUS RENDER JOB CLIENT
 * ----------------------------------------------------------------------------
 * A full 64-second montage takes 1.5-8 minutes of real FFmpeg work. Holding an
 * HTTP request open that long is unreliable: hosting proxies (Render, nginx,
 * Cloudflare) cap origin response time and sever the connection, which the
 * browser surfaces as the infamous **"Failed to fetch"** — even though the
 * server eventually produced a valid MP4.
 *
 * The backend now runs the render as a background JOB:
 *
 *   POST /api/render-full-cinematic  -> 202 { jobId }   (returns in < 1s)
 *   GET  /api/render-progress        -> { percent, stage, status }
 *   GET  /api/render-result          -> the final payload (status === 'done')
 *
 * This module hides that dance behind a single awaitable call so the UI code
 * stays readable, while never hanging a request open for minutes.
 */

import { API_BASE_URL, getApiUrl } from '../config/api';

export interface RenderProgress {
  percent: number;
  stage: string;
  status?: 'idle' | 'running' | 'done' | 'error';
  jobId?: string;
  error?: string;
}

export interface RenderResult {
  success: boolean;
  videoUrl: string;
  posterUrl?: string;
  duration: number;
  fileSize: number;
  localPath?: string;
  rifeApplied?: boolean;
  rifeMultiplier?: number;
  error?: string;
}

const POLL_INTERVAL_MS = 1200;
// Generous ceiling: a slow Render CPU can need several minutes. The server is
// the source of truth for completion; this only prevents an infinite spinner.
const MAX_WAIT_MS = Number(
  (import.meta as any).env?.VITE_RENDER_MAX_WAIT_MS || 15 * 60 * 1000
);

async function readJson<T>(endpoint: string, init?: RequestInit): Promise<{ ok: boolean; status: number; data: T | null }> {
  const url = getApiUrl(endpoint);
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (netErr: any) {
    // A transient network blip while polling is NOT fatal — signal it so the
    // caller can keep waiting instead of aborting the whole render.
    return { ok: false, status: 0, data: null };
  }
  let data: any = null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    try { data = await res.json(); } catch { data = null; }
  }
  return { ok: res.ok, status: res.status, data };
}

/**
 * Starts a render job and drives it to completion by polling.
 *
 * @param onProgress called on every poll tick with the live progress
 * @returns the final render payload, or throws with a precise reason
 */
export async function runCinematicRender(
  body: Record<string, any>,
  onProgress?: (p: RenderProgress) => void
): Promise<RenderResult> {
  // 1) Kick off the job.
  const start = await readJson<{ success: boolean; jobId?: string; busy?: boolean; error?: string }>(
    '/api/render-full-cinematic',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }
  );

  // 400/409/500 are deterministic server responses with useful JSON bodies.
  if (start.status >= 400 && start.data) {
    const reason = start.data.error || `Render could not start (HTTP ${start.status}).`;
    throw new Error(reason);
  }
  // A genuine network failure on the *first* call is worth reporting verbatim,
  // because it usually means the backend is unreachable/asleep.
  if (start.status === 0) {
    throw new Error(
      `Could not reach the render server at ${API_BASE_URL}. It may be waking up — please retry in a few seconds.`
    );
  }

  const jobId = start.data?.jobId;

  // Backward compatibility: an older backend may still answer synchronously with
  // the finished payload. Detect it and return as-is.
  const maybeSync = start.data as any;
  if (maybeSync && maybeSync.videoUrl && maybeSync.success && !maybeSync.async) {
    return maybeSync as RenderResult;
  }

  if (!jobId) {
    throw new Error('The server accepted the render but returned no job id. Please retry.');
  }

  // 2) Poll until done / error / timeout.
  const startedAt = Date.now();
  let lastPercent = -1;

  while (Date.now() - startedAt < MAX_WAIT_MS) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));

    const poll = await readJson<RenderProgress>('/api/render-progress', { method: 'GET' });

    // Network blip (status 0) or proxy 502/503 while the container is busy —
    // keep waiting; the job keeps running server-side.
    if (poll.status === 0 || poll.status === 502 || poll.status === 503) {
      onProgress?.({ percent: Math.max(0, lastPercent), stage: 'Server is busy rendering — waiting…' });
      continue;
    }

    const p = poll.data;
    if (!p) continue;

    if (typeof p.percent === 'number' && p.percent !== lastPercent) {
      lastPercent = p.percent;
    }
    onProgress?.({ percent: p.percent ?? Math.max(0, lastPercent), stage: p.stage || 'Rendering…', status: p.status });

    if (p.status === 'error') {
      throw new Error(p.error || 'The render failed on the server. Please check the logs and retry.');
    }
    if (p.status === 'done') {
      const result = await readJson<RenderResult>(`/api/render-result?jobId=${encodeURIComponent(jobId)}`, { method: 'GET' });
      if (result.status === 200 && result.data?.success) {
        return result.data;
      }
      // Job is done but the result fetch blipped — retry the result endpoint a
      // few times before giving up (the file already exists on the server).
      for (let attempt = 0; attempt < 5; attempt++) {
        await new Promise((r) => setTimeout(r, 1500));
        const retry = await readJson<RenderResult>(`/api/render-result?jobId=${encodeURIComponent(jobId)}`, { method: 'GET' });
        if (retry.status === 200 && retry.data?.success) return retry.data;
      }
      throw new Error('The render finished but the result could not be retrieved. Please retry.');
    }
  }

  throw new Error(
    `The render is taking longer than ${Math.round(MAX_WAIT_MS / 60000)} minutes. ` +
      'It may still be finishing on the server — please retry, and check the server logs if it persists.'
  );
}
