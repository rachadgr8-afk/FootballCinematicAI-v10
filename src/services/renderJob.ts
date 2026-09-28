/**
 * ASYNCHRONOUS CINEMATIC RENDER JOB CLIENT
 *
 * The backend starts the heavy FFmpeg render in the background and immediately
 * returns a jobId. The frontend polls the job until it finishes.
 *
 * Important:
 * - No long HTTP request is kept open.
 * - Temporary network errors do not cancel the render.
 * - The client timeout is intentionally generous because 64s cinematic
 *   rendering can take a long time on Render Free CPU.
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

/*
 * Poll every 2 seconds.
 * 1200ms was unnecessarily aggressive for Render Free.
 */
const POLL_INTERVAL_MS = 2000;

/*
 * IMPORTANT:
 * The old value was 15 minutes and caused the exact error shown in the UI:
 *
 * "The render is taking longer than 15 minutes..."
 *
 * Give the server up to 60 minutes.
 */
const MAX_WAIT_MS = Number(
  (import.meta as any).env?.VITE_RENDER_MAX_WAIT_MS ||
    60 * 60 * 1000
);

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function readJson<T>(
  endpoint: string,
  init?: RequestInit
): Promise<{
  ok: boolean;
  status: number;
  data: T | null;
}> {
  const url = getApiUrl(endpoint);

  let response: Response;

  try {
    response = await fetch(url, {
      ...init,
      cache: 'no-store',
    });
  } catch {
    /*
     * A temporary network failure must NOT kill the render.
     * The backend job continues independently.
     */
    return {
      ok: false,
      status: 0,
      data: null,
    };
  }

  let data: any = null;

  const contentType =
    response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    try {
      data = await response.json();
    } catch {
      data = null;
    }
  }

  return {
    ok: response.ok,
    status: response.status,
    data,
  };
}

/**
 * Start the cinematic render and poll until completion.
 */
export async function runCinematicRender(
  body: Record<string, any>,
  onProgress?: (progress: RenderProgress) => void
): Promise<RenderResult> {

  /*
   * ------------------------------------------------------------
   * 1. START BACKGROUND JOB
   * ------------------------------------------------------------
   */

  const start = await readJson<{
    success: boolean;
    jobId?: string;
    busy?: boolean;
    async?: boolean;
    error?: string;
    videoUrl?: string;
    posterUrl?: string;
    duration?: number;
    fileSize?: number;
    localPath?: string;
  }>(
    '/api/render-full-cinematic',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }
  );

  /*
   * Server explicitly rejected the job.
   */
  if (start.status >= 400) {
    const message =
      start.data?.error ||
      `Render could not start (HTTP ${start.status}).`;

    throw new Error(message);
  }

  /*
   * Backend unreachable.
   */
  if (start.status === 0) {
    throw new Error(
      `Could not reach the render server at ${API_BASE_URL}. ` +
      `The server may be waking up. Please retry in a few seconds.`
    );
  }

  const response = start.data;

  /*
   * Backward compatibility with an older synchronous backend.
   */
  if (
    response &&
    response.success &&
    response.videoUrl &&
    !response.async
  ) {
    return {
      success: true,
      videoUrl: response.videoUrl,
      posterUrl: response.posterUrl,
      duration: Number(response.duration || 64),
      fileSize: Number(response.fileSize || 0),
      localPath: response.localPath,
    };
  }

  /*
   * New asynchronous backend.
   */
  const jobId = response?.jobId;

  if (!jobId) {
    throw new Error(
      'The server accepted the render but returned no job ID. Please retry.'
    );
  }

  onProgress?.({
    percent: 0,
    stage: 'Render job started...',
    status: 'running',
    jobId,
  });

  /*
   * ------------------------------------------------------------
   * 2. POLL JOB
   * ------------------------------------------------------------
   */

  const startedAt = Date.now();
  let lastPercent = 0;
  let lastStage = 'Starting render...';

  while (Date.now() - startedAt < MAX_WAIT_MS) {

    await sleep(POLL_INTERVAL_MS);

    /*
     * IMPORTANT:
     * Pass jobId explicitly.
     *
     * This is safer than relying only on the server's active job.
     */
    const progress = await readJson<RenderProgress>(
      `/api/render-progress?jobId=${encodeURIComponent(jobId)}`,
      {
        method: 'GET',
      }
    );

    /*
     * Temporary network/proxy error.
     *
     * Do NOT abort the render.
     */
    if (
      progress.status === 0 ||
      progress.status === 502 ||
      progress.status === 503 ||
      progress.status === 504
    ) {
      onProgress?.({
        percent: lastPercent,
        stage:
          'Server is busy rendering — reconnecting...',
        status: 'running',
        jobId,
      });

      continue;
    }

    /*
     * Unexpected HTTP error.
     */
    if (progress.status >= 400) {
      /*
       * A transient 404 can happen immediately after a restart.
       * Give the server another chance instead of killing the job.
       */
      if (progress.status === 404) {
        onProgress?.({
          percent: lastPercent,
          stage:
            'Render job temporarily unavailable — retrying...',
          status: 'running',
          jobId,
        });

        continue;
      }

      throw new Error(
        progress.data?.error ||
        `Render progress request failed (HTTP ${progress.status}).`
      );
    }

    const data = progress.data;

    if (!data) {
      continue;
    }

    /*
     * Keep the latest valid progress.
     */
    if (typeof data.percent === 'number') {
      lastPercent = Math.max(
        0,
        Math.min(100, data.percent)
      );
    }

    if (data.stage) {
      lastStage = data.stage;
    }

    onProgress?.({
      percent: lastPercent,
      stage: lastStage,
      status: data.status,
      jobId,
      error: data.error,
    });

    /*
     * ----------------------------------------------------------
     * SERVER RENDER FAILED
     * ----------------------------------------------------------
     */
    if (data.status === 'error') {
      throw new Error(
        data.error ||
        'The cinematic render failed on the server.'
      );
    }

    /*
     * ----------------------------------------------------------
     * SERVER RENDER FINISHED
     * ----------------------------------------------------------
     */
    if (data.status === 'done') {

      onProgress?.({
        percent: 100,
        stage: 'Render finished — retrieving final MP4...',
        status: 'done',
        jobId,
      });

      /*
       * Retrieve the actual result.
       */
      for (let attempt = 0; attempt < 10; attempt++) {

        const result = await readJson<RenderResult>(
          `/api/render-result?jobId=${encodeURIComponent(jobId)}`,
          {
            method: 'GET',
          }
        );

        /*
         * Success.
         */
        if (
          result.status === 200 &&
          result.data?.success &&
          result.data.videoUrl
        ) {
          return result.data;
        }

        /*
         * The job is done but the result endpoint may briefly
         * fail while the file/storage operation completes.
         */
        if (
          result.status === 0 ||
          result.status === 502 ||
          result.status === 503 ||
          result.status === 504
        ) {
          await sleep(2000);
          continue;
        }

        await sleep(1500);
      }

      throw new Error(
        'The render finished, but the final MP4 could not be retrieved. Please retry.'
      );
    }
  }

  /*
   * ------------------------------------------------------------
   * 3. CLIENT-SIDE SAFETY TIMEOUT
   * ------------------------------------------------------------
   *
   * This is now 60 minutes instead of 15 minutes.
   * The backend job may still continue after this point.
   */

  const minutes = Math.round(
    MAX_WAIT_MS / 60000
  );

  throw new Error(
    `The render is taking longer than ${minutes} minutes. ` +
    `The server job may still be running. ` +
    `Check the render status and retry if necessary.`
  );
}