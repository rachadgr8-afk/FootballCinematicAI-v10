/**
 * Centralized API configuration for FOOTBALL CINEMATIC AI
 * Single source of truth for backend routes and upload pipeline
 */

/**
 * Backend base URL.
 *
 * DEFAULT = '' (empty / same-origin). Because the production deployment is
 * MONOLITHIC (the Express server serves both the built SPA and the API from one
 * origin), every relative call such as `/api/upload-video` or `/videos/x.mp4`
 * must resolve to the SAME server that served the page.
 *
 * The production Android build uses the configured Render backend `https://fotbal-1.onrender.com`; Vite can still override it with `VITE_API_BASE_URL` for another deployment. The old hardcoded fallback was the root cause
 * of the "Upload service endpoint was not found on the server" 404: whenever the
 * external Render backend was asleep, rebuilt, or its free instance expired, all
 * uploads/render calls hit a dead host and 404'd — even though the local
 * monolithic server had those exact routes.
 *
 * For the packaged Capacitor app (WebView origin `https://localhost`), the build
 * injects VITE_API_BASE_URL so relative paths still reach the backend.
 */
export const DEFAULT_API_BASE_URL: string = 'https://fotbal-1.onrender.com';

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string) ?? DEFAULT_API_BASE_URL;

export const UPLOAD_ENDPOINT: string = '/api/upload-video';

/**
 * Builds full endpoint URL using the centralized API_BASE_URL
 */
export function getApiUrl(path: string, baseUrl: string = API_BASE_URL): string {
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('blob:')) {
    return path;
  }
  const cleanBase = baseUrl.replace(/\/+$/, '');
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return cleanBase ? `${cleanBase}${cleanPath}` : cleanPath;
}

/**
 * Resolves a media path (video/poster) to an absolute URL.
 *
 * Essential for the packaged Android app (Capacitor): the WebView origin is
 * `https://localhost`, so a relative `/videos/foo.mp4` would wrongly point at
 * the app itself. This always prefixes the backend API base URL for relative
 * paths, while leaving absolute (http/https/blob/data) URLs untouched.
 */
export function resolveMediaUrl(path?: string | null): string {
  if (!path) return '';
  return getApiUrl(path);
}

/**
 * Maps HTTP status codes to user-friendly diagnostic messages.
 *
 * `endpoint` is optional context so a 404 can say WHICH route was missing
 * (e.g. the upload route vs. the analyse route) instead of the generic and
 * misleading "please verify file format" wording.
 */
export function getFriendlyErrorMessage(status: number, bodySnippet: string = '', endpoint: string = ''): string {
  const route = endpoint ? ` (${endpoint})` : '';
  switch (status) {
    case 400:
      return 'Bad request. Please verify the video file format and parameters.';
    case 401:
    case 403:
      return 'Request blocked. The server refused this operation (authentication/permission).';
    case 404:
      return `The server did not find the required API route${route}. The backend may be running an older build or a different service. Run /api/version and redeploy the current backend.`;
    case 413:
      return 'Video file is too large (maximum allowed is 500MB).';
    case 415:
      return 'Video format is not supported. Please select an MP4, MOV, or WEBM file.';
    case 429:
      return 'Gemini API quota is momentarily exhausted on every key. The server rotates keys automatically — please retry in a few seconds.';
    case 500:
      return `The server encountered an internal error while handling${route || ' the request'}. Check the server logs for details.`;
    case 502:
    case 503:
      return `The server is temporarily unavailable or still starting up${route}. Please retry in a few seconds.`;
    default:
      if (bodySnippet.length > 0) {
        return `Server returned error (${status})${route}: ${bodySnippet.substring(0, 180)}`;
      }
      return `Server error (${status})${route}.`;
  }
}

/**
 * Safe JSON fetch utility with strict Content-Type and response.ok verification.
 * Guarantees that HTML error pages never trigger "Unexpected token '<'" exceptions.
 */
export async function safeFetchJson<T = any>(
  endpoint: string,
  options?: RequestInit,
  preferBaseUrl?: string
): Promise<T> {
  const url = getApiUrl(endpoint, preferBaseUrl);
  let response: Response;

  try {
    response = await fetch(url, options);
  } catch (netErr: any) {
    throw new Error(`Network connection failure to ${url}: ${netErr.message}`);
  }

  const contentType = response.headers.get('content-type') || '';
  const isJson = contentType.includes('application/json');

  if (!response.ok) {
    let errorText = '';
    try {
      errorText = await response.text();
    } catch {
      errorText = response.statusText;
    }

    // Never output raw HTML stack traces or doctype tags
    const cleanSnippet = errorText.replace(/<[^>]*>?/gm, ' ').replace(/\s+/g, ' ').trim();
    const friendlyMsg = getFriendlyErrorMessage(response.status, cleanSnippet, endpoint);

    console.warn(`[API DEBUG] HTTP ${response.status} from ${url}:`, {
      contentType,
      status: response.status,
      bodySnippet: cleanSnippet.substring(0, 300),
    });

    throw new Error(friendlyMsg);
  }

  if (!isJson) {
    const rawBody = await response.text();
    const cleanSnippet = rawBody.replace(/<[^>]*>?/gm, ' ').replace(/\s+/g, ' ').trim();
    console.warn(`[API DEBUG] Expected JSON but received non-JSON (${contentType}) from ${url}:`, cleanSnippet.substring(0, 300));
    throw new Error(
      `Server returned unexpected format (${contentType || 'non-JSON'}). Expected JSON response.`
    );
  }

  try {
    return (await response.json()) as T;
  } catch (jsonErr: any) {
    throw new Error(`Failed to parse server JSON response: ${jsonErr.message}`);
  }
}

/**
 * Validates a video file selected by the user
 * Reads dimensions, duration, MIME type, and confirms it can be decoded
 */
export async function validateVideoFile(file: File): Promise<{
  duration: number;
  width: number;
  height: number;
  blobUrl: string;
}> {
  if (!file) {
    throw new Error('No video file selected.');
  }

  // 1. Verify file size is not empty
  if (file.size === 0) {
    throw new Error('Cannot read this video file. The selected file is empty (0 bytes).');
  }

  // 2. Check file size maximum limit (500MB)
  if (file.size > 500 * 1024 * 1024) {
    throw new Error('Video file is too large (maximum allowed size is 500MB).');
  }

  // 3. Verify file can actually be opened and read from storage (tests OS/URI permissions)
  try {
    const chunk = file.slice(0, Math.min(file.size, 4096));
    const buffer = await chunk.arrayBuffer();
    if (buffer.byteLength === 0 && file.size > 0) {
      throw new Error('Cannot read this video file.');
    }
  } catch (readErr: any) {
    console.error('File binary read verification failed:', readErr);
    throw new Error('Cannot read this video file.');
  }

  // 4. Check MIME type or video extension
  const validExtensions = ['.mp4', '.mov', '.webm', '.mkv', '.m4v', '.avi', '.ts'];
  const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
  const isVideoMime = file.type.startsWith('video/') || validExtensions.includes(ext);

  if (!isVideoMime) {
    throw new Error('Video format is not supported. Please select an MP4, MOV, or WEBM video file.');
  }

  // 5. Create object URL for client preview & metadata probe
  let blobUrl = '';
  try {
    blobUrl = URL.createObjectURL(file);
  } catch {
    blobUrl = '';
  }

  // 6. Non-blocking metadata probe via HTML5 Video
  // If the browser does not natively support the codec (e.g. HEVC/H.265/ProRes in an iframe),
  // we do NOT fail the upload — FFmpeg on the server will decode and probe it with 100% precision.
  return new Promise((resolve) => {
    if (!blobUrl) {
      resolve({
        duration: 60.0,
        width: 1920,
        height: 1080,
        blobUrl: '',
      });
      return;
    }

    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;

    let hasCompleted = false;
    const complete = (meta: { duration: number; width: number; height: number; blobUrl: string }) => {
      if (!hasCompleted) {
        hasCompleted = true;
        clearTimeout(timeout);
        try {
          video.src = '';
        } catch {
          // ignore
        }
        resolve(meta);
      }
    };

    const timeout = setTimeout(() => {
      complete({
        duration: 60.0,
        width: 1920,
        height: 1080,
        blobUrl,
      });
    }, 2000);

    video.onloadedmetadata = () => {
      const dur = video.duration && !isNaN(video.duration) && video.duration > 0 ? video.duration : 60.0;
      const w = video.videoWidth || 1920;
      const h = video.videoHeight || 1080;
      complete({
        duration: dur,
        width: w,
        height: h,
        blobUrl,
      });
    };

    video.onerror = () => {
      // Browser native decoder cannot render this container/codec (e.g. HEVC/ProRes in sandboxed iframe)
      // Allow upload to continue; server FFmpeg handles decoding and metadata extraction
      console.warn('[METADATA PROBE] HTML5 video decoder could not parse preview. Server FFmpeg will decode.');
      complete({
        duration: 60.0,
        width: 1920,
        height: 1080,
        blobUrl,
      });
    };

    try {
      video.src = blobUrl;
      video.load();
    } catch {
      complete({
        duration: 60.0,
        width: 1920,
        height: 1080,
        blobUrl,
      });
    }
  });
}

export interface UploadProgressInfo {
  percent: number;
  stage: string;
  loadedBytes?: number;
  totalBytes?: number;
}

export interface VideoUploadResult {
  success: boolean;
  videoId: string;
  videoUrl: string;
  localPath: string;
  filename: string;
  title: string;
  size: number;
  mimeType: string;
  duration?: number;
  width?: number;
  height?: number;
  posterUrl?: string;
}

/**
 * Pings the remote backend's /health route before uploading.
 *
 * Free-tier hosts like Render spin the server down after inactivity. The
 * first request after a sleep period can take 20-60s to respond and, while
 * waking up, the platform's own proxy returns an HTML error page (502/503)
 * instead of JSON — which is exactly what caused the raw
 * "Unexpected token '<' ... is not valid JSON" crash. Retrying a cheap
 * health check first (with visible progress) absorbs that cold-start window
 * instead of letting the actual upload request fail on it.
 */
async function wakeUpBackend(
  baseUrl: string,
  onProgress?: (progress: UploadProgressInfo) => void,
  maxWaitMs: number = 45000
): Promise<boolean> {
  if (!baseUrl || !(baseUrl.startsWith('http://') || baseUrl.startsWith('https://'))) {
    // Relative/local base — same origin as the app, no cold start to wait out.
    return true;
  }

  const healthUrl = getApiUrl('/health', baseUrl);
  const start = Date.now();
  let attempt = 0;

  while (Date.now() - start < maxWaitMs) {
    attempt += 1;
    try {
      const res = await fetch(healthUrl, { method: 'GET' });
      const contentType = res.headers.get('content-type') || '';
      if (res.ok && contentType.includes('application/json')) {
        if (attempt > 1) {
          onProgress?.({ percent: 3, stage: 'Server is awake. Starting upload...' });
        }
        return true;
      }
      // Server responded but not healthy yet (HTML error page, wrong status) — keep waiting.
    } catch {
      // Network error while sleeping/booting — keep waiting.
    }

    onProgress?.({
      percent: 1,
      stage: `Waking up upload server (attempt ${attempt})... this can take up to a minute on first use.`,
    });
    await new Promise((r) => setTimeout(r, 4000));
  }

  // Timed out waiting — let the real upload attempt run anyway and surface its own error.
  return false;
}

/**
 * Uploads a video file using XMLHttpRequest for REAL upload progress reporting.
 * Handles boundary generation automatically (never sets Content-Type manually).
 * Falls back to the local full-stack video engine if the remote AI backend lacks the upload route.
 */
export async function uploadVideo(
  file: File,
  onProgress?: (progress: UploadProgressInfo) => void
): Promise<VideoUploadResult> {
  // Step 1: Validate file locally first
  onProgress?.({ percent: 0, stage: 'Preparing and validating video...' });
  const meta = await validateVideoFile(file);

  onProgress?.({ percent: 5, stage: 'Video validated. Initiating upload stream...' });

  // Step 1.5: Absorb any cold-start delay on the remote backend BEFORE attempting
  // the real multipart upload, so the user sees a clear "waking up" message instead
  // of a raw JSON-parse crash.
  await wakeUpBackend(API_BASE_URL, onProgress);

  const formData = new FormData();
  formData.append('video', file, file.name);

  // Helper to execute XHR upload
  const executeXhrUpload = (targetUrl: string): Promise<VideoUploadResult> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', targetUrl, true);

      // Real progress handler from browser's upload stream
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.min(98, Math.round((event.loaded / event.total) * 90));
          onProgress?.({
            percent: pct,
            stage: `Uploading video (${pct}%)...`,
            loadedBytes: event.loaded,
            totalBytes: event.total,
          });
        } else {
          onProgress?.({ percent: 50, stage: 'Uploading video data...' });
        }
      };

      xhr.onload = () => {
        const status = xhr.status;
        const contentType = xhr.getResponseHeader('Content-Type') || '';
        const rawResponse = xhr.responseText || '';
        const cleanSnippet = rawResponse.replace(/<[^>]*>?/gm, ' ').replace(/\s+/g, ' ').trim();

        console.log(`[UPLOAD DEBUG] Status: ${status}, Content-Type: ${contentType}, Target: ${targetUrl}`);

        // Status check
        if (status < 200 || status >= 300) {
          const friendlyMessage = getFriendlyErrorMessage(status, cleanSnippet, '/api/upload-video');
          reject(new Error(`Upload failed (${status}). ${friendlyMessage}`));
          return;
        }

        // Content-Type validation
        if (!contentType.includes('application/json')) {
          reject(
            new Error(
              `Upload server returned non-JSON response (${contentType || 'empty'}): ${cleanSnippet.substring(0, 300)}`
            )
          );
          return;
        }

        // Safe JSON parsing
        try {
          const data = JSON.parse(rawResponse);
          if (data.success === false && data.error) {
            reject(new Error(data.error.message || 'Upload rejected by server'));
            return;
          }
          onProgress?.({ percent: 100, stage: 'Upload complete! Processing video metadata...' });
          resolve({
            success: true,
            videoId: data.videoId || data.filename || `vid_${Date.now()}`,
            videoUrl: data.videoUrl || meta.blobUrl,
            localPath: data.localPath || `public/videos/${data.filename}`,
            filename: data.filename || file.name,
            title: data.title || file.name,
            size: data.size || file.size,
            mimeType: data.mimeType || file.type || 'video/mp4',
            duration: data.duration || meta.duration,
            width: data.width || meta.width,
            height: data.height || meta.height,
            posterUrl: data.posterUrl,
          });
        } catch (e: any) {
          reject(new Error(`Failed to parse server JSON response: ${e.message}`));
        }
      };

      xhr.onerror = () => {
        reject(new Error('Network error during video upload. Please check connection.'));
      };

      xhr.ontimeout = () => {
        reject(new Error('Video upload timed out. File may be too large for current connection.'));
      };

      xhr.timeout = 180000; // 3 minute timeout for large match videos

      // Send form data; browser sets multipart boundary automatically
      xhr.send(formData);
    });
  };

  // Step 2: Try primary upload endpoint
  // If API_BASE_URL is remote and returns 404 (e.g. Render server has no upload route),
  // automatically route to local application server where Multer & FFmpeg are active
  const primaryUrl = getApiUrl('/api/upload-video', API_BASE_URL);

  const isRemote = API_BASE_URL.startsWith('http://') || API_BASE_URL.startsWith('https://');

  try {
    return await executeXhrUpload(primaryUrl);
  } catch (primaryErr: any) {
    const msg: string = primaryErr.message || '';
    const isEndpointNotFound = msg.includes('not found') || msg.includes('404');
    // Cold-start / gateway issues (server waking up, restarting, or momentarily
    // unavailable) surface as 502/503, or as a non-JSON/parse error when the
    // platform's own HTML error page slips through instead of the API's JSON.
    const isTransientServerIssue =
      msg.includes('temporarily unavailable') ||
      msg.includes('502') ||
      msg.includes('503') ||
      msg.includes('non-JSON') ||
      msg.includes('Failed to parse server JSON response') ||
      msg.includes('Unexpected token');

    if (isRemote && isTransientServerIssue) {
      // Give the remote host one more chance to finish waking up, then retry once.
      console.warn(`[UPLOAD DEBUG] Primary remote endpoint ${primaryUrl} looked like a transient/cold-start failure. Retrying once after a short wait.`);
      onProgress?.({ percent: 2, stage: 'Upload server seems to be starting up — retrying in a moment...' });
      await new Promise((r) => setTimeout(r, 8000));
      const stillAwake = await wakeUpBackend(API_BASE_URL, onProgress, 30000);
      if (stillAwake) {
        try {
          return await executeXhrUpload(primaryUrl);
        } catch (retryErr: any) {
          // Fall through to local fallback / final error below.
          primaryErr = retryErr;
        }
      }
    }

    if (isEndpointNotFound && isRemote) {
      console.warn(`[UPLOAD DEBUG] Primary remote endpoint ${primaryUrl} returned 404. Falling back to local video engine /api/upload-video`);
      onProgress?.({ percent: 10, stage: 'Routing to application video engine...' });
      const localUrl = '/api/upload-video';
      return await executeXhrUpload(localUrl);
    }

    throw primaryErr;
  }
}
