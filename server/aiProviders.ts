// ============================================================================
// OPTIONAL AI PROVIDERS — OpenRouter (text + VISION) and DeepSeek (text)
// ----------------------------------------------------------------------------
// Both providers are OPTIONAL. The render pipeline never depends on them:
//   - key present  -> the call is really executed (status 'executed')
//   - key missing  -> status 'not_configured' and the caller uses its fallback
//   - call fails   -> status 'failed'        and the caller uses its fallback
//
// SECURITY
//   * Keys are read from the environment ONLY (OPENROUTER_API_KEY /
//     DEEPSEEK_API_KEY). No key is ever hardcoded, logged, echoed in a response
//     or written to disk.
//   * Every error string is passed through `sanitize()` which strips anything
//     that looks like a key/Authorization header before it can reach a log.
// ============================================================================

export type ProviderName = 'openrouter' | 'deepseek';
export type ProviderStatus = 'executed' | 'failed' | 'not_configured';

export interface ProviderResult<T = any> {
  status: ProviderStatus;
  provider: ProviderName;
  model: string;
  httpStatus: number;
  latencyMs: number;
  data?: T;
  /** Sanitized, human-readable error (never contains a key/header). */
  error?: string;
}

const OPENROUTER_BASE = (process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1').replace(/\/+$/, '');
const DEEPSEEK_BASE = (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\/+$/, '');

export const OPENROUTER_TEXT_MODEL = process.env.OPENROUTER_TEXT_MODEL || 'google/gemini-2.5-flash-lite';
export const OPENROUTER_VISION_MODEL = process.env.OPENROUTER_VISION_MODEL || 'google/gemini-2.5-flash-lite';
export const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL || 'deepseek-chat';

const DEFAULT_TEXT_TIMEOUT_MS = Number(process.env.AI_TEXT_TIMEOUT_MS || 60000);
const DEFAULT_VISION_TIMEOUT_MS = Number(process.env.AI_VISION_TIMEOUT_MS || 120000);

export function openRouterKey(): string {
  return (process.env.OPENROUTER_API_KEY || '').trim();
}
export function deepseekKey(): string {
  return (process.env.DEEPSEEK_API_KEY || '').trim();
}

export function providerConfigured(provider: ProviderName): boolean {
  return provider === 'openrouter' ? openRouterKey().length > 0 : deepseekKey().length > 0;
}

/**
 * Remove anything that could leak a secret from text that may be logged.
 * Kills API keys (sk-…, sk-or-…), bearer tokens and Authorization headers.
 */
export function sanitize(input: any): string {
  let s = typeof input === 'string' ? input : (() => {
    try { return JSON.stringify(input); } catch { return String(input); }
  })();
  s = s.replace(/sk-[A-Za-z0-9._-]{8,}/g, 'sk-***');
  s = s.replace(/(Bearer|bearer)\s+[A-Za-z0-9._-]{6,}/g, '$1 ***');
  s = s.replace(/("?authorization"?\s*[:=]\s*)("[^"]*"|[^,}\s]+)/gi, '$1"***"');
  s = s.replace(/api[_-]?key["']?\s*[:=]\s*["']?[A-Za-z0-9._-]{6,}/gi, 'api_key=***');
  return s.slice(0, 500);
}

interface ChatOptions {
  system?: string;
  user: string;
  /** Optional data-URL images (base64) for VLM calls. */
  images?: string[];
  model: string;
  maxTokens?: number;
  temperature?: number;
  json?: boolean;
  timeoutMs?: number;
  provider: ProviderName;
}

function buildContent(opts: ChatOptions): any {
  if (opts.images && opts.images.length) {
    const parts: any[] = [{ type: 'text', text: opts.user }];
    for (const url of opts.images) parts.push({ type: 'image_url', image_url: { url } });
    return parts;
  }
  return opts.user;
}

async function postChat(opts: ChatOptions, apiKey: string, base: string, extraHeaders: Record<string, string> = {}): Promise<ProviderResult<any>> {
  const started = Date.now();
  const model = opts.model;
  const messages: any[] = [];
  if (opts.system) messages.push({ role: 'system', content: opts.system });
  messages.push({ role: 'user', content: buildContent(opts) });

  const body: any = {
    model,
    messages,
    max_tokens: opts.maxTokens ?? 1024,
    temperature: opts.temperature ?? 0.4,
  };
  if (opts.json) body.response_format = { type: 'json_object' };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), opts.timeoutMs ?? DEFAULT_TEXT_TIMEOUT_MS);

  try {
    const res = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: {
        // Never logged. The header value is never included in any error path.
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        ...extraHeaders,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const latencyMs = Date.now() - started;
    const rawText = await res.text();
    let parsed: any = null;
    try { parsed = JSON.parse(rawText); } catch { /* non-JSON error page */ }

    if (!res.ok) {
      const msg = parsed?.error?.message || parsed?.message || rawText || `HTTP ${res.status}`;
      return { status: 'failed', provider: opts.provider, model, httpStatus: res.status, latencyMs, error: sanitize(msg) };
    }

    const content = parsed?.choices?.[0]?.message?.content ?? parsed?.choices?.[0]?.text ?? '';
    return {
      status: 'executed',
      provider: opts.provider,
      model,
      httpStatus: res.status,
      latencyMs,
      data: { content, usage: parsed?.usage ?? null, raw: parsed },
    };
  } catch (err: any) {
    const latencyMs = Date.now() - started;
    const msg = err?.name === 'AbortError' ? `timeout after ${opts.timeoutMs ?? DEFAULT_TEXT_TIMEOUT_MS}ms` : err?.message || 'network error';
    return { status: 'failed', provider: opts.provider, model, httpStatus: 0, latencyMs, error: sanitize(msg) };
  } finally {
    clearTimeout(timeout);
  }
}

/** OpenRouter text completion (optional). */
export async function openRouterText(user: string, opts: Partial<ChatOptions> = {}): Promise<ProviderResult<any>> {
  const key = openRouterKey();
  const model = opts.model || OPENROUTER_TEXT_MODEL;
  if (!key) return { status: 'not_configured', provider: 'openrouter', model, httpStatus: 0, latencyMs: 0, error: 'OPENROUTER_API_KEY not set' };
  return postChat(
    { provider: 'openrouter', model, user, system: opts.system, maxTokens: opts.maxTokens, temperature: opts.temperature, json: opts.json, timeoutMs: opts.timeoutMs ?? DEFAULT_TEXT_TIMEOUT_MS },
    key,
    OPENROUTER_BASE,
    { 'HTTP-Referer': process.env.OPENROUTER_REFERER || 'https://footballcinematicai.app', 'X-Title': 'Football Cinematic AI' }
  );
}

/** OpenRouter VLM completion: REAL image data (data URLs) is sent to the model. */
export async function openRouterVision(images: string[], user: string, opts: Partial<ChatOptions> = {}): Promise<ProviderResult<any>> {
  const key = openRouterKey();
  const model = opts.model || OPENROUTER_VISION_MODEL;
  if (!key) return { status: 'not_configured', provider: 'openrouter', model, httpStatus: 0, latencyMs: 0, error: 'OPENROUTER_API_KEY not set' };
  if (!images || !images.length) return { status: 'failed', provider: 'openrouter', model, httpStatus: 0, latencyMs: 0, error: 'no image frames provided' };
  return postChat(
    { provider: 'openrouter', model, user, images, system: opts.system, maxTokens: opts.maxTokens ?? 1200, temperature: opts.temperature ?? 0.3, json: opts.json ?? true, timeoutMs: opts.timeoutMs ?? DEFAULT_VISION_TIMEOUT_MS },
    key,
    OPENROUTER_BASE,
    { 'HTTP-Referer': process.env.OPENROUTER_REFERER || 'https://footballcinematicai.app', 'X-Title': 'Football Cinematic AI' }
  );
}

/** DeepSeek text completion (optional). */
export async function deepseekText(user: string, opts: Partial<ChatOptions> = {}): Promise<ProviderResult<any>> {
  const key = deepseekKey();
  const model = opts.model || DEEPSEEK_MODEL;
  if (!key) return { status: 'not_configured', provider: 'deepseek', model, httpStatus: 0, latencyMs: 0, error: 'DEEPSEEK_API_KEY not set' };
  return postChat(
    { provider: 'deepseek', model, user, system: opts.system, maxTokens: opts.maxTokens, temperature: opts.temperature, json: opts.json, timeoutMs: opts.timeoutMs ?? DEFAULT_TEXT_TIMEOUT_MS },
    key,
    DEEPSEEK_BASE
  );
}

export interface ProviderSnapshot {
  configured: boolean;
  status: ProviderStatus;
  model: string;
  httpStatus: number;
  latencyMs: number;
  error?: string;
  at: number;
}

/** In-memory, secret-free record of the last real call per provider (for /api/ai/status). */
class ProviderHealth {
  private last: Record<string, ProviderSnapshot | undefined> = {};
  public record(provider: ProviderName, kind: 'text' | 'vision', r: ProviderResult<any>): void {
    const prev = this.last[`${provider}:${kind}`];
    // A successful execution overwrites; a not_configured must not hide a real result.
    if (r.status === 'not_configured' && prev && prev.status === 'executed') return;
    this.last[`${provider}:${kind}`] = {
      configured: providerConfigured(provider),
      status: r.status,
      model: r.model,
      httpStatus: r.httpStatus,
      latencyMs: r.latencyMs,
      error: r.error,
      at: Date.now(),
    };
  }
  public snapshot(): Record<string, ProviderSnapshot | undefined> {
    return { ...this.last };
  }
}
export const providerHealth = new ProviderHealth();
