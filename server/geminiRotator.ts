// ============================================================================
// GEMINI API KEY ROTATOR
// ----------------------------------------------------------------------------
// The pipeline relies on several Gemini API keys so a single exhausted key
// (429 RESOURCE_EXHAUSTED) no longer blocks a render. This module:
//
//   1. Loads every configured key (GEMINI_API_KEYS, comma/newline/space
//      separated, plus the legacy single GEMINI_API_KEY as a fallback).
//   2. Hands out keys in a fair round-robin order, skipping keys that are
//      currently cooling down or marked invalid.
//   3. Wraps a whole operation in `run()`: if the operation fails with a
//      quota / overload / transient error, the key is put on cooldown and the
//      SAME operation is retried with the next healthy key.
//
// IMPORTANT: the Gemini Files API is scoped per key. An uploaded video URI can
// only be read with the key that uploaded it. `run()` therefore wraps a whole
// atomic operation (upload + poll + generate) so every step inside one attempt
// uses a single key. Only the retry uses a different key.
// ============================================================================

import { GoogleGenAI } from '@google/genai';

export type KeyStatus = 'active' | 'cooling' | 'invalid';

export interface KeyState {
  index: number;
  key: string;
  status: KeyStatus;
  cooldownUntil: number; // epoch ms, 0 when not cooling
  errorCount: number;
  successCount: number;
  lastError?: string;
  lastUsedAt?: number;
}

export interface RotatorStatus {
  totalKeys: number;
  activeKeys: number;
  coolingKeys: number;
  invalidKeys: number;
  keys: Array<{
    index: number;
    label: string;
    status: KeyStatus;
    cooldownRemainingMs: number;
    errorCount: number;
    successCount: number;
    lastError?: string;
  }>;
}

// How long a key stays on cooldown after a quota error when the API does not
// tell us a specific retry delay.
const DEFAULT_QUOTA_COOLDOWN_MS = Number(process.env.GEMINI_QUOTA_COOLDOWN_MS || 60_000);
// Short cooldown for transient overloads (503 / "high demand").
const OVERLOAD_COOLDOWN_MS = Number(process.env.GEMINI_OVERLOAD_COOLDOWN_MS || 12_000);
// Max time to wait for the next healthy key before replaying a cooled one.
const MAX_COOLDOWN_BACKOFF_MS = Number(process.env.GEMINI_MAX_COOLDOWN_MS || 5 * 60_000);

function parseKeysFromEnv(): string[] {
  const raw = [
    process.env.GEMINI_API_KEYS || '',
    process.env.GEMINI_API_KEY || '',
    process.env.GOOGLE_API_KEY || '',
  ]
    .join(',')
    .split(/[\s,;]+/)
    .map((k) => k.trim())
    .filter(Boolean);

  // De-duplicate while preserving order.
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const k of raw) {
    if (!seen.has(k)) {
      seen.add(k);
      unique.push(k);
    }
  }
  return unique;
}

export function maskKey(key: string): string {
  if (!key) return '(none)';
  if (key.length <= 10) return `${key.slice(0, 3)}***`;
  return `${key.slice(0, 8)}...${key.slice(-4)}`;
}

/** Detect a quota / rate-limit error (429 / RESOURCE_EXHAUSTED). */
export function isQuotaError(err: any): boolean {
  const s = `${err?.message || ''} ${err?.status || ''} ${err?.code || ''} ${safeStringify(err)}`;
  return /429|RESOURCE_EXHAUSTED|quota|rate limit|exceeded your current quota|exhausted/i.test(s);
}

/** Detect a permanently invalid key (401/403 / API_KEY_INVALID). */
export function isInvalidKeyError(err: any): boolean {
  const s = `${err?.message || ''} ${err?.status || ''} ${err?.code || ''} ${safeStringify(err)}`;
  return /API_KEY_INVALID|API key not valid|PERMISSION_DENIED|UNAUTHENTICATED|401|403/i.test(s);
}

/** Detect a transient overload / server error that deserves a key rotation. */
export function isOverloadError(err: any): boolean {
  const s = `${err?.message || ''} ${err?.status || ''} ${err?.code || ''} ${safeStringify(err)}`;
  return /503|UNAVAILABLE|high demand|overloaded|temporar|internal error|500|502|504|DEADLINE_EXCEEDED|timeout/i.test(s);
}

/**
 * Errors that justify rotating to another key and replaying the operation.
 *
 * IMPORTANT: a permanently invalid/dead key (API_KEY_INVALID / 401 / 403) MUST
 * be rotatable. The key pool is expected to contain a mix of valid and expired
 * keys, and the whole point of the rotator is to skip a dead key and use a live
 * one. Omitting `isInvalidKeyError` here made `run()` rethrow on the first dead
 * key instead of continuing, so a pool with ANY expired key failed the entire
 * render — the second cause of the "generation stops / app no longer works"
 * outage.
 */
export function isRotatableError(err: any): boolean {
  return isInvalidKeyError(err) || isQuotaError(err) || isOverloadError(err);
}

function safeStringify(obj: any): string {
  try {
    return typeof obj === 'string' ? obj : JSON.stringify(obj);
  } catch {
    return String(obj);
  }
}

/** Parse a Google retryDelay (e.g. "37s") into milliseconds, if present. */
function parseRetryDelayMs(err: any): number | null {
  try {
    const text = safeStringify(err);
    const m = text.match(/"?retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s"?/i);
    if (m) return Math.ceil(parseFloat(m[1]) * 1000);
  } catch {
    /* ignore */
  }
  return null;
}

export class GeminiKeyRotator {
  private states: KeyState[] = [];
  private clients: Map<number, GoogleGenAI> = new Map();
  private cursor = 0;
  private readonly httpHeaders: Record<string, string>;

  constructor() {
    this.httpHeaders = { 'User-Agent': 'aistudio-build' };
    this.reload();
  }

  /** (Re)load keys from the environment. Safe to call at boot. */
  public reload(): void {
    const keys = parseKeysFromEnv();
    this.states = keys.map((key, index) => ({
      index,
      key,
      status: 'active' as KeyStatus,
      cooldownUntil: 0,
      errorCount: 0,
      successCount: 0,
    }));
    this.clients.clear();
    this.cursor = 0;

    if (!this.states.length) {
      console.warn('[KEY ROTATOR] No Gemini API keys configured. Set GEMINI_API_KEYS (comma separated).');
    } else {
      console.log(`[KEY ROTATOR] Loaded ${this.states.length} Gemini key(s): ${this.states.map((s) => maskKey(s.key)).join(', ')}`);
    }
  }

  public count(): number {
    return this.states.length;
  }

  public hasKeys(): boolean {
    return this.states.length > 0;
  }

  /** Cached GoogleGenAI client for a key index. */
  public clientFor(index: number): GoogleGenAI {
    let client = this.clients.get(index);
    if (!client) {
      client = new GoogleGenAI({
        apiKey: this.states[index].key,
        httpOptions: { headers: this.httpHeaders },
      });
      this.clients.set(index, client);
    }
    return client;
  }

  /** Raw key string for a given index (used to sign Veo download URIs). */
  public keyForIndex(index: number): string {
    return this.states[index]?.key || '';
  }

  /** Number of keys currently marked active (healthy). */
  public activeCount(): number {
    this.refreshStatuses();
    return this.states.filter((s) => s.status === 'active').length;
  }

  private refreshStatuses(now: number = Date.now()): void {
    for (const s of this.states) {
      if (s.status === 'cooling' && s.cooldownUntil <= now) {
        s.status = 'active';
        s.cooldownUntil = 0;
      }
    }
  }

  /**
   * Pick the next healthy key (round-robin). Falls back to the key whose
   * cooldown expires soonest when every key is cooling down.
   */
  public acquire(): KeyState {
    const now = Date.now();
    this.refreshStatuses(now);

    const n = this.states.length;
    if (n === 0) throw new Error('No Gemini API keys are configured on the server.');

    for (let i = 0; i < n; i++) {
      const s = this.states[this.cursor % n];
      this.cursor = (this.cursor + 1) % n;
      if (s.status === 'active') return s;
    }

    // All keys are cooling/invalid — return the one recovering first so the
    // caller can still attempt it (last-resort behaviour).
    const candidates = this.states.filter((s) => s.status === 'cooling');
    if (!candidates.length) {
      // Everything is invalid — reset flags and try again rather than hard-fail.
      for (const s of this.states) {
        s.status = 'active';
        s.cooldownUntil = 0;
      }
      return this.states[this.cursor++ % n];
    }
    return candidates.sort((a, b) => a.cooldownUntil - b.cooldownUntil)[0];
  }

  /** Register a successful use of a key. */
  public reportSuccess(state: KeyState): void {
    state.status = 'active';
    state.cooldownUntil = 0;
    state.errorCount = 0;
    state.successCount += 1;
    state.lastUsedAt = Date.now();
  }

  /** Register a failed use of a key and schedule its cooldown. */
  public reportError(state: KeyState, err: any): void {
    state.errorCount += 1;
    state.lastUsedAt = Date.now();
    state.lastError = String(err?.message || err || 'unknown error').slice(0, 240);

    if (isInvalidKeyError(err)) {
      state.status = 'invalid';
      state.cooldownUntil = Date.now() + MAX_COOLDOWN_BACKOFF_MS;
      console.warn(`[KEY ROTATOR] Key #${state.index} (${maskKey(state.key)}) marked INVALID: ${state.lastError}`);
      return;
    }

    let cooldown = DEFAULT_QUOTA_COOLDOWN_MS;
    if (isQuotaError(err)) {
      const retry = parseRetryDelayMs(err);
      cooldown = retry ? Math.max(retry + 500, 5_000) : DEFAULT_QUOTA_COOLDOWN_MS;
    } else if (isOverloadError(err)) {
      cooldown = OVERLOAD_COOLDOWN_MS;
    } else {
      cooldown = 15_000;
    }
    cooldown = Math.min(cooldown, MAX_COOLDOWN_BACKOFF_MS);

    state.status = 'cooling';
    state.cooldownUntil = Date.now() + cooldown;
    console.warn(
      `[KEY ROTATOR] Key #${state.index} (${maskKey(state.key)}) cooling for ${Math.round(cooldown / 1000)}s after error: ${state.lastError}`
    );
  }

  /**
   * Run an atomic Gemini operation with automatic key rotation.
   *
   * @param label  Human-readable operation name (for logs).
   * @param op     Callback receiving the client + key state. MUST perform the
   *               whole atomic unit (e.g. upload + poll + generate) so the same
   *               key is used throughout one attempt.
   * @param maxAttempts  Max key rotations (defaults to the number of keys).
   */
  public async run<T>(
    label: string,
    op: (client: GoogleGenAI, state: KeyState) => Promise<T>,
    maxAttempts?: number
  ): Promise<T> {
    const n = this.states.length;
    if (n === 0) throw new Error('No Gemini API keys are configured on the server.');
    const attempts = Math.max(1, Math.min(maxAttempts || n, n));

    let lastErr: any;
    for (let attempt = 0; attempt < attempts; attempt++) {
      const state = this.acquire();
      try {
        const result = await op(this.clientFor(state.index), state);
        this.reportSuccess(state);
        return result;
      } catch (err: any) {
        lastErr = err;
        if (isRotatableError(err)) {
          this.reportError(state, err);
          console.warn(`[KEY ROTATOR] ${label}: attempt ${attempt + 1}/${attempts} failed, rotating key...`);
          continue;
        }
        // Non-rotatable (e.g. invalid prompt / missing file) — surface it.
        this.reportError(state, err);
        throw err;
      }
    }

    throw new Error(
      `All ${n} Gemini key(s) are exhausted or unavailable for "${label}". Last error: ${
        lastErr?.message || lastErr
      }`
    );
  }

  /** Snapshot for the /api/keys/status endpoint. */
  public status(): RotatorStatus {
    const now = Date.now();
    this.refreshStatuses(now);
    const keys = this.states.map((s) => ({
      index: s.index,
      label: maskKey(s.key),
      status: s.status,
      cooldownRemainingMs: s.status === 'cooling' ? Math.max(0, s.cooldownUntil - now) : 0,
      errorCount: s.errorCount,
      successCount: s.successCount,
      lastError: s.lastError,
    }));
    return {
      totalKeys: keys.length,
      activeKeys: keys.filter((k) => k.status === 'active').length,
      coolingKeys: keys.filter((k) => k.status === 'cooling').length,
      invalidKeys: keys.filter((k) => k.status === 'invalid').length,
      keys,
    };
  }
}

export const geminiRotator = new GeminiKeyRotator();
