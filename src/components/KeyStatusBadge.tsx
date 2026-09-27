import React, { useEffect, useState } from 'react';
import { KeyRound, RefreshCw, CheckCircle2, AlertTriangle, XCircle, Loader2 } from 'lucide-react';
import { safeFetchJson } from '../config/api';

interface KeyEntry {
  index: number;
  label: string;
  status: 'active' | 'cooling' | 'invalid';
  cooldownRemainingMs: number;
  errorCount: number;
  successCount: number;
  lastError?: string;
}

interface KeysStatus {
  success?: boolean;
  totalKeys: number;
  activeKeys: number;
  coolingKeys: number;
  invalidKeys: number;
  keys: KeyEntry[];
}

const statusColor = (s: KeyEntry['status']) =>
  s === 'active' ? 'text-emerald-400' : s === 'cooling' ? 'text-amber-400' : 'text-red-400';

const statusDot = (s: KeyEntry['status']) =>
  s === 'active' ? 'bg-emerald-400' : s === 'cooling' ? 'bg-amber-400' : 'bg-red-400';

const statusIcon = (s: KeyEntry['status']) =>
  s === 'active' ? CheckCircle2 : s === 'cooling' ? AlertTriangle : XCircle;

/**
 * Live badge for the Gemini API key rotation pool.
 * Polls /api/keys/status and lets the operator confirm every configured key is
 * healthy and that the rotation is working (keys flip to "cooling" on a quota
 * hit and recover automatically).
 */
export const KeyStatusBadge: React.FC = () => {
  const [status, setStatus] = useState<KeysStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(false);

  const load = async () => {
    try {
      const data = await safeFetchJson<KeysStatus>('/api/keys/status', { method: 'GET' });
      setStatus(data);
      setError(false);
    } catch {
      setError(true);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  const reload = async () => {
    setRefreshing(true);
    try {
      await safeFetchJson('/api/keys/reload', { method: 'POST' });
    } catch {
      /* ignore */
    }
    await load();
    setRefreshing(false);
  };

  if (!status && !error) {
    return (
      <span className="px-2.5 py-1 rounded-xl bg-slate-800 border border-slate-700 text-[11px] text-slate-400 flex items-center gap-1.5">
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        Keys…
      </span>
    );
  }

  const total = status?.totalKeys ?? 0;
  const active = status?.activeKeys ?? 0;
  const cooling = status?.coolingKeys ?? 0;
  const invalid = status?.invalidKeys ?? 0;

  const tone =
    error || total === 0
      ? 'bg-red-950/60 border-red-500/50 text-red-300'
      : active === total
      ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300'
      : active === 0
      ? 'bg-red-950/60 border-red-500/50 text-red-300'
      : 'bg-amber-950/60 border-amber-500/40 text-amber-300';

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={`px-2.5 py-1 rounded-xl border text-[11px] font-bold flex items-center gap-1.5 transition ${tone}`}
        title="Gemini API key rotation pool"
      >
        <KeyRound className="w-3.5 h-3.5" />
        {error || total === 0 ? 'Keys: none' : `${active}/${total} keys`}
        {cooling > 0 && <span className="text-[10px]">· {cooling} cooling</span>}
        {invalid > 0 && <span className="text-[10px]">· {invalid} bad</span>}
      </button>

      {open && (
        <div className="absolute right-0 mt-2 z-50 w-80 max-w-[90vw] bg-slate-900 border border-slate-700 rounded-xl shadow-2xl p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-black text-white uppercase tracking-wider">
              Gemini Key Pool
            </span>
            <button
              onClick={reload}
              className="text-slate-400 hover:text-white transition"
              title="Reload keys from environment"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>

          <p className="text-[10px] text-slate-400 mb-2 leading-relaxed">
            Keys are used in automatic round-robin. On a quota (429) or overload (503) hit, a key
            is put on cooldown and the same operation is retried on the next healthy key.
          </p>

          {error && (
            <div className="text-[11px] text-red-300 mb-2">
              Unable to reach /api/keys/status on the current backend.
            </div>
          )}

          <div className="max-h-64 overflow-y-auto flex flex-col gap-1.5">
            {(status?.keys || []).map((k) => {
              const Icon = statusIcon(k.status);
              return (
                <div
                  key={k.index}
                  className="flex items-center justify-between bg-slate-950/70 border border-slate-800 rounded-lg px-2.5 py-1.5"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={`w-1.5 h-1.5 rounded-full ${statusDot(k.status)}`} />
                    <span className="text-[11px] font-mono text-slate-300 truncate">
                      #{k.index + 1} {k.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-[10px] text-slate-500">
                      {k.status === 'cooling' && k.cooldownRemainingMs > 0
                        ? `${Math.ceil(k.cooldownRemainingMs / 1000)}s`
                        : `✓${k.successCount}`}
                    </span>
                    <Icon className={`w-3.5 h-3.5 ${statusColor(k.status)}`} />
                  </div>
                </div>
              );
            })}
            {!error && (status?.keys?.length || 0) === 0 && (
              <div className="text-[11px] text-red-300">
                No keys configured. Set <code>GEMINI_API_KEYS</code> on the server.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
