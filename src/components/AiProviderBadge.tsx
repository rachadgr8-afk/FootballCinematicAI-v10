import React, { useEffect, useState } from 'react';
import { Cpu, Eye, Sparkles } from 'lucide-react';
import { safeFetchJson } from '../config/api';

interface ProviderState {
  configured: boolean;
  status: 'executed' | 'failed' | 'not_configured';
  model: string;
  httpStatus: number;
  latencyMs: number;
  error?: string;
}

interface AiStatus {
  providers: {
    openrouter: { configured: boolean; visionModel: string; textModel: string };
    deepseek: { configured: boolean; model: string };
  };
  lastCalls: Record<string, ProviderState | undefined>;
}

const tone = (s?: ProviderState) => {
  if (!s || s.status === 'not_configured') return 'bg-slate-800/80 border-slate-700 text-slate-400';
  if (s.status === 'executed') return 'bg-emerald-950/70 border-emerald-500/40 text-emerald-300';
  return 'bg-amber-950/70 border-amber-500/40 text-amber-300';
};

/**
 * Live badge for the OPTIONAL AI providers (OpenRouter vision/text + DeepSeek).
 * Reports executed / failed / not_configured from the secret-free /api/ai/status
 * endpoint. The app never requires these providers.
 */
export const AiProviderBadge: React.FC = () => {
  const [status, setStatus] = useState<AiStatus | null>(null);
  const [open, setOpen] = useState(false);

  const load = async () => {
    try { setStatus(await safeFetchJson<AiStatus>('/api/ai/status', { method: 'GET' })); } catch { /* keep previous */ }
  };

  useEffect(() => { load(); const id = setInterval(load, 6000); return () => clearInterval(id); }, []);

  const vision = status?.lastCalls?.['openrouter:vision'];
  const text = status?.lastCalls?.['openrouter:text'];
  const ds = status?.lastCalls?.['deepseek:text'];
  const anyConfigured = Boolean(status?.providers?.openrouter?.configured || status?.providers?.deepseek?.configured);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={`px-2.5 py-1 rounded-xl border text-[11px] font-bold flex items-center gap-1.5 transition ${
          anyConfigured ? 'bg-indigo-950/60 border-indigo-500/40 text-indigo-300' : 'bg-slate-800/80 border-slate-700 text-slate-400'
        }`}
        title="Optional AI providers (OpenRouter / DeepSeek)"
      >
        <Cpu className="w-3.5 h-3.5" />
        {anyConfigured ? 'AI: on' : 'AI: off'}
      </button>

      {open && (
        <div className="absolute right-0 mt-2 z-50 w-80 max-w-[90vw] bg-slate-900 border border-slate-700 rounded-xl shadow-2xl p-3 text-left">
          <span className="text-xs font-black text-white uppercase tracking-wider">Optional AI Providers</span>
          <p className="text-[10px] text-slate-400 my-1.5 leading-relaxed">
            Real frames are sent to the OpenRouter VLM. DeepSeek refines the plan/commentary. Both are optional —
            the render falls back to the local engine when a provider is missing or fails.
          </p>

          {[
            { label: 'OpenRouter Vision', icon: Eye, s: vision, cfg: status?.providers?.openrouter },
            { label: 'OpenRouter Text', icon: Sparkles, s: text, cfg: status?.providers?.openrouter },
            { label: 'DeepSeek', icon: Cpu, s: ds, cfg: status?.providers?.deepseek },
          ].map((row) => (
            <div key={row.label} className={`mt-1.5 flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 ${tone(row.s)}`}>
              <div className="flex items-center gap-2 min-w-0">
                <row.icon className="w-3.5 h-3.5 shrink-0" />
                <span className="text-[11px] font-bold truncate">{row.label}</span>
              </div>
              <span className="text-[10px] font-mono shrink-0">
                {row.s ? `${row.s.status}${row.s.httpStatus ? ` · ${row.s.httpStatus}` : ''}${row.s.latencyMs ? ` · ${row.s.latencyMs}ms` : ''}` : 'idle'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
