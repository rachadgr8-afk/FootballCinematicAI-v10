import React from 'react';
import {
  ShieldCheck,
  CheckCircle,
  AlertCircle,
  TrendingUp,
  Sparkles,
  X,
} from 'lucide-react';
import { QCReview } from '../types/football';

interface GeminiQCPanelProps {
  review: QCReview;
  onClose: () => void;
}

export const GeminiQCPanel: React.FC<GeminiQCPanelProps> = ({ review, onClose }) => {
  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-4 sm:p-5 text-slate-100 flex flex-col gap-4 animate-in fade-in duration-200">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center border border-emerald-500/30">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <h4 className="font-bold text-sm text-white">Gemini Editorial Quality Control</h4>
            <p className="text-[11px] text-slate-400">Sports Director Rigorous Review</p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Scores & Verdict */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase font-bold text-slate-400">Verdict</span>
          <span className="text-xs font-black text-emerald-400 mt-1 text-center">
            {review.qc_verdict.replace(/_/g, ' ')}
          </span>
        </div>

        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase font-bold text-slate-400">Pacing Score</span>
          <div className="flex items-center gap-1 mt-1">
            <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-sm font-black text-white">{review.pacing_score}/10</span>
          </div>
        </div>

        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase font-bold text-slate-400">Cinematic</span>
          <div className="flex items-center gap-1 mt-1">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-sm font-black text-white">{review.cinematic_score}/10</span>
          </div>
        </div>
      </div>

      {/* Director Critique */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3">
        <span className="text-[11px] font-bold text-emerald-400 block mb-1">
          Chief Sports Editor Verdict:
        </span>
        <p className="text-xs text-slate-300 leading-relaxed italic">
          "{review.overall_critique}"
        </p>
      </div>

      {/* Corrections List */}
      <div className="flex flex-col gap-2">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
          AI Auto-Applied Optimizations ({review.corrections.length})
        </span>

        {review.corrections.length === 0 ? (
          <p className="text-xs text-slate-500 py-2">
            No flaws detected. Source timing and climax perfectly locked.
          </p>
        ) : (
          <div className="flex flex-col gap-2">
            {review.corrections.map((corr, idx) => (
              <div
                key={idx}
                className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-2.5 flex items-start gap-2.5"
              >
                <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="text-white">Segment #{corr.timeline_index + 1}</span>
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-cyan-300">
                      {corr.change.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-0.5">{corr.reason}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
