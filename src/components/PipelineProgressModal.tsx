import React from 'react';
import {
  Upload,
  Cpu,
  Film,
  Sparkles,
  Scissors,
  CheckCircle2,
  FileCheck,
  Zap,
} from 'lucide-react';
import { PipelineStage } from '../types/football';

interface PipelineProgressModalProps {
  stage: PipelineStage;
  subStatus: string;
  progressPercent: number;
}

const STAGES_CONFIG = [
  { id: 'uploading', label: 'Uploading...', icon: Upload, desc: 'Processing raw football video frames' },
  { id: 'analyzing', label: 'Analyzing with Gemini...', icon: Cpu, desc: 'Detecting players, ball, dribbles, passes & emotional intensity' },
  { id: 'building_story', label: 'Building story...', icon: Film, desc: 'Structuring 64-second vertical narrative & climax timing' },
  { id: 'generating_veo', label: 'Generating cinematic shots...', icon: Sparkles, desc: 'Veo 3.1 neural football cinematography synthesis' },
  { id: 'editing', label: 'Editing...', icon: Scissors, desc: 'Local video engine: smart 9:16 crop, zoom push-in, sound sync' },
  { id: 'ai_review', label: 'AI reviewing...', icon: FileCheck, desc: 'Gemini Quality Control checking pacing and climax impact' },
  { id: 'final_render', label: 'Final rendering...', icon: CheckCircle2, desc: 'Assembling 1080x1920 64-second master short' },
];

export const PipelineProgressModal: React.FC<PipelineProgressModalProps> = ({
  stage,
  subStatus,
  progressPercent,
}) => {
  const currentStageIndex = STAGES_CONFIG.findIndex((s) => s.id === stage);

  return (
    <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-7 shadow-2xl flex flex-col gap-6 text-slate-100 animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center border border-emerald-500/30">
              <Zap className="w-5 h-5 fill-current animate-pulse" />
            </div>
            <div>
              <h3 className="font-black text-sm text-white tracking-wider">
                FOOTBALL CINEMATIC AI
              </h3>
              <p className="text-xs text-slate-400">Autonomous Director Pipeline</p>
            </div>
          </div>
          <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-400">
            {progressPercent}%
          </span>
        </div>

        {/* Global Progress Bar */}
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden relative">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 transition-all duration-300 rounded-full"
            style={{ width: `${Math.max(5, progressPercent)}%` }}
          />
        </div>

        {/* Stages Checklist */}
        <div className="flex flex-col gap-3 py-1">
          {STAGES_CONFIG.map((item, index) => {
            const isCompleted = currentStageIndex > index;
            const isCurrent = currentStageIndex === index;
            const isPending = currentStageIndex < index;
            const Icon = item.icon;

            return (
              <div
                key={item.id}
                className={`flex items-start gap-3 p-2.5 rounded-xl transition-all ${
                  isCurrent
                    ? 'bg-slate-800/80 border border-emerald-500/40 shadow-sm'
                    : isCompleted
                    ? 'opacity-80'
                    : 'opacity-40'
                }`}
              >
                <div
                  className={`w-7 h-7 rounded-lg flex items-center justify-center mt-0.5 shrink-0 transition-colors ${
                    isCompleted
                      ? 'bg-emerald-500 text-slate-950'
                      : isCurrent
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 animate-pulse'
                      : 'bg-slate-800 text-slate-500'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-xs font-bold ${
                        isCurrent ? 'text-emerald-300' : isCompleted ? 'text-white' : 'text-slate-400'
                      }`}
                    >
                      {item.label}
                    </span>
                    {isCompleted && (
                      <span className="text-[10px] font-semibold text-emerald-400">Complete</span>
                    )}
                    {isCurrent && (
                      <span className="text-[10px] font-semibold text-cyan-400 animate-pulse">
                        In Progress
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-400 truncate mt-0.5">
                    {isCurrent ? subStatus || item.desc : item.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Live Sub-status footer */}
        <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3 text-center">
          <p className="text-xs text-slate-300 font-mono flex items-center justify-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            {subStatus || 'Directing 64-second cinematic football masterpiece...'}
          </p>
        </div>
      </div>
    </div>
  );
};
