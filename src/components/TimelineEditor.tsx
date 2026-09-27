import React, { useState } from 'react';
import {
  Edit3,
  Sliders,
  Sparkles,
  Crop,
  Gauge,
  Type,
  Layers,
  Check,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { EditPlan, TimelineClip } from '../types/football';
import { videoRenderEngine } from '../services/videoEngine';

interface TimelineEditorProps {
  editPlan: EditPlan;
  onUpdatePlan: (updated: EditPlan) => void;
  onClose?: () => void;
}

export const TimelineEditor: React.FC<TimelineEditorProps> = ({
  editPlan,
  onUpdatePlan,
  onClose,
}) => {
  const [selectedClipIdx, setSelectedClipIdx] = useState<number>(0);
  const [isColorGradeOpen, setIsColorGradeOpen] = useState<boolean>(false);

  const activeClip = editPlan.timeline[selectedClipIdx] || editPlan.timeline[0];

  const handleClipFieldChange = (field: keyof TimelineClip, value: any) => {
    const newTimeline = editPlan.timeline.map((clip, idx) => {
      if (idx === selectedClipIdx) {
        return { ...clip, [field]: value };
      }
      return clip;
    });

    const updatedPlan: EditPlan = {
      ...editPlan,
      timeline: newTimeline,
    };

    onUpdatePlan(updatedPlan);
    videoRenderEngine.setEditPlan(updatedPlan);
    videoRenderEngine.renderFrame(activeClip.output_start);
  };

  const handleColorGradeChange = (field: keyof EditPlan['color_grade'], value: number) => {
    const updatedPlan: EditPlan = {
      ...editPlan,
      color_grade: {
        ...editPlan.color_grade,
        [field]: value,
      },
    };
    onUpdatePlan(updatedPlan);
    videoRenderEngine.setEditPlan(updatedPlan);
    videoRenderEngine.renderFrame(videoRenderEngine.getCurrentTime());
  };

  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-4 sm:p-5 text-slate-100 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-emerald-400" />
          <h3 className="font-bold text-base text-white">Manual Edit Studio (64s Timeline)</h3>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="px-3 py-1 text-xs font-semibold rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 flex items-center gap-1 transition"
          >
            <Check className="w-3.5 h-3.5" />
            Done
          </button>
        )}
      </div>

      {/* 64-Second Story Strip Overview */}
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
          Story Structure Sequence (11 Segments)
        </span>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-11 gap-1.5 overflow-x-auto py-1">
          {editPlan.timeline.map((clip, idx) => {
            const isSelected = idx === selectedClipIdx;
            const isClimax = clip.output_start >= 44.5 && clip.output_start <= 49.5;
            return (
              <button
                key={idx}
                onClick={() => {
                  setSelectedClipIdx(idx);
                  videoRenderEngine.seek(clip.output_start);
                }}
                className={`p-2 rounded-xl text-left transition flex flex-col justify-between border min-w-[70px] ${
                  isSelected
                    ? 'bg-emerald-950/80 border-emerald-400 ring-2 ring-emerald-500/30'
                    : isClimax
                    ? 'bg-red-950/40 border-red-800/80 hover:border-red-600'
                    : 'bg-slate-800/60 border-slate-700/60 hover:bg-slate-800'
                }`}
              >
                <div className="flex items-center justify-between text-[10px] font-bold">
                  <span className={isSelected ? 'text-emerald-300' : 'text-slate-400'}>
                    #{idx + 1}
                  </span>
                  {clip.veo_needed && (
                    <span title="Veo Shot">
                      <Sparkles className="w-2.5 h-2.5 text-amber-400" />
                    </span>
                  )}
                </div>
                <div className="text-[10px] font-bold text-white truncate my-0.5">
                  {clip.output_start}s - {clip.output_end}s
                </div>
                <div className="text-[9px] text-slate-400 truncate">
                  {clip.action.split(':')[0]}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Detailed Segment Fine-Tuner */}
      {activeClip && (
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <span className="text-sm font-black text-white">
                Segment #{selectedClipIdx + 1} ({activeClip.output_start}s - {activeClip.output_end}s)
              </span>
            </div>

            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={activeClip.veo_needed}
                  onChange={(e) => handleClipFieldChange('veo_needed', e.target.checked)}
                  className="rounded text-emerald-500 focus:ring-emerald-400"
                />
                <Sparkles className="w-3 h-3 text-amber-400" />
                <span className="text-[11px] font-bold">Veo 3.1 Shot</span>
              </label>
            </div>
          </div>

          <div className="text-xs text-slate-400 bg-slate-900/60 p-2 rounded-lg border border-slate-800">
            <span className="font-semibold text-slate-200">Director Note:</span> {activeClip.action}
          </div>

          {/* Controls Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 pt-1">
            {/* Speed Ramping */}
            <div className="flex flex-col gap-1 bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-between text-xs text-slate-300">
                <span className="flex items-center gap-1">
                  <Gauge className="w-3.5 h-3.5 text-emerald-400" />
                  Speed Ramp
                </span>
                <span className="font-mono text-emerald-400 font-bold">{activeClip.speed}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.05"
                value={activeClip.speed}
                onChange={(e) => handleClipFieldChange('speed', parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
              <div className="flex justify-between text-[9px] text-slate-500">
                <span>0.5x (Slo-mo)</span>
                <span>1.0x</span>
                <span>2.0x (Hype)</span>
              </div>
            </div>

            {/* Smart 9:16 Crop X Anchor (Subject Tracking) */}
            <div className="flex flex-col gap-1 bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-between text-xs text-slate-300">
                <span className="flex items-center gap-1">
                  <Crop className="w-3.5 h-3.5 text-cyan-400" />
                  9:16 Horizontal Focus
                </span>
                <span className="font-mono text-cyan-400 font-bold">
                  {Math.round(activeClip.crop_x * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.02"
                value={activeClip.crop_x}
                onChange={(e) => handleClipFieldChange('crop_x', parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
              <div className="flex justify-between text-[9px] text-slate-500">
                <span>Left Wing</span>
                <span>Center Goal</span>
                <span>Right Wing</span>
              </div>
            </div>

            {/* Dynamic Zoom Push-In */}
            <div className="flex flex-col gap-1 bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-between text-xs text-slate-300">
                <span className="flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5 text-purple-400" />
                  Dynamic Push-In
                </span>
                <span className="font-mono text-purple-400 font-bold">
                  {activeClip.zoom_end.toFixed(2)}x
                </span>
              </div>
              <input
                type="range"
                min="1.0"
                max="1.45"
                step="0.02"
                value={activeClip.zoom_end}
                onChange={(e) => handleClipFieldChange('zoom_end', parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-400"
              />
              <div className="flex justify-between text-[9px] text-slate-500">
                <span>Static (1.0x)</span>
                <span>Medium (1.2x)</span>
                <span>Intense (1.45x)</span>
              </div>
            </div>

            {/* Transition Dropdown */}
            <div className="flex flex-col gap-1 bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/80">
              <label className="text-xs text-slate-300 font-semibold">Transition</label>
              <select
                value={activeClip.transition}
                onChange={(e) => handleClipFieldChange('transition', e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
              >
                <option value="hard_cut">Hard Cut (Instant Beat Drop)</option>
                <option value="fade">Cinematic Fade (Black dissolve)</option>
                <option value="flash">Impact Flash (White floodlight blast)</option>
                <option value="directional_blur">Directional Whip Blur</option>
                <option value="match_cut">Match Cut (Action link)</option>
              </select>
            </div>

            {/* Caption / Kinetic Typography */}
            <div className="flex flex-col gap-1 bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/80 sm:col-span-2">
              <label className="text-xs text-slate-300 font-semibold flex items-center gap-1">
                <Type className="w-3.5 h-3.5 text-amber-400" />
                Kinetic Caption Overlay
              </label>
              <input
                type="text"
                placeholder="e.g. UNSTOPPABLE STRIKE"
                value={activeClip.text}
                onChange={(e) => handleClipFieldChange('text', e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
              />
            </div>
          </div>
        </div>
      )}

      {/* Color Grade Accordion */}
      <div className="border border-slate-800 rounded-xl overflow-hidden">
        <button
          onClick={() => setIsColorGradeOpen(!isColorGradeOpen)}
          className="w-full flex items-center justify-between p-3 bg-slate-800/50 hover:bg-slate-800 transition text-left"
        >
          <span className="text-xs font-bold text-slate-200">
            Documentary Color Grading & Film Grain
          </span>
          {isColorGradeOpen ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </button>

        {isColorGradeOpen && (
          <div className="p-4 bg-slate-950 grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs text-slate-300">
                <span>Contrast</span>
                <span className="font-mono text-emerald-400">
                  {editPlan.color_grade.contrast.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.8"
                max="1.6"
                step="0.05"
                value={editPlan.color_grade.contrast}
                onChange={(e) => handleColorGradeChange('contrast', parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
            </div>

            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs text-slate-300">
                <span>Saturation</span>
                <span className="font-mono text-cyan-400">
                  {editPlan.color_grade.saturation.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.6"
                max="1.5"
                step="0.05"
                value={editPlan.color_grade.saturation}
                onChange={(e) => handleColorGradeChange('saturation', parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
            </div>

            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs text-slate-300">
                <span>Film Grain</span>
                <span className="font-mono text-purple-400">
                  {editPlan.color_grade.grain.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.5"
                step="0.05"
                value={editPlan.color_grade.grain}
                onChange={(e) => handleColorGradeChange('grain', parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-400"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
