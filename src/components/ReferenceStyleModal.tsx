import React, { useState } from 'react';
import {
  Film,
  Sparkles,
  Check,
  X,
  Sliders,
  HelpCircle,
} from 'lucide-react';
import { StyleProfile } from '../types/football';

import { safeFetchJson } from '../config/api';

interface ReferenceStyleModalProps {
  onApplyProfile: (profile: StyleProfile) => void;
  onClose: () => void;
  currentProfile: StyleProfile | null;
}

export const ReferenceStyleModal: React.FC<ReferenceStyleModalProps> = ({
  onApplyProfile,
  onClose,
  currentProfile,
}) => {
  const [refTitle, setRefTitle] = useState('Reference Cinematic Football Reel');
  const [refNotes, setRefNotes] = useState(
    '9:16 emotional football montage; immediate hook; tight player close-ups and football details; 1-5 second shots; hard cuts; controlled punch-ins; selective slow motion; small white uppercase captions near the lower third; dark cinematic stadium green; one restrained flash at a verified impact; strong climax and emotional outro.'
  );
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [generatedProfile, setGeneratedProfile] = useState<StyleProfile | null>(
    currentProfile || null
  );

  const handleAnalyzeReference = async () => {
    setIsAnalyzing(true);
    try {
      const data = await safeFetchJson<{ success: boolean; styleProfile?: StyleProfile }>(
        '/api/analyze-reference',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            referenceTitle: refTitle,
            referenceDescription: refNotes,
          }),
        }
      );
      if (data.styleProfile) {
        setGeneratedProfile(data.styleProfile);
      }
    } catch (err) {
      console.error('Failed to analyze reference video:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleApply = () => {
    if (generatedProfile) {
      onApplyProfile(generatedProfile);
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl flex flex-col gap-4 text-slate-100 animate-in fade-in duration-200">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center border border-purple-500/30">
              <Film className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-white">Reference Video Style Match</h3>
              <p className="text-[11px] text-slate-400">
                Extract editing rhythm, pacing & color profile
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Info Callout */}
        <div className="flex items-start gap-2 bg-purple-950/40 border border-purple-800/60 rounded-xl p-3 text-xs text-purple-200">
          <HelpCircle className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
          <p>
            Gemini copies purely editorial rhythm (average shot length, zoom curves, transition
            frequencies) — zero copyrighted logos, watermarks, or footage are cloned.
          </p>
        </div>

        {/* Inputs */}
        <div className="flex flex-col gap-3">
          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-1">
              Reference Video Title or Style Inspiration
            </label>
            <input
              type="text"
              value={refTitle}
              onChange={(e) => setRefTitle(e.target.value)}
              className="w-full bg-slate-800/80 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-300 block mb-1">
              Editorial Characteristics / Visual Notes
            </label>
            <textarea
              rows={3}
              value={refNotes}
              onChange={(e) => setRefNotes(e.target.value)}
              placeholder="e.g. Ultra-fast cuts during dribbles, slow-mo celebration, high contrast, dramatic violin crescendo..."
              className="w-full bg-slate-800/80 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>

          <button
            onClick={handleAnalyzeReference}
            disabled={isAnalyzing}
            className="w-full py-2 px-4 rounded-xl bg-purple-600 hover:bg-purple-500 font-semibold text-xs text-white flex items-center justify-center gap-2 transition disabled:opacity-50"
          >
            <Sparkles className="w-4 h-4" />
            {isAnalyzing ? 'Extracting Style Profile with Gemini...' : 'Analyze Reference Style'}
          </button>
        </div>

        {/* Generated Style Profile */}
        {generatedProfile && (
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 flex flex-col gap-2">
            <span className="text-xs font-bold text-purple-400 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" />
              Generated Style Profile
            </span>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Avg Shot Duration</span>
                <span className="font-bold text-white">
                  {generatedProfile.average_shot_duration}s
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Zoom Intensity</span>
                <span className="font-bold text-white">
                  {Math.round(generatedProfile.zoom_intensity * 100)}%
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Cut Frequency</span>
                <span className="font-bold text-white">
                  {Math.round(generatedProfile.transition_frequency * 100)}%
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800 col-span-2">
                <span className="text-[10px] text-slate-400 block">Color Mood</span>
                <span className="font-semibold text-emerald-400 truncate block">
                  {generatedProfile.color_style}
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">BPM Target</span>
                <span className="font-bold text-amber-400">
                  {generatedProfile.recommended_bpm || 128}
                </span>
              </div>
            </div>

            <button
              onClick={handleApply}
              className="mt-2 w-full py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 transition"
            >
              <Check className="w-4 h-4" />
              Apply Style Profile to Project
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
