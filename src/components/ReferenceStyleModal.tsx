import React, { useRef, useState } from 'react';
import {
  Film,
  Sparkles,
  Check,
  X,
  Sliders,
  HelpCircle,
  Upload,
} from 'lucide-react';
import { ReferenceStyleProfile, StyleProfile } from '../types/football';

import { safeFetchJson, getApiUrl } from '../config/api';

interface ReferenceStyleModalProps {
  onApplyProfile: (profile: StyleProfile, measured?: ReferenceStyleProfile, localPath?: string) => void;
  onClose: () => void;
  currentProfile: StyleProfile | null;
}

/**
 * Reference Style Match.
 *
 * Upload a reference reel -> the backend MEASURES it frame-by-frame (shot
 * durations, cut density, camera movement, zoom intensity, speed handling,
 * framing mix, typography, transitions, colour, audio and hero-shot structure)
 * and returns a ReferenceStyleProfile. Only those STYLE PARAMETERS drive the
 * Cinematic Director; no timestamps, shot order, frames, logos or watermarks are
 * ever copied into the result.
 */
export const ReferenceStyleModal: React.FC<ReferenceStyleModalProps> = ({
  onApplyProfile,
  onClose,
  currentProfile,
}) => {
  const [refTitle, setRefTitle] = useState('Reference Cinematic Football Reel');
  const [refNotes, setRefNotes] = useState(
    '9:16 emotional football montage; immediate hook; tight player close-ups and football details; hard-cut rhythm with a few narrative dissolves; controlled punch-ins; selective slow motion at the real impact; small white uppercase captions near the lower third; dark cinematic stadium green with protected skin tones.'
  );
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [generatedProfile, setGeneratedProfile] = useState<StyleProfile | null>(currentProfile || null);
  const [measured, setMeasured] = useState<ReferenceStyleProfile | null>(null);
  const [referenceLocalPath, setReferenceLocalPath] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  /** Real measured analysis: uploads the reference reel to the backend. */
  const handleUploadAndAnalyze = async (file: File) => {
    setIsAnalyzing(true);
    setStatus('Uploading the reference video...');
    try {
      const form = new FormData();
      form.append('reference', file);
      const res = await fetch(getApiUrl('/api/reference-style/analyze'), { method: 'POST', body: form });
      const data = await res.json();
      if (data?.styleProfile) {
        setGeneratedProfile(data.styleProfile);
        setMeasured(data.referenceStyleProfile || null);
        setReferenceLocalPath(data.referenceLocalPath || null);
        setStatus(`Measured ${data.referenceStyleProfile?.shot_count ?? 0} shots over ${data.referenceStyleProfile?.duration ?? 0}s.`);
      } else {
        setStatus(data?.error || 'The reference video could not be measured.');
      }
    } catch (err) {
      console.error('Failed to analyze the reference video:', err);
      setStatus('The reference video could not be measured.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  /** Title/notes-only profile (legacy deterministic contract, no upload). */
  const handleAnalyzeReference = async () => {
    setIsAnalyzing(true);
    setStatus(null);
    try {
      const data = await safeFetchJson<{ success: boolean; styleProfile?: StyleProfile; referenceStyleProfile?: ReferenceStyleProfile }>(
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
        setMeasured(data.referenceStyleProfile || null);
      }
    } catch (err) {
      console.error('Failed to analyze reference video:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleApply = () => {
    if (generatedProfile) {
      onApplyProfile(generatedProfile, measured || undefined, referenceLocalPath || undefined);
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl flex flex-col gap-4 text-slate-100 animate-in fade-in duration-200 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center border border-purple-500/30">
              <Film className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-white">Reference Video Style Match</h3>
              <p className="text-[11px] text-slate-400">
                Measure real pacing, framing, colour &amp; sound characteristics
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
            The reference supplies <strong>style parameters only</strong> (average shot duration, cut density,
            zoom intensity, speed variation, text frequency, transition frequency, colour characteristics and
            hero-shot structure). No timestamps, shot order, frames, logos or watermarks are ever copied.
          </p>
        </div>

        {/* Real measurement */}
        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-slate-300">Reference Video File (measured analysis)</label>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleUploadAndAnalyze(file);
            }}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={isAnalyzing}
            className="w-full py-2 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 font-semibold text-xs text-white flex items-center justify-center gap-2 transition disabled:opacity-50 border border-slate-700"
          >
            <Upload className="w-4 h-4" />
            {isAnalyzing ? 'Measuring the reference video...' : 'Upload a reference video'}
          </button>
          {status && <p className="text-[11px] text-emerald-300">{status}</p>}
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
              className="w-full bg-slate-800/80 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>

          <button
            onClick={handleAnalyzeReference}
            disabled={isAnalyzing}
            className="w-full py-2 px-4 rounded-xl bg-purple-600 hover:bg-purple-500 font-semibold text-xs text-white flex items-center justify-center gap-2 transition disabled:opacity-50"
          >
            <Sparkles className="w-4 h-4" />
            {isAnalyzing ? 'Extracting style profile...' : 'Analyze Reference Style (title only)'}
          </button>
        </div>

        {/* Generated Style Profile */}
        {generatedProfile && (
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 flex flex-col gap-2">
            <span className="text-xs font-bold text-purple-400 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" />
              {measured ? 'Measured Style Profile' : 'Generated Style Profile'}
            </span>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Avg Shot Duration</span>
                <span className="font-bold text-white">{generatedProfile.average_shot_duration}s</span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Cut Density</span>
                <span className="font-bold text-white">
                  {measured ? `${measured.cut_density}/s` : `${Math.round(generatedProfile.transition_frequency * 100)}%`}
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Zoom Intensity</span>
                <span className="font-bold text-white">{Math.round(generatedProfile.zoom_intensity * 100)}%</span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Subject Shots</span>
                <span className="font-bold text-white">
                  {measured ? `${Math.round(measured.subject_shot_ratio * 100)}%` : '—'}
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Slow Motion</span>
                <span className="font-bold text-white">
                  {Math.round((measured?.slow_motion_shot_ratio ?? generatedProfile.slow_motion_frequency) * 100)}%
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">BPM Target</span>
                <span className="font-bold text-amber-400">{generatedProfile.recommended_bpm || 128}</span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800 col-span-2 sm:col-span-3">
                <span className="text-[10px] text-slate-400 block">Color Mood</span>
                <span className="font-semibold text-emerald-400 block break-words">
                  {generatedProfile.color_style}
                </span>
              </div>
              {measured && (
                <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800 col-span-2 sm:col-span-3">
                  <span className="text-[10px] text-slate-400 block">Measured Structure</span>
                  <span className="text-[11px] text-slate-300 block">
                    {measured.shot_count} shots · {measured.duration}s · hard-cut{' '}
                    {Math.round((measured.transition_weights?.hard_cut ?? 0) * 100)}% · cut/impact sync{' '}
                    {Math.round((measured.audio?.cut_impact_sync_ratio ?? 0) * 100)}% · text{' '}
                    {measured.text_per_shot}/shot
                  </span>
                </div>
              )}
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
