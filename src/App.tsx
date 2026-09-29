import React, { useState, useEffect, useRef } from 'react';
import { AiProviderBadge } from './components/AiProviderBadge';
import {
  Zap,
  Film,
  Sparkles,
  Upload,
  Play,
  RotateCcw,
  Sliders,
  ShieldCheck,
  Download,
  Share2,
  FolderPlus,
  RefreshCw,
  Flame,
  Heart,
  ChevronRight,
  Info,
  Check,
  AlertCircle,
  Video,
  FileCheck2,
  Gauge,
  Type,
  Crop,
  ExternalLink,
  Eye,
} from 'lucide-react';

import {
  AIStyle,
  GenerationTier,
  PipelineStage,
  EditPlan,
  FootballVideoMetadata,
  CinematicMode,
  ReferenceStyleProfile,
  QCReview,
  StyleProfile,
} from './types/football';

import { API_BASE_URL, safeFetchJson, uploadVideo, getApiUrl, resolveMediaUrl } from './config/api';
import { runCinematicRender } from './services/renderJob';
import { AndroidPhoneFrame } from './components/AndroidPhoneFrame';
import { AndroidMedia3Player } from './components/AndroidMedia3Player';
import { TimelineEditor } from './components/TimelineEditor';
import { GeminiQCPanel } from './components/GeminiQCPanel';
import { ReferenceStyleModal } from './components/ReferenceStyleModal';
import { KeyStatusBadge } from './components/KeyStatusBadge';

export default function App() {
  // Preset match videos
  const [presets, setPresets] = useState<any[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<any>(null);

  // Settings
  const [selectedStyle, setSelectedStyle] = useState<AIStyle>('REFERENCE CINEMATIC REEL');
  const [generationTier, setGenerationTier] = useState<GenerationTier>('AI CINEMATIC');
  const [referenceProfile, setReferenceProfile] = useState<StyleProfile | null>(null);
  // CINEMATIC MODE: STANDARD | PRO | REFERENCE. REFERENCE mode consumes the
  // MEASURED profile of a reference video (style parameters only).
  const [cinematicMode, setCinematicMode] = useState<CinematicMode>('STANDARD');
  const [referenceStyleProfile, setReferenceStyleProfile] = useState<ReferenceStyleProfile | null>(null);
  const [referenceLocalPath, setReferenceLocalPath] = useState<string | null>(null);

  // Active Video URL in Player
  const [activePlayerUrl, setActivePlayerUrl] = useState<string>(resolveMediaUrl('/videos/football_match.mp4'));
  const [activePlayerPoster, setActivePlayerPoster] = useState<string>(resolveMediaUrl('/videos/poster_10s.jpg'));
  const [activePlayerTitle, setActivePlayerTitle] = useState<string>('El Clasico Final (Live Match Footage)');
  const [isMasterRendered, setIsMasterRendered] = useState<boolean>(false);

  // Testing Pipeline State
  const [isTestRunning, setIsTestRunning] = useState<boolean>(false);
  const [testStatus, setTestStatus] = useState<string>('');

  // Full Pipeline State
  const [isRenderingFull, setIsRenderingFull] = useState<boolean>(false);
  const [renderProgress, setRenderProgress] = useState<{ percent: number; stage: string }>({
    percent: 0,
    stage: 'Idle',
  });

  // Generated Plan & Reviews
  const [editPlan, setEditPlan] = useState<EditPlan | null>(null);
  const [qcReview, setQcReview] = useState<QCReview | null>(null);

  // Human-readable pipeline error (shown inline instead of a blocking alert)
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  // UI Modals & Views
  const [isExpandedView, setIsExpandedView] = useState<boolean>(true);
  const [showTimelineEditor, setShowTimelineEditor] = useState<boolean>(false);
  const [showQCModal, setShowQCModal] = useState<boolean>(false);
  const [showRefModal, setShowRefModal] = useState<boolean>(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load presets on startup
  useEffect(() => {
    safeFetchJson('/api/presets')
      .then((data) => {
        if (data.presets && data.presets.length > 0) {
          setPresets(data.presets);
          setSelectedVideo(data.presets[0]);
          setActivePlayerUrl(resolveMediaUrl(data.presets[0].sourceUrl));
          setActivePlayerPoster(resolveMediaUrl(data.presets[0].posterUrl || '/videos/poster_10s.jpg'));
          setActivePlayerTitle(data.presets[0].title);
        }
      })
      .catch((err) => {
        console.error('Failed to load presets from backend:', err);
      });
  }, []);

  // Handle custom video upload to backend server
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setTestStatus(`Validating ${file.name}...`);
    setIsTestRunning(true);

    try {
      const data = await uploadVideo(file, (progress) => {
        setTestStatus(progress.stage);
      });

      if (!data.success) {
        throw new Error('Upload failed to complete.');
      }

      const uploadedMeta: FootballVideoMetadata = {
        id: data.videoId || `upload-${Date.now()}`,
        title: data.title || file.name,
        description: `Real football footage (${(data.size / (1024 * 1024)).toFixed(1)} MB, ${data.duration ? data.duration.toFixed(1) + 's' : 'HD'})`,
        duration: data.duration || 75.0,
        sourceUrl: resolveMediaUrl(data.videoUrl),
        posterUrl: resolveMediaUrl(data.posterUrl),
        localPath: data.localPath,
        tags: ['Custom Upload', `${data.width || 1920}x${data.height || 1080}`, data.mimeType],
        defaultSubject: 'Striker / Player',
      };

      setSelectedVideo(uploadedMeta);
      setActivePlayerUrl(resolveMediaUrl(data.videoUrl));
      if (data.posterUrl) setActivePlayerPoster(resolveMediaUrl(data.posterUrl));
      setActivePlayerTitle(`Uploaded: ${file.name}`);
      setUploadedFileName(file.name);
      setIsMasterRendered(false);
      setTestStatus(`Video uploaded & verified! Ready for local motion analysis and real FFmpeg rendering.`);
    } catch (err: any) {
      console.error('Upload handler notice:', err.message || err);
      setTestStatus(`Upload notice: ${err.message || 'Could not upload video.'}`);
    } finally {
      setIsTestRunning(false);
      // Reset input value so re-selecting same file works
      if (e.target) {
        e.target.value = '';
      }
    }
  };

  // TEST 1: Extract 5s, 9:16 Crop
  const runTest1 = async () => {
    setIsTestRunning(true);
    setTestStatus(`Executing TEST 1 on ${API_BASE_URL}: FFmpeg 5s 9:16 extraction...`);

    try {
      const data = await safeFetchJson<{
        success: boolean;
        videoUrl: string;
        posterUrl?: string;
        testName?: string;
      }>('/api/test-render-1', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          localPath: selectedVideo?.localPath,
        }),
      });

      if (!data.success) throw new Error('Test 1 render failed');

      setActivePlayerUrl(resolveMediaUrl(data.videoUrl));
      if (data.posterUrl) setActivePlayerPoster(resolveMediaUrl(data.posterUrl));
      setActivePlayerTitle('TEST 1 OUTPUT: 5s 9:16 Vertical (test_output.mp4)');
      setIsMasterRendered(true);
      setTestStatus('TEST 1 PASSED! Real 9:16 MP4 rendered and loaded into Media3 player.');
    } catch (err: any) {
      setTestStatus(`TEST 1 Failed: ${err.message}`);
    } finally {
      setIsTestRunning(false);
    }
  };

  // TEST 2: 10s -> 15s, 0.7x Speed + 10% Zoom
  const runTest2 = async () => {
    setIsTestRunning(true);
    setTestStatus(`Executing TEST 2 on ${API_BASE_URL}: FFmpeg 0.7x speed + zoompan...`);

    try {
      const data = await safeFetchJson<{
        success: boolean;
        videoUrl: string;
        posterUrl?: string;
        testName?: string;
      }>('/api/test-render-2', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          localPath: selectedVideo?.localPath,
        }),
      });

      if (!data.success) throw new Error('Test 2 render failed');

      setActivePlayerUrl(resolveMediaUrl(data.videoUrl));
      if (data.posterUrl) setActivePlayerPoster(resolveMediaUrl(data.posterUrl));
      setActivePlayerTitle('TEST 2 OUTPUT: 0.7x Slo-mo + Zoom (test_effects.mp4)');
      setIsMasterRendered(true);
      setTestStatus('TEST 2 PASSED! Pixels slowed to 0.7x with animated zoom in test_effects.mp4.');
    } catch (err: any) {
      setTestStatus(`TEST 2 Failed: ${err.message}`);
    } finally {
      setIsTestRunning(false);
    }
  };

  // TEST 3: Burned-in Text Overlay
  const runTest3 = async () => {
    setIsTestRunning(true);
    setTestStatus(`Executing TEST 3 on ${API_BASE_URL}: FFmpeg drawtext...`);

    try {
      const data = await safeFetchJson<{
        success: boolean;
        videoUrl: string;
        posterUrl?: string;
        testName?: string;
      }>('/api/test-render-3', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          localPath: selectedVideo?.localPath,
        }),
      });

      if (!data.success) throw new Error('Test 3 render failed');

      setActivePlayerUrl(resolveMediaUrl(data.videoUrl));
      if (data.posterUrl) setActivePlayerPoster(resolveMediaUrl(data.posterUrl));
      setActivePlayerTitle('TEST 3 OUTPUT: Burned-in Text (test_text.mp4)');
      setIsMasterRendered(true);
      setTestStatus('TEST 3 PASSED! Real drawtext rendered inside test_text.mp4.');
    } catch (err: any) {
      setTestStatus(`TEST 3 Failed: ${err.message}`);
    } finally {
      setIsTestRunning(false);
    }
  };

  // COMPLETE REAL PIPELINE: Local motion plan -> Real FFmpeg Execution -> final_video.mp4
  const runCompleteRealPipeline = async (customStyle?: AIStyle) => {
    if (!selectedVideo) return;
    const styleToUse = customStyle || selectedStyle;

    setIsRenderingFull(true);
    setPipelineError(null);
    setRenderProgress({ percent: 5, stage: 'Analyzing local motion of the uploaded footage...' });

    try {
      const versionData = await safeFetchJson<any>('/api/version', { method: 'GET' });
      const backendVersion = String(versionData?.buildVersion || '');
      if (!backendVersion.includes('exceptional-v10')) {
        throw new Error(
          `Backend is incompatible (${backendVersion || 'unknown'}). ` +
            `Contacted backend: ${API_BASE_URL}. Deploy the FootballCinematicAI v10 backend before rendering.`
        );
      }

      // 1. Local motion analysis (no external API)
      const analyzeData = await safeFetchJson<{
        success: boolean;
        editPlan?: EditPlan;
      }>('/api/analyze-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoMetadata: selectedVideo,
          style: styleToUse,
          generationTier,
          referenceStyle: referenceProfile,
          referenceStyleProfile,
          referenceLocalPath,
          cinematicMode,
          trackingEnabled: true,
        }),
      });

      if (!analyzeData.success || !analyzeData.editPlan) {
        throw new Error('Failed to obtain the local edit plan');
      }

      let activePlan: EditPlan = analyzeData.editPlan;
      setEditPlan(activePlan);

      // 2. Real FFmpeg Video Processing
      setRenderProgress({ percent: 15, stage: 'Calling FFmpeg rendering engine...' });

      // Async job: submits immediately, then polls. Never holds a request open
      // for minutes (which is what produced the production "Failed to fetch").
      const renderData = await runCinematicRender(
        {
          localPath: selectedVideo.localPath,
          editPlan: activePlan,
          generationTier,
          musicVolume: 0.24,
          originalVolume: 0.72,
        },
        (p) => setRenderProgress({ percent: p.percent, stage: p.stage })
      );

      if (!renderData.success) {
        throw new Error(renderData.error || 'FFmpeg rendering failed');
      }

      // 3. Local Quality Control review
      try {
        const qcData = await safeFetchJson<{
          success: boolean;
          review?: QCReview;
        }>('/api/qc-review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ outputLocalPath: renderData.localPath, editPlan: activePlan, style: styleToUse }),
        });
        if (qcData.review) {
          setQcReview(qcData.review);

          // One automatic polish pass. Corrections are applied only to the already
          // verified real timeline; no new footage or timestamps are invented.
          if (qcData.review.qc_verdict === 'REVISE_PACING' && Array.isArray(qcData.review.corrections) && qcData.review.corrections.length) {
            const revisedPlan: EditPlan = {
              ...activePlan,
              timeline: activePlan.timeline.map((clip) => {
                const correction = qcData.review!.corrections.find((c) => c.timeline_index === clip.timeline_index);
                if (!correction) return clip;
                return {
                  ...clip,
                  speed: typeof correction.recommended_speed === 'number' ? correction.recommended_speed : clip.speed,
                  crop_x: typeof correction.recommended_crop_x === 'number' ? correction.recommended_crop_x : clip.crop_x,
                  crop_y: typeof correction.recommended_crop_y === 'number' ? correction.recommended_crop_y : clip.crop_y,
                  text: correction.recommended_text || clip.text,
                  transition: (['hard_cut','directional_blur','flash'].includes(correction.recommended_transition || '')
                    ? correction.recommended_transition
                    : clip.transition) as any,
                };
              }),
            };
            activePlan = revisedPlan;
            setEditPlan(revisedPlan);
            setRenderProgress({ percent: 92, stage: 'Local QC found pacing issues — applying one automatic polish pass...' });
            // Same async contract as the primary render: a second full montage
            // can take minutes, so it must NOT block a single HTTP request either.
            const polishData = await runCinematicRender(
              {
                localPath: selectedVideo.localPath,
                editPlan: activePlan,
                generationTier,
                musicVolume: 0.24,
                originalVolume: 0.72,
              },
              (p) => setRenderProgress({ percent: p.percent, stage: p.stage })
            );
            if (polishData.success) {
              renderData.videoUrl = polishData.videoUrl;
              renderData.posterUrl = polishData.posterUrl;
              renderData.duration = polishData.duration;
              renderData.localPath = polishData.localPath;
            }
          }
        }
      } catch (e) {
        console.warn('Visual QC/polish pass skipped:', e);
      }

      // 4. ElevenLabs commentator: generate Arabic voice-over and mux it into the final MP4.
      // The provider key stays on the backend.
      setRenderProgress({ percent: 96, stage: 'AI Football Director is composing a natural Arabic performance...' });
      try {
        const commentaryData = await safeFetchJson<any>('/api/commentary/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            editPlan: activePlan,
            style: styleToUse,
            videoLocalPath: renderData.localPath,
          }),
        });
        if (commentaryData.finalVideoUrl) {
          // Bust the cache: commentary muxing creates a fresh MP4 but keeps a
          // predictable /videos/final_commentary_*.mp4 URL. Never let the player
          // reuse an older cached master.
          renderData.videoUrl = `${commentaryData.finalVideoUrl}${String(commentaryData.finalVideoUrl).includes('?') ? '&' : '?'}t=${Date.now()}`;
        }
      } catch (commentaryErr) {
        console.warn('ElevenLabs commentary skipped:', commentaryErr);
      }

      // 5. Update Player with the final polished real-footage master MP4
      setActivePlayerUrl(resolveMediaUrl(renderData.videoUrl));
      if (renderData.posterUrl) setActivePlayerPoster(resolveMediaUrl(renderData.posterUrl));
      setActivePlayerTitle(`64s Master Cut: final_video.mp4 (${renderData.duration.toFixed(1)}s)`);
      setIsMasterRendered(true);

      setRenderProgress({ percent: 100, stage: 'Master video ready!' });
    } catch (err: any) {
      setPipelineError(err.message || 'The render pipeline failed.');
      setRenderProgress({ percent: 0, stage: `Pipeline error: ${err.message || 'unknown'}` });
    } finally {
      setIsRenderingFull(false);
    }
  };

  const handleOpenVideoInNewTab = () => {
    window.open(getApiUrl(activePlayerUrl), '_blank');
  };

  const handleSaveToDevice = () => {
    const a = document.createElement('a');
    a.href = getApiUrl(activePlayerUrl);
    a.download = `final_video_${Date.now()}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleShare = () => {
    const shareUrl = getApiUrl(activePlayerUrl);
    if (navigator.share) {
      navigator.share({
        title: 'FOOTBALL CINEMATIC AI Master',
        url: shareUrl,
      });
    } else {
      navigator.clipboard.writeText(shareUrl);
      alert('Video link copied to clipboard!');
    }
  };

  const handleReset = () => {
    if (presets.length > 0) {
      setSelectedVideo(presets[0]);
      setActivePlayerUrl(resolveMediaUrl(presets[0].sourceUrl));
      setActivePlayerTitle(presets[0].title);
      setIsMasterRendered(false);
      setEditPlan(null);
      setQcReview(null);
      setShowTimelineEditor(false);
    }
  };

  return (
    <AndroidPhoneFrame
      isExpandedView={isExpandedView}
      onToggleView={() => setIsExpandedView(!isExpandedView)}
      title="FOOTBALL CINEMATIC AI"
    >
      <div className="flex-1 flex flex-col p-3 sm:p-5 gap-4 overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 text-slate-950 flex items-center justify-center font-black shadow-lg shadow-emerald-500/20">
              <Zap className="w-5 h-5 fill-current" />
            </div>
            <div>
              <h1 className="font-black text-sm sm:text-base text-white tracking-wide">
                FOOTBALL CINEMATIC AI
              </h1>
              <p className="text-[11px] text-slate-400">
                Real FFmpeg Video Engine • Local Motion Director • Android Media3
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <AiProviderBadge />
            <KeyStatusBadge />
            {isMasterRendered && (
              <span className="px-2.5 py-1 rounded-xl bg-red-600/90 text-white font-bold text-xs flex items-center gap-1 shadow animate-pulse">
                <Check className="w-3.5 h-3.5" />
                REAL MP4 RENDERED
              </span>
            )}
          </div>
        </div>

        {/* SECTION A: REAL PIPELINE VERIFICATION SUITE */}
        <div className="bg-slate-900 border border-emerald-500/40 rounded-2xl p-4 flex flex-col gap-3 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
              <h3 className="font-black text-xs sm:text-sm text-emerald-400 uppercase tracking-wider">
                Real Video Pipeline Verification Tests
              </h3>
            </div>
            <span className="text-[10px] text-slate-400">FFmpeg 4.4 Engine Live</span>
          </div>

          <p className="text-xs text-slate-300">
            Click any test to run real FFmpeg processing on the selected footage. The actual rendered MP4 loads into the player immediately:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {/* Test 1 */}
            <button
              onClick={runTest1}
              disabled={isTestRunning || isRenderingFull}
              className="p-3 rounded-xl bg-slate-950/80 hover:bg-slate-800 border border-slate-700 text-left transition flex flex-col justify-between gap-1.5 disabled:opacity-50 group hover:border-emerald-400"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white group-hover:text-emerald-300">
                  TEST 1: 5s 9:16 Crop
                </span>
                <Crop className="w-3.5 h-3.5 text-emerald-400" />
              </div>
              <p className="text-[10px] text-slate-400">
                Extracts 0-5s, scales & crops to 1080×1920 (test_output.mp4)
              </p>
            </button>

            {/* Test 2 */}
            <button
              onClick={runTest2}
              disabled={isTestRunning || isRenderingFull}
              className="p-3 rounded-xl bg-slate-950/80 hover:bg-slate-800 border border-slate-700 text-left transition flex flex-col justify-between gap-1.5 disabled:opacity-50 group hover:border-cyan-400"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white group-hover:text-cyan-300">
                  TEST 2: 0.7x Speed + Zoom
                </span>
                <Gauge className="w-3.5 h-3.5 text-cyan-400" />
              </div>
              <p className="text-[10px] text-slate-400">
                Extracts 10-15s, alters PTS to 0.7x, adds 10% zoompan (test_effects.mp4)
              </p>
            </button>

            {/* Test 3 */}
            <button
              onClick={runTest3}
              disabled={isTestRunning || isRenderingFull}
              className="p-3 rounded-xl bg-slate-950/80 hover:bg-slate-800 border border-slate-700 text-left transition flex flex-col justify-between gap-1.5 disabled:opacity-50 group hover:border-amber-400"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white group-hover:text-amber-300">
                  TEST 3: Burned-in Text
                </span>
                <Type className="w-3.5 h-3.5 text-amber-400" />
              </div>
              <p className="text-[10px] text-slate-400">
                Burns "TEST CINEMATIC" text into pixels at 2-4s (test_text.mp4)
              </p>
            </button>
          </div>

          {testStatus && (
            <div className="p-2.5 rounded-xl bg-slate-950/90 border border-slate-800 text-xs font-mono text-emerald-300 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
              <span>{testStatus}</span>
            </div>
          )}
        </div>

        {/* SECTION B: ANDROID MEDIA3 REAL VIDEO PLAYER & CONTROLS */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-start">
          {/* Real Media3 Player */}
          <div className="md:col-span-6 flex flex-col gap-2">
            <AndroidMedia3Player
              videoUrl={activePlayerUrl}
              posterUrl={activePlayerPoster}
              title={activePlayerTitle}
              editPlan={editPlan}
              isRenderedMaster={isMasterRendered}
            />

            {/* Direct File Action Links */}
            <div className="flex items-center justify-between text-xs px-1">
              <span className="text-slate-400 truncate max-w-[200px]">
                File: {activePlayerUrl.split('?')[0]}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleOpenVideoInNewTab}
                  className="text-cyan-400 hover:underline flex items-center gap-1 font-semibold"
                >
                  <ExternalLink className="w-3 h-3" />
                  Open in New Tab
                </button>
              </div>
            </div>
          </div>

          {/* Side Controls & Autonomous Director Pipeline */}
          <div className="md:col-span-6 flex flex-col gap-3">
            {/* Input Selection & Upload */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Video className="w-4 h-4 text-emerald-400" />
                  Select Football Footage
                </span>

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3 py-1 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 text-xs font-bold flex items-center gap-1.5 border border-emerald-500/30 transition"
                >
                  <Upload className="w-3.5 h-3.5" />
                  Upload Custom Video
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>

              {/* Match Preset choices */}
              <div className="grid grid-cols-2 gap-2">
                {presets.map((preset) => {
                  const isSelected = selectedVideo?.id === preset.id;
                  return (
                    <button
                      key={preset.id}
                      onClick={() => {
                        setSelectedVideo(preset);
                        setActivePlayerUrl(resolveMediaUrl(preset.sourceUrl));
                        setActivePlayerPoster(resolveMediaUrl(preset.posterUrl || '/videos/poster_10s.jpg'));
                        setActivePlayerTitle(preset.title);
                        setIsMasterRendered(false);
                      }}
                      className={`p-2.5 rounded-xl border text-left transition flex flex-col justify-between ${
                        isSelected
                          ? 'bg-emerald-950/60 border-emerald-400 text-white ring-1 ring-emerald-500/40'
                          : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      <span className="text-xs font-bold truncate">{preset.title}</span>
                      <span className="text-[10px] text-slate-400 mt-1">
                        {preset.duration}s duration
                      </span>
                    </button>
                  );
                })}
              </div>

              {uploadedFileName && (
                <div className="p-2 bg-emerald-950/40 border border-emerald-800 rounded-lg text-[11px] text-emerald-300 truncate">
                  Active Upload: {uploadedFileName}
                </div>
              )}
            </div>

            {/* AI Style & Reference */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-2.5">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Cinematic Style Direction
              </span>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'CINEMATIC SPORTS', label: 'Cinematic Sports', icon: Flame },
                  { id: 'DARK FOOTBALL DOCUMENTARY', label: 'Dark Documentary', icon: Film },
                  { id: 'HYPE / VIRAL FOOTBALL', label: 'Hype / Viral', icon: Zap },
                  { id: 'EMOTIONAL FOOTBALL STORY', label: 'Emotional Story', icon: Heart },
                  { id: 'REFERENCE CINEMATIC REEL', label: 'Reference Reel', icon: Sparkles },
                  { id: 'PSYCHOLOGICAL DRAMA', label: 'Psychological Drama', icon: Eye },
                ].map((st) => {
                  const isSelected = selectedStyle === st.id;
                  const Icon = st.icon;
                  return (
                    <button
                      key={st.id}
                      onClick={() => setSelectedStyle(st.id as AIStyle)}
                      className={`p-2.5 rounded-xl border text-left flex items-center gap-2 transition ${
                        isSelected
                          ? 'bg-emerald-950/60 border-emerald-400 text-white'
                          : 'bg-slate-950/40 border-slate-800 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Icon
                        className={`w-3.5 h-3.5 shrink-0 ${
                          isSelected ? 'text-emerald-400' : 'text-slate-500'
                        }`}
                      />
                      <span className="text-xs font-bold leading-tight">{st.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* RENDER MODE */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Render Mode</span>
                <span className="text-[10px] text-emerald-300">100% local render — no external API</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {([
                  { id: 'ORIGINAL FOOTAGE ONLY', label: 'Source Only' },
                  { id: 'AI ENHANCED', label: 'Enhanced' },
                  { id: 'AI CINEMATIC', label: 'Cinematic' },
                ] as { id: GenerationTier; label: string }[]).map((tier) => (
                  <button
                    key={tier.id}
                    onClick={() => setGenerationTier(tier.id)}
                    className={`p-2 rounded-xl border text-[10px] font-black transition ${generationTier === tier.id
                      ? 'bg-emerald-950/70 border-emerald-400 text-emerald-200'
                      : 'bg-slate-950/50 border-slate-800 text-slate-400 hover:text-white'}`}
                  >
                    {tier.label}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-slate-400">
                Every mode renders strictly from the uploaded footage with the local FFmpeg engine. Cinematic adds stronger local color grading and extra close-up inserts. No external AI video API is ever called.
              </p>
            </div>

            {/* CINEMATIC MODE — drives the Cinematic Director */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Cinematic Mode</span>
                <span className="text-[10px] text-purple-300">Director v11 — player-first, event-driven</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {([
                  { id: 'STANDARD', label: 'Standard', hint: 'Safe pacing' },
                  { id: 'PRO', label: 'Pro', hint: 'Stronger dynamics' },
                  { id: 'REFERENCE', label: 'Reference', hint: 'Measured style' },
                ] as { id: CinematicMode; label: string; hint: string }[]).map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setCinematicMode(m.id)}
                    className={`p-2 rounded-xl border text-[10px] font-black transition ${cinematicMode === m.id
                      ? 'bg-purple-950/70 border-purple-400 text-purple-200'
                      : 'bg-slate-950/50 border-slate-800 text-slate-400 hover:text-white'}`}
                  >
                    <span className="block">{m.label}</span>
                    <span className="block text-[9px] font-semibold opacity-70">{m.hint}</span>
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-slate-400">
                {cinematicMode === 'REFERENCE'
                  ? 'REFERENCE mode uses the measured style parameters of your reference video (pacing, cut density, zoom, speed, text, transitions, colour). It never copies its timestamps or shot order.'
                  : 'The director selects real player/ball-tracked moments, applies eased dynamic reframing, event-driven speed ramps, one cinematic grade and a hard-cut rhythm.'}
              </p>
            </div>

            {/* MASTER RENDER ACTION */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                    Full 64-Second Master Assembly
                  </h4>
                  <p className="text-[11px] text-slate-400">
                    Local motion analysis → cinematic reframing → speed ramps → captions → Final MP4
                  </p>
                </div>
              </div>

              {isRenderingFull ? (
                <div className="flex flex-col gap-2 bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="flex justify-between text-xs font-bold">
                    <span className="text-emerald-400">Rendering with FFmpeg...</span>
                    <span className="font-mono text-white">{renderProgress.percent}%</span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all duration-300"
                      style={{ width: `${renderProgress.percent}%` }}
                    />
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 truncate">
                    {renderProgress.stage}
                  </span>
                </div>
              ) : (
                <button
                  onClick={() => runCompleteRealPipeline()}
                  disabled={isTestRunning || isRenderingFull}
                  className="w-full py-3.5 rounded-xl bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 hover:from-emerald-400 hover:to-cyan-300 text-slate-950 font-black text-sm tracking-wider shadow-lg shadow-emerald-500/20 transition transform active:scale-98 flex items-center justify-center gap-2"
                >
                  <Zap className="w-4 h-4 fill-current" />
                  RENDER 64S CINEMATIC MASTER (AI + FFMPEG)
                </button>
              )}

              {pipelineError && (
                <div className="p-3 rounded-xl bg-red-950/50 border border-red-500/50 text-[11px] text-red-200 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
                  <div className="flex flex-col gap-1">
                    <span className="font-bold text-red-300">Pipeline error</span>
                    <span className="leading-relaxed">{pipelineError}</span>
                    <button
                      onClick={() => setPipelineError(null)}
                      className="self-start mt-1 px-2 py-0.5 rounded bg-red-500/20 hover:bg-red-500/30 text-red-200 font-semibold"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              )}

              {/* Master Output Action Bar */}
              {isMasterRendered && (
                <div className="grid grid-cols-3 gap-2 pt-1 border-t border-slate-800">
                  <button
                    onClick={handleOpenVideoInNewTab}
                    className="py-2 px-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs flex items-center justify-center gap-1 border border-slate-700"
                  >
                    <ExternalLink className="w-3.5 h-3.5 text-cyan-400" />
                    OPEN VIDEO
                  </button>

                  <button
                    onClick={handleSaveToDevice}
                    className="py-2 px-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-1 shadow"
                  >
                    <Download className="w-3.5 h-3.5" />
                    SAVE GALLERY
                  </button>

                  <button
                    onClick={handleShare}
                    className="py-2 px-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs flex items-center justify-center gap-1 border border-slate-700"
                  >
                    <Share2 className="w-3.5 h-3.5 text-purple-400" />
                    SHARE
                  </button>
                </div>
              )}
            </div>

            {/* Quick Actions & Timeline Fine-Tuning */}
            {editPlan && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowTimelineEditor(!showTimelineEditor)}
                  className="flex-1 py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-bold text-xs flex items-center justify-center gap-1.5 transition"
                >
                  <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                  {showTimelineEditor ? 'Hide Timeline Editor' : 'Open Timeline Editor'}
                </button>

                {qcReview && (
                  <button
                    onClick={() => setShowQCModal(true)}
                    className="py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-emerald-400 font-bold text-xs flex items-center gap-1.5 transition"
                  >
                    <ShieldCheck className="w-3.5 h-3.5" />
                    QC Report
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* SECTION C: MANUAL TIMELINE EDITOR (When Opened) */}
        {showTimelineEditor && editPlan && (
          <TimelineEditor
            editPlan={editPlan}
            onUpdatePlan={(updated) => setEditPlan(updated)}
            onClose={() => setShowTimelineEditor(false)}
          />
        )}
      </div>

      {/* QC Modal */}
      {showQCModal && qcReview && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-lg">
            <GeminiQCPanel review={qcReview} onClose={() => setShowQCModal(false)} />
          </div>
        </div>
      )}

      {/* Reference Video Modal */}
      {showRefModal && (
        <ReferenceStyleModal
          currentProfile={referenceProfile}
          onApplyProfile={(profile, measured, localPath) => {
            setReferenceProfile(profile);
            if (measured) setReferenceStyleProfile(measured);
            if (localPath) setReferenceLocalPath(localPath);
            if (measured) setCinematicMode('REFERENCE');
          }}
          onClose={() => setShowRefModal(false)}
        />
      )}
    </AndroidPhoneFrame>
  );
}
