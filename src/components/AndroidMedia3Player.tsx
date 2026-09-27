import React, { useRef, useEffect, useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Maximize2,
  Zap,
  Film,
  Sparkles,
  Download,
  ExternalLink,
  Sliders,
  AlertCircle,
  Eye,
} from 'lucide-react';
import { EditPlan } from '../types/football';

interface AndroidMedia3PlayerProps {
  videoUrl: string;
  posterUrl?: string;
  editPlan?: EditPlan | null;
  title?: string;
  isRenderedMaster?: boolean;
}

export const AndroidMedia3Player: React.FC<AndroidMedia3PlayerProps> = ({
  videoUrl,
  posterUrl = '/videos/poster_10s.jpg',
  editPlan,
  title = 'Real Match Footage',
  isRenderedMaster = false,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Reload video whenever videoUrl changes
  useEffect(() => {
    if (videoRef.current) {
      setHasError(false);
      setErrorMessage('');
      videoRef.current.load();
      videoRef.current.currentTime = 0;
      setCurrentTime(0);
      setIsPlaying(false);
    }
  }, [videoUrl]);

  // Robust play handler supporting mobile autoplay and user gesture policies
  const handlePlay = async () => {
    if (!videoRef.current) return;

    if (videoRef.current.paused) {
      try {
        await videoRef.current.play();
        setIsPlaying(true);
      } catch (err: any) {
        console.warn('Initial play failed, trying muted playback:', err);
        // Fallback for mobile strict policies: mute first, then play
        videoRef.current.muted = true;
        setIsMuted(true);
        try {
          await videoRef.current.play();
          setIsPlaying(true);
        } catch (retryErr: any) {
          setHasError(true);
          setErrorMessage('Please use the native player controls or open in new tab.');
        }
      }
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    const next = !isMuted;
    videoRef.current.muted = next;
    setIsMuted(next);
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || 64);
      setHasError(false);
    }
  };

  const handleError = () => {
    if (videoRef.current?.error) {
      const code = videoRef.current.error.code;
      const msg = videoRef.current.error.message || `MediaError ${code}`;
      setHasError(true);
      setErrorMessage(msg);
    }
  };

  const formatTime = (sec: number) => {
    if (isNaN(sec) || !isFinite(sec)) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col gap-2 w-full max-w-[420px] mx-auto">
      {/* Player Container */}
      <div className="relative w-full aspect-[9/16] bg-slate-950 rounded-2xl overflow-hidden shadow-2xl border-2 border-slate-800 flex items-center justify-center">
        {/* Real HTML5 Video element with native controls for 100% Android guarantee */}
        <video
          ref={videoRef}
          src={videoUrl}
          poster={posterUrl}
          playsInline
          preload="auto"
          controls
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onError={handleError}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          className="w-full h-full object-contain bg-black"
        />

        {/* Top Floating Badge */}
        <div className="absolute top-2 left-2 right-2 pointer-events-none flex items-center justify-between z-10">
          <span
            className={`px-2 py-0.5 text-[10px] font-black uppercase tracking-wider rounded text-white flex items-center gap-1 shadow ${
              isRenderedMaster ? 'bg-red-600' : 'bg-emerald-600'
            }`}
          >
            <Zap className="w-3 h-3 fill-current" />
            {isRenderedMaster ? '1080x1920 MP4' : 'Source Match'}
          </span>

          <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-slate-900/90 text-slate-300 border border-slate-700">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>

        {/* Error Fallback */}
        {hasError && (
          <div className="absolute inset-0 bg-black/95 p-6 flex flex-col items-center justify-center text-center gap-3 z-30">
            <AlertCircle className="w-10 h-10 text-amber-400" />
            <h4 className="text-white font-bold text-sm">Media Player Notice</h4>
            <p className="text-xs text-slate-300">{errorMessage}</p>
            <a
              href={videoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 rounded-xl bg-emerald-500 text-slate-950 font-bold text-xs flex items-center gap-1.5 shadow mt-2"
            >
              <ExternalLink className="w-4 h-4" />
              Open Video in New Tab
            </a>
          </div>
        )}
      </div>

      {/* Auxiliary Mobile Quick Action Bar */}
      <div className="flex items-center justify-between gap-2 px-1">
        <button
          onClick={handlePlay}
          className="flex-1 py-2 px-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 shadow transition"
        >
          {isPlaying ? (
            <>
              <Pause className="w-4 h-4 fill-current" />
              <span>إيقاف مؤقت (Pause)</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>تشغيل الفيديو (Play)</span>
            </>
          )}
        </button>

        <button
          onClick={toggleMute}
          className="py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold flex items-center gap-1 transition"
        >
          {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          <span>{isMuted ? 'Muted' : 'Sound ON'}</span>
        </button>

        <a
          href={videoUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 text-xs font-semibold flex items-center gap-1 transition"
          title="Open in new browser window"
        >
          <ExternalLink className="w-4 h-4" />
          <span>فتح كامل</span>
        </a>
      </div>
    </div>
  );
};
