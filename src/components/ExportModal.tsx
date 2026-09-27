import React, { useState } from 'react';
import {
  Download,
  Share2,
  FolderPlus,
  CheckCircle2,
  X,
  FileVideo,
  Smartphone,
  Copy,
  Check,
} from 'lucide-react';
import { videoRenderEngine } from '../services/videoEngine';

interface ExportModalProps {
  onClose: () => void;
  onNewProject: () => void;
}

export const ExportModal: React.FC<ExportModalProps> = ({ onClose, onNewProject }) => {
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [copiedShare, setCopiedShare] = useState(false);

  const startExport = async () => {
    setIsExporting(true);
    setExportProgress(0);
    setStatusText('Preparing 1080x1920 rendering pipeline...');

    try {
      const blob = await videoRenderEngine.exportMasterVideo((pct, status) => {
        setExportProgress(pct);
        setStatusText(status);
      });

      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
      setIsExporting(false);
    } catch (err: any) {
      console.error('Export error:', err);
      setIsExporting(false);
      setStatusText('Export failed: ' + err.message);
    }
  };

  const handleDownload = () => {
    if (!downloadUrl) return;
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `FOOTBALL_CINEMATIC_64S_${Date.now()}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleSaveToDevice = () => {
    // Android file save simulation / trigger
    handleDownload();
  };

  const handleShare = async () => {
    if (navigator.share && downloadUrl) {
      try {
        await navigator.share({
          title: 'FOOTBALL CINEMATIC AI Short',
          text: 'Check out this 64-second cinematic football master short created with Gemini & Veo!',
          url: window.location.href,
        });
      } catch (err) {}
    } else {
      navigator.clipboard.writeText(window.location.href);
      setCopiedShare(true);
      setTimeout(() => setCopiedShare(false), 2000);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl flex flex-col gap-5 text-slate-100 animate-in fade-in duration-200">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center border border-emerald-500/30">
              <FileVideo className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-white">Export 64s Cinematic Master</h3>
              <p className="text-[11px] text-slate-400">1080×1920 • 9:16 Vertical • H.264 AAC</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Export Spec Details */}
        <div className="bg-slate-950 border border-slate-800 rounded-2xl p-4 flex flex-col gap-2.5">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Format</span>
            <span className="font-mono font-bold text-white">MP4 / H.264</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Resolution</span>
            <span className="font-mono font-bold text-emerald-400">1080 × 1920 (9:16)</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Total Duration</span>
            <span className="font-mono font-bold text-white">64.0 Seconds Exact</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Audio Codec</span>
            <span className="font-mono font-bold text-white">Stereo 48kHz AAC Synced</span>
          </div>
        </div>

        {/* Progress or Actions */}
        {isExporting ? (
          <div className="flex flex-col gap-3 py-2">
            <div className="flex justify-between text-xs font-semibold">
              <span className="text-emerald-400">Rendering Master...</span>
              <span className="font-mono text-white">{exportProgress}%</span>
            </div>
            <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-200"
                style={{ width: `${exportProgress}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400 font-mono text-center truncate">
              {statusText}
            </p>
          </div>
        ) : downloadUrl ? (
          <div className="flex flex-col gap-2.5">
            <div className="flex items-center gap-2 p-3 bg-emerald-950/60 border border-emerald-700/60 rounded-xl text-emerald-300 text-xs">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Master short ready for export!</span>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-1">
              <button
                onClick={handleSaveToDevice}
                className="py-2.5 px-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 transition shadow"
              >
                <Smartphone className="w-4 h-4" />
                SAVE TO DEVICE
              </button>

              <button
                onClick={handleShare}
                className="py-2.5 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs flex items-center justify-center gap-1.5 transition border border-slate-700"
              >
                {copiedShare ? (
                  <>
                    <Check className="w-4 h-4 text-emerald-400" />
                    COPIED LINK
                  </>
                ) : (
                  <>
                    <Share2 className="w-4 h-4" />
                    SHARE
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={startExport}
            className="w-full py-3 rounded-2xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-slate-950 font-black text-sm tracking-wide shadow-lg shadow-emerald-500/20 transition transform active:scale-98 flex items-center justify-center gap-2"
          >
            <Download className="w-4 h-4" />
            START 1080P RENDERING
          </button>
        )}

        {/* Footer buttons */}
        <div className="border-t border-slate-800 pt-3 flex justify-between">
          <button
            onClick={() => {
              onClose();
              onNewProject();
            }}
            className="text-xs text-slate-400 hover:text-white flex items-center gap-1.5 transition"
          >
            <FolderPlus className="w-3.5 h-3.5" />
            NEW PROJECT
          </button>

          <button
            onClick={onClose}
            className="text-xs text-slate-400 hover:text-white transition"
          >
            Back to Editor
          </button>
        </div>
      </div>
    </div>
  );
};
