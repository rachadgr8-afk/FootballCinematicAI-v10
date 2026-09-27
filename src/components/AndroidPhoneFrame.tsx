import React from 'react';
import { Smartphone, Monitor, Wifi, BatteryMedium, Signal } from 'lucide-react';

interface AndroidPhoneFrameProps {
  children: React.ReactNode;
  isExpandedView: boolean;
  onToggleView: () => void;
  title?: string;
}

export const AndroidPhoneFrame: React.FC<AndroidPhoneFrameProps> = ({
  children,
  isExpandedView,
  onToggleView,
  title = 'FOOTBALL CINEMATIC AI',
}) => {
  return (
    <div className="flex flex-col items-center justify-center w-full min-h-screen bg-slate-950 text-slate-100 p-2 sm:p-4">
      {/* Top Bar for View Toggling */}
      <div className="w-full max-w-6xl flex items-center justify-between py-2 px-3 mb-2 bg-slate-900/80 backdrop-blur rounded-xl border border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-black text-sm tracking-wider bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            {title}
          </span>
          <span className="hidden sm:inline-block text-xs text-slate-400 px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
            Android Media3 Engine
          </span>
        </div>

        <button
          onClick={onToggleView}
          className="flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition border border-slate-700"
          title={isExpandedView ? 'Switch to Android Pixel Frame' : 'Switch to Expanded Studio Mode'}
        >
          {isExpandedView ? (
            <>
              <Smartphone className="w-3.5 h-3.5 text-emerald-400" />
              <span>Android Phone View</span>
            </>
          ) : (
            <>
              <Monitor className="w-3.5 h-3.5 text-cyan-400" />
              <span>Expanded Studio Mode</span>
            </>
          )}
        </button>
      </div>

      {/* Frame Container */}
      {isExpandedView ? (
        <div className="w-full max-w-6xl bg-slate-900 rounded-2xl border border-slate-800 shadow-2xl p-4 sm:p-6 overflow-hidden">
          {children}
        </div>
      ) : (
        <div className="relative w-full max-w-[420px] aspect-[9/19.5] bg-slate-900 rounded-[44px] p-3 shadow-2xl border-[4px] border-slate-800 flex flex-col justify-between overflow-hidden">
          {/* Android Punch-hole Camera */}
          <div className="absolute top-4 left-1/2 -translate-x-1/2 w-4 h-4 bg-black rounded-full z-40 border border-slate-800 flex items-center justify-center">
            <div className="w-1.5 h-1.5 rounded-full bg-slate-950" />
          </div>

          {/* Android Status Bar */}
          <div className="relative z-30 flex items-center justify-between px-6 pt-1 pb-1 text-[11px] font-semibold text-slate-300">
            <span>09:41</span>
            <div className="flex items-center gap-2">
              <Signal className="w-3.5 h-3.5" />
              <Wifi className="w-3.5 h-3.5" />
              <BatteryMedium className="w-4 h-4 text-emerald-400" />
            </div>
          </div>

          {/* Inner Content Viewport */}
          <div className="flex-1 w-full overflow-y-auto overflow-x-hidden rounded-[32px] bg-slate-950 flex flex-col relative">
            {children}
          </div>

          {/* Android System Navigation Gesture Pill */}
          <div className="relative z-30 pt-2 pb-1 flex justify-center">
            <div className="w-32 h-1 bg-slate-600 rounded-full" />
          </div>
        </div>
      )}
    </div>
  );
};
