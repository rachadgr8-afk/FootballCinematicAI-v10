import { EditPlan, TimelineClip } from '../types/football';
import { footballAudioEngine } from './audioEngine';

export interface RenderState {
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  activeClipIndex: number;
  activeClip: TimelineClip | null;
  phaseLabel: string;
}

export class VideoRenderEngine {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private sourceVideo: HTMLVideoElement | null = null;
  private veoVideo: HTMLVideoElement | null = null;

  private editPlan: EditPlan | null = null;
  private isPlaying: boolean = false;
  private currentTime: number = 0;
  private duration: number = 64.0;
  private animationFrameId: number | null = null;
  private lastTimestamp: number = 0;

  private onStateChange: ((state: RenderState) => void) | null = null;

  // MediaRecorder export
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];

  constructor() {
    this.sourceVideo = document.createElement('video');
    this.sourceVideo.crossOrigin = 'anonymous';
    this.sourceVideo.playsInline = true;
    this.sourceVideo.muted = true; // Mute video element, WebAudio handles synced soundtrack & impact SFX

    this.veoVideo = document.createElement('video');
    this.veoVideo.crossOrigin = 'anonymous';
    this.veoVideo.playsInline = true;
    this.veoVideo.muted = true;
  }

  public attachCanvas(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
  }

  public setSourceVideoUrl(url: string) {
    if (this.sourceVideo) {
      this.sourceVideo.src = url;
      this.sourceVideo.load();
    }
  }

  public setEditPlan(plan: EditPlan) {
    this.editPlan = plan;
    this.duration = plan.duration || 64.0;
    footballAudioEngine.configure(plan.music.bpm, plan.music.energy_curve);
  }

  public onUpdate(callback: (state: RenderState) => void) {
    this.onStateChange = callback;
  }

  public play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.lastTimestamp = performance.now();
    footballAudioEngine.start(this.currentTime);

    const loop = (timestamp: number) => {
      if (!this.isPlaying) return;
      const deltaSec = (timestamp - this.lastTimestamp) / 1000;
      this.lastTimestamp = timestamp;

      // Update current time according to active clip's playback speed
      const clip = this.getActiveClip(this.currentTime);
      const speedMultiplier = clip ? (clip.speed || 1.0) : 1.0;
      this.currentTime += deltaSec * speedMultiplier;

      if (this.currentTime >= this.duration) {
        this.currentTime = 0;
      }

      this.renderFrame(this.currentTime);
      this.emitState();
      this.animationFrameId = requestAnimationFrame(loop);
    };

    this.animationFrameId = requestAnimationFrame(loop);
  }

  public pause() {
    this.isPlaying = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    footballAudioEngine.stop();
    if (this.sourceVideo) this.sourceVideo.pause();
    if (this.veoVideo) this.veoVideo.pause();
    this.emitState();
  }

  public seek(targetSec: number) {
    this.currentTime = Math.max(0, Math.min(this.duration, targetSec));
    this.renderFrame(this.currentTime);
    this.emitState();
    if (this.isPlaying) {
      footballAudioEngine.start(this.currentTime);
    }
  }

  public getCurrentTime(): number {
    return this.currentTime;
  }

  public getDuration(): number {
    return this.duration;
  }

  public getActiveClip(time: number): TimelineClip | null {
    if (!this.editPlan || !this.editPlan.timeline) return null;
    for (const clip of this.editPlan.timeline) {
      if (time >= clip.output_start && time < clip.output_end) {
        return clip;
      }
    }
    return this.editPlan.timeline[this.editPlan.timeline.length - 1] || null;
  }

  public getStoryPhase(time: number): string {
    if (time < 4) return '00–04 HOOK';
    if (time < 9) return '04–09 PLAYER INTRO';
    if (time < 15) return '09–15 FIRST ACTION';
    if (time < 22) return '15–22 BUILD-UP';
    if (time < 29) return '22–29 TENSION';
    if (time < 37) return '29–37 SKILL SEQUENCE';
    if (time < 45) return '37–45 ANTICIPATION';
    if (time < 49) return '45–49 CLIMAX';
    if (time < 56) return '49–56 CELEBRATION / REACTION';
    if (time < 61) return '56–61 HERO SHOT';
    return '61–64 ENDING';
  }

  private emitState() {
    if (!this.onStateChange) return;
    const clip = this.getActiveClip(this.currentTime);
    const activeIndex = clip ? clip.timeline_index : 0;
    this.onStateChange({
      currentTime: this.currentTime,
      duration: this.duration,
      isPlaying: this.isPlaying,
      activeClipIndex: activeIndex,
      activeClip: clip,
      phaseLabel: this.getStoryPhase(this.currentTime),
    });
  }

  /**
   * Render single 9:16 frame onto canvas with smart cropping, zoom, color grading & kinetic graphics
   */
  public renderFrame(time: number) {
    if (!this.canvas || !this.ctx) return;
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    const clip = this.getActiveClip(time);
    if (!clip) {
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);
      return;
    }

    // Determine clip relative progress (0.0 to 1.0)
    const clipDuration = Math.max(0.1, clip.output_end - clip.output_start);
    const clipProgress = Math.min(1.0, Math.max(0.0, (time - clip.output_start) / clipDuration));

    // Calculate source time for video element
    const sourceDuration = clip.source_end - clip.source_start;
    const sourceCurrentTime = clip.source_start + clipProgress * sourceDuration;

    // Determine which video element to render from
    let videoEl: HTMLVideoElement = this.sourceVideo!;
    if (clip.veo_clip_url && (clip.veo_status === 'ready' || clip.veo_status === 'fallback')) {
      if (this.veoVideo?.src !== clip.veo_clip_url) {
        this.veoVideo!.src = clip.veo_clip_url;
      }
      videoEl = this.veoVideo!;
    }

    // Sync video time if drift is > 0.15s
    if (videoEl.readyState >= 2) {
      const targetVidTime = videoEl === this.sourceVideo ? sourceCurrentTime : (clipProgress * 4.0);
      if (Math.abs(videoEl.currentTime - targetVidTime) > 0.18) {
        videoEl.currentTime = targetVidTime;
      }
    }

    // Clear background
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, width, height);

    // Dynamic zoom interpolation (zoom_start -> zoom_end)
    const currentZoom = clip.zoom_start + (clip.zoom_end - clip.zoom_start) * clipProgress;

    // Apply Smart 9:16 vertical crop
    // Horizontal widescreen (e.g. 1920x1080) needs to be cropped to 9:16 aspect ratio:
    // Crop width = sourceHeight * (9 / 16) / currentZoom
    // Center crop around (crop_x, crop_y)
    const vidW = videoEl.videoWidth || 1280;
    const vidH = videoEl.videoHeight || 720;
    const targetAspect = 9 / 16;

    // The desired window in the source video
    let cropH = vidH / currentZoom;
    let cropW = (vidH * targetAspect) / currentZoom;

    if (cropW > vidW) {
      cropW = vidW;
      cropH = vidW / targetAspect;
    }

    // Center crop around crop_x and crop_y
    const centerX = (clip.crop_x ?? 0.5) * vidW;
    const centerY = (clip.crop_y ?? 0.5) * vidH;

    let sx = centerX - cropW / 2;
    let sy = centerY - cropH / 2;

    // Clamp inside source bounds
    sx = Math.max(0, Math.min(vidW - cropW, sx));
    sy = Math.max(0, Math.min(vidH - cropH, sy));

    // Subtle camera shake on skill sequence and climax
    let shakeX = 0;
    let shakeY = 0;
    if (time >= 44.5 && time <= 49.0) {
      shakeX = (Math.random() - 0.5) * 6;
      shakeY = (Math.random() - 0.5) * 6;
    }

    ctx.save();

    // Color grading filters
    const grade = this.editPlan?.color_grade || { contrast: 1.2, saturation: 1.15 };
    ctx.filter = `contrast(${grade.contrast * 100}%) saturate(${grade.saturation * 100}%) brightness(102%)`;

    // Directional motion blur / whip pan during transition
    if (clip.transition === 'directional_blur' && clipProgress < 0.12) {
      const blurAmount = (1 - clipProgress / 0.12) * 8;
      ctx.filter += ` blur(${blurAmount}px)`;
    }

    // Draw video frame with smart crop
    if (videoEl.readyState >= 2) {
      ctx.drawImage(videoEl, sx, sy, cropW, cropH, shakeX, shakeY, width, height);
    } else {
      // Placeholder aesthetic football pitch gradient
      const grad = ctx.createLinearGradient(0, 0, width, height);
      grad.addColorStop(0, '#0c1b2a');
      grad.addColorStop(0.5, '#08281a');
      grad.addColorStop(1, '#050a12');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);
    }

    ctx.restore();

    // 1. Film grain & vignette
    this.renderVignetteAndAtmosphere(ctx, width, height, grade);

    // 2. Cinematic Transitions (Fade, Flash)
    this.renderTransitions(ctx, width, height, clip, clipProgress);

    // 3. Football Action Kinetic Typography & HUD
    this.renderKineticText(ctx, width, height, clip, time);
  }

  private renderVignetteAndAtmosphere(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    grade: any
  ) {
    // Subtle cinematic vignette
    const radial = ctx.createRadialGradient(
      width / 2,
      height / 2,
      width * 0.35,
      width / 2,
      height / 2,
      width * 0.95
    );
    radial.addColorStop(0, 'rgba(0, 0, 0, 0)');
    radial.addColorStop(0.7, 'rgba(0, 0, 0, 0.25)');
    radial.addColorStop(1, 'rgba(0, 0, 0, 0.7)');

    ctx.fillStyle = radial;
    ctx.fillRect(0, 0, width, height);

    // Subtle sports documentary grain
    if (grade.grain > 0) {
      ctx.fillStyle = `rgba(255, 255, 255, ${grade.grain * 0.04})`;
      for (let i = 0; i < 40; i++) {
        const gx = Math.random() * width;
        const gy = Math.random() * height;
        const gw = Math.random() * 2 + 1;
        ctx.fillRect(gx, gy, gw, gw);
      }
    }
  }

  private renderTransitions(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    clip: TimelineClip,
    progress: number
  ) {
    // Flash transition (white blast on goal or impact)
    if (clip.transition === 'flash' && progress < 0.15) {
      const alpha = 1.0 - progress / 0.15;
      ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.85})`;
      ctx.fillRect(0, 0, width, height);
    }

    // Fade transition (fade in from black at beginning of clip)
    if (clip.transition === 'fade' && progress < 0.18) {
      const alpha = 1.0 - progress / 0.18;
      ctx.fillStyle = `rgba(0, 0, 0, ${alpha * 0.95})`;
      ctx.fillRect(0, 0, width, height);
    }
  }

  private renderKineticText(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    clip: TimelineClip,
    time: number
  ) {
    // Top Sports Broadcast Header
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(0, 0, width, 56);

    // Phase Badge (Hook, Climax, Skill, etc.)
    const phase = this.getStoryPhase(time);
    ctx.fillStyle = time >= 45 && time < 49 ? '#ef4444' : '#10b981';
    ctx.beginPath();
    ctx.roundRect(16, 14, 110, 26, 4);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px monospace';
    ctx.fillText(phase.split(' ')[1] || 'FOOTBALL', 24, 30);

    // Timecode in top right
    const mins = Math.floor(time / 60);
    const secs = (time % 60).toFixed(2).padStart(5, '0');
    ctx.font = 'bold 12px monospace';
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(`${mins}:${secs} / 01:04.00`, width - 140, 31);

    // Main Kinetic Action Text if present
    if (clip.text && clip.text.trim().length > 0) {
      const text = clip.text.toUpperCase();
      ctx.textAlign = 'center';

      const isClimax = time >= 45 && time <= 49;
      const fontSize = isClimax ? 44 : 32;
      ctx.font = `900 ${fontSize}px sans-serif`;

      const textY = isClimax ? height * 0.42 : height * 0.78;

      // Dark background pill/ribbon for readability
      ctx.font = `900 ${fontSize}px sans-serif`;
      const textMetrics = ctx.measureText(text);
      const bgW = textMetrics.width + 48;
      const bgH = fontSize + 24;

      ctx.fillStyle = isClimax ? 'rgba(239, 68, 68, 0.9)' : 'rgba(0, 0, 0, 0.75)';
      ctx.beginPath();
      ctx.roundRect(width / 2 - bgW / 2, textY - bgH / 2 - 8, bgW, bgH, 8);
      ctx.fill();

      // Border glow
      ctx.strokeStyle = isClimax ? '#fef08a' : '#38bdf8';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Text stroke and fill
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 4;
      ctx.strokeText(text, width / 2, textY);

      ctx.fillStyle = isClimax ? '#ffffff' : '#f8fafc';
      ctx.fillText(text, width / 2, textY);
    }

    // Bottom Subject Tag
    if (this.editPlan?.subject?.name) {
      ctx.textAlign = 'left';
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.fillText(`PROTAGONIST: ${this.editPlan.subject.name.toUpperCase()}`, 18, height - 20);
    }

    ctx.restore();
  }

  /**
   * Export final 1080x1920 9:16 MP4 video
   */
  public async exportMasterVideo(
    onProgress: (percent: number, status: string) => void
  ): Promise<Blob> {
    this.pause();

    // Create high-res offscreen rendering canvas
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = 1080;
    exportCanvas.height = 1920;
    const origCanvas = this.canvas;
    const origCtx = this.ctx;

    this.canvas = exportCanvas;
    this.ctx = exportCanvas.getContext('2d', { alpha: false });

    // Setup MediaStream
    const canvasStream = exportCanvas.captureStream(30); // 30 FPS
    const audioStream = footballAudioEngine.getAudioStream();

    const combinedTracks = [...canvasStream.getVideoTracks()];
    if (audioStream) {
      audioStream.getAudioTracks().forEach((track) => combinedTracks.push(track));
    }
    const combinedStream = new MediaStream(combinedTracks);

    // Pick best supported MIME type
    const mimeTypes = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
      'video/mp4',
    ];
    let selectedMime = 'video/webm';
    for (const mime of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mime)) {
        selectedMime = mime;
        break;
      }
    }

    this.recordedChunks = [];
    const recorder = new MediaRecorder(combinedStream, {
      mimeType: selectedMime,
      videoBitsPerSecond: 8000000, // 8 Mbps high-quality 1080x1920
    });

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        this.recordedChunks.push(e.data);
      }
    };

    return new Promise(async (resolve, reject) => {
      recorder.onstop = () => {
        // Restore interactive canvas
        this.canvas = origCanvas;
        this.ctx = origCtx;
        const blob = new Blob(this.recordedChunks, { type: 'video/mp4' });
        resolve(blob);
      };

      recorder.onerror = (e) => {
        this.canvas = origCanvas;
        this.ctx = origCtx;
        reject(e);
      };

      recorder.start(500);
      footballAudioEngine.start(0);

      // Render 64 seconds in real-time recording
      const totalSec = 64.0;
      const fps = 30;
      const totalFrames = totalSec * fps;
      let frame = 0;

      const recordStep = () => {
        const renderTime = (frame / totalFrames) * totalSec;
        this.renderFrame(renderTime);

        const pct = Math.floor((frame / totalFrames) * 100);
        if (frame % 30 === 0) {
          onProgress(pct, `Encoding 1080x1920 H.264 Master: frame ${frame}/${totalFrames}...`);
        }

        frame++;
        if (frame <= totalFrames) {
          setTimeout(recordStep, 1000 / fps);
        } else {
          recorder.stop();
          footballAudioEngine.stop();
        }
      };

      recordStep();
    });
  }

  public destroy() {
    this.pause();
    if (this.sourceVideo) {
      this.sourceVideo.src = '';
    }
    if (this.veoVideo) {
      this.veoVideo.src = '';
    }
  }
}

export const videoRenderEngine = new VideoRenderEngine();
