/**
 * WebAudio Football Cinematic Sound Engine
 * Dynamically synthesizes orchestral hybrid percussion, sub-bass braams,
 * crowd atmosphere, and riser impacts synchronized to Gemini's BPM & energy curve.
 */

export class FootballAudioEngine {
  private ctx: AudioContext | null = null;
  private isPlaying: boolean = false;
  private mediaStreamDest: MediaStreamAudioDestinationNode | null = null;
  private masterGain: GainNode | null = null;
  private crowdGain: GainNode | null = null;
  private crowdSource: AudioBufferSourceNode | null = null;
  private timerId: number | null = null;

  private bpm: number = 128;
  private energyCurve: number[] = [0.3, 0.4, 0.55, 0.65, 0.75, 0.85, 0.7, 1.0, 0.9, 0.7, 0.5];
  private isMuted: boolean = false;

  constructor() {}

  public init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      this.ctx = new AudioCtx();
      this.mediaStreamDest = this.ctx.createMediaStreamDestination();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.85;

      // Connect master to hardware speakers and to media stream
      this.masterGain.connect(this.ctx.destination);
      this.masterGain.connect(this.mediaStreamDest);
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  public getAudioStream(): MediaStream | null {
    if (!this.mediaStreamDest) {
      this.init();
    }
    return this.mediaStreamDest ? this.mediaStreamDest.stream : null;
  }

  public configure(bpm: number, energyCurve: number[]) {
    this.bpm = bpm || 128;
    if (energyCurve && energyCurve.length > 0) {
      this.energyCurve = energyCurve;
    }
  }

  public setMute(muted: boolean) {
    this.isMuted = muted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setValueAtTime(muted ? 0 : 0.85, this.ctx.currentTime);
    }
  }

  public start(currentTimeOffsetSec = 0) {
    this.init();
    if (!this.ctx || !this.masterGain) return;
    this.stop();
    this.isPlaying = true;

    // Start crowd ambience bed
    this.startCrowdAmbience();

    // Beat scheduler
    const secondsPerBeat = 60 / this.bpm;
    let nextBeatTime = this.ctx.currentTime;
    let beatIndex = Math.floor(currentTimeOffsetSec / secondsPerBeat);

    const scheduleBeats = () => {
      if (!this.isPlaying || !this.ctx) return;

      while (nextBeatTime < this.ctx.currentTime + 0.3) {
        const videoTime = beatIndex * secondsPerBeat;
        if (videoTime >= 64.0) {
          break;
        }

        // Determine energy at current playback second
        const energyIndex = Math.min(
          this.energyCurve.length - 1,
          Math.floor((videoTime / 64.0) * this.energyCurve.length)
        );
        const energy = this.energyCurve[energyIndex] || 0.6;

        this.triggerBeatSounds(nextBeatTime, beatIndex, energy, videoTime);
        nextBeatTime += secondsPerBeat;
        beatIndex++;
      }

      this.timerId = window.setTimeout(scheduleBeats, 50);
    };

    scheduleBeats();
  }

  public stop() {
    this.isPlaying = false;
    if (this.timerId) {
      clearTimeout(this.timerId);
      this.timerId = null;
    }
    if (this.crowdSource) {
      try {
        this.crowdSource.stop();
        this.crowdSource.disconnect();
      } catch (e) {}
      this.crowdSource = null;
    }
  }

  private triggerBeatSounds(time: number, beatIndex: number, energy: number, videoTime: number) {
    if (!this.ctx || !this.masterGain || this.isMuted) return;

    const isMeasureStart = beatIndex % 4 === 0;
    const isClimaxWindow = videoTime >= 44.5 && videoTime <= 49.5;
    const isRiserWindow = videoTime >= 41.0 && videoTime < 45.0;

    // 1. Kick / Cinematic Impact Braam on downbeats
    if (isMeasureStart || isClimaxWindow) {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      const baseFreq = isClimaxWindow ? 42 : 55;
      osc.type = isClimaxWindow ? 'sawtooth' : 'triangle';
      osc.frequency.setValueAtTime(baseFreq * 2.2, time);
      osc.frequency.exponentialRampToValueAtTime(baseFreq, time + 0.12);
      osc.frequency.exponentialRampToValueAtTime(28, time + 0.6);

      const impactVol = isClimaxWindow ? 0.95 : 0.45 * energy;
      gain.gain.setValueAtTime(impactVol, time);
      gain.gain.exponentialRampToValueAtTime(0.001, time + 0.7);

      osc.connect(gain);
      gain.connect(this.masterGain);

      osc.start(time);
      osc.stop(time + 0.75);
    }

    // 2. High-tension tick / percussion
    if (beatIndex % 2 === 1 || energy > 0.7) {
      const noiseBuffer = this.createSnareNoise();
      if (noiseBuffer) {
        const noise = this.ctx.createBufferSource();
        noise.buffer = noiseBuffer;
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'highpass';
        filter.frequency.value = 2400;

        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(0.12 * energy, time);
        gain.gain.exponentialRampToValueAtTime(0.001, time + 0.08);

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);

        noise.start(time);
        noise.stop(time + 0.09);
      }
    }

    // 3. Climax Riser Whoosh
    if (isRiserWindow && beatIndex % 4 === 0) {
      const riserOsc = this.ctx.createOscillator();
      const riserGain = this.ctx.createGain();
      riserOsc.type = 'sawtooth';
      riserOsc.frequency.setValueAtTime(120, time);
      riserOsc.frequency.exponentialRampToValueAtTime(880, time + 1.2);

      riserGain.gain.setValueAtTime(0.05, time);
      riserGain.gain.linearRampToValueAtTime(0.35, time + 1.1);
      riserGain.gain.exponentialRampToValueAtTime(0.001, time + 1.25);

      riserOsc.connect(riserGain);
      riserGain.connect(this.masterGain);
      riserOsc.start(time);
      riserOsc.stop(time + 1.3);
    }
  }

  private startCrowdAmbience() {
    if (!this.ctx || !this.masterGain) return;

    // Synthesize low stadium hum / crowd roar
    const bufferSize = this.ctx.sampleRate * 3;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    let lastOut = 0.0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1;
      data[i] = (lastOut + 0.02 * white) / 1.02; // Pink noise
      lastOut = data[i];
    }

    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.loop = true;

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 650;

    this.crowdGain = this.ctx.createGain();
    this.crowdGain.gain.value = 0.15;

    source.connect(filter);
    filter.connect(this.crowdGain);
    this.crowdGain.connect(this.masterGain);

    source.start();
    this.crowdSource = source;
  }

  private createSnareNoise(): AudioBuffer | null {
    if (!this.ctx) return null;
    const bufferSize = Math.floor(this.ctx.sampleRate * 0.1);
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    return buffer;
  }

  public triggerImpactSFX() {
    if (!this.ctx || !this.masterGain) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(140, now);
    osc.frequency.exponentialRampToValueAtTime(30, now + 0.4);

    gain.gain.setValueAtTime(0.8, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);

    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(now);
    osc.stop(now + 0.55);
  }
}

export const footballAudioEngine = new FootballAudioEngine();
