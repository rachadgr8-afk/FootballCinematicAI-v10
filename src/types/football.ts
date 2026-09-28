export type AIStyle =
  | 'CINEMATIC SPORTS'
  | 'DARK FOOTBALL DOCUMENTARY'
  | 'HYPE / VIRAL FOOTBALL'
  | 'EMOTIONAL FOOTBALL STORY'
  | 'REFERENCE CINEMATIC REEL'
  | 'PSYCHOLOGICAL DRAMA';

/**
 * Cinematic Mode drives the Cinematic Director.
 *  - STANDARD : conservative pacing, safe dynamics.
 *  - PRO      : stronger dynamics, more punches and speed ramps.
 *  - REFERENCE: pacing/zoom/speed/text/colour are taken from the MEASURED
 *               ReferenceStyleProfile of a supplied reference video. The
 *               reference supplies style parameters only — never timestamps,
 *               shot order, frames, logos or watermarks.
 */
export type CinematicMode = 'STANDARD' | 'PRO' | 'REFERENCE';

export type GenerationTier =
  | 'ORIGINAL FOOTAGE ONLY'
  | 'AI ENHANCED'
  | 'AI CINEMATIC';

export type PipelineStage =
  | 'idle'
  | 'uploading'
  | 'analyzing'
  | 'building_story'
  | 'generating_veo'
  | 'editing'
  | 'ai_review'
  | 'final_render'
  | 'preview_ready';

export interface TimelineClip {
  timeline_index: number;
  source_start: number;
  source_end: number;
  output_start: number;
  output_end: number;
  action: string;
  importance: number; // 1-10
  speed: number; // 0.5 - 2.0
  zoom_start: number; // 1.0 - 1.5
  zoom_end: number;
  crop_x: number; // 0.0 - 1.0 horizontal center anchor
  crop_y: number; // 0.0 - 1.0 vertical center anchor
  /** Optional exit anchor: the reframe eases from crop_x/crop_y to these. */
  crop_x_end?: number;
  crop_y_end?: number;
  transition: 'hard_cut' | 'fade' | 'match_cut' | 'directional_blur' | 'flash' | 'dissolve';
  text: string;
  shot_type?: 'wide' | 'medium' | 'close_up' | 'extreme_close_up' | 'action' | 'reaction' | 'crowd' | 'detail' | 'eye_close_up';
  beat_role?: 'hook' | 'setup' | 'escalation' | 'impact' | 'reaction' | 'climax' | 'outro';
  veo_needed: boolean;
  veo_prompt?: string;
  veo_status?: 'not_requested' | 'pending' | 'ready' | 'fallback';
  veo_clip_url?: string | null;
  /** Optional spoken voice-over / inner-monologue line (psychological drama mode). */
  narration?: string;
  /** Where the reframe anchor came from: real player/ball track, or the fallback. */
  subject_anchor?: 'player_track' | 'ball_track' | 'motion_fallback' | string;
  /** True only when the director had event evidence for this slowdown. */
  slow_motion?: boolean;
  /** True when this shot should isolate its subject (background dimming). */
  subject_isolation?: boolean;
  isolation_reason?: string;
  /** Optional SAM segmentation block, attached by the SAM pre-pass. */
  sam?: any;
  evidence?: {
    event_score: number;
    ball_window_score: number;
    has_ball_evidence: boolean;
    energy: number;
  };
}

export interface MusicConfig {
  style: string;
  bpm: number;
  energy_curve: number[];
}

export interface ColorGrade {
  contrast: number;
  saturation: number;
  highlights: number;
  shadows: number;
  grain: number;
  protect_skin_tones?: boolean;
  cooler_shadows?: boolean;
  warmer_highlights?: boolean;
  prevent_neon_grass?: boolean;
  reference_skin_ratio?: number;
  reference_neon_grass_ratio?: number;
}

export interface EditPlan {
  duration: number; // strictly 64
  aspect_ratio: '9:16';
  subject: {
    name: string;
    confidence: number;
    track_id?: string | null;
  };
  timeline: TimelineClip[];
  music: MusicConfig;
  color_grade: ColorGrade;
  sound_design?: {
    impact_clips: number[];
    riser_starts: number[];
    silence_before_ms: number;
    impact_source: string;
  };
  style_name?: string;
  cinematicMode?: CinematicMode;
  /** Director observability: hero moment, evidence level, real track counts. */
  cinematic_director?: CinematicDirectorReport;
}

export interface CinematicDirectorReport {
  version: string;
  mode: CinematicMode | string;
  evidence_level: 'low' | 'medium' | 'high' | string;
  hero_moment: { t: number; energy: number; score: number; event_backed: boolean };
  protagonist_track: string | null;
  ball_tracked: boolean;
  player_tracks: number;
  subject_isolation_clips: number[];
  speed_ramp_clips: number[];
  transition_mix: Record<string, number>;
  rules: Record<string, boolean>;
}

/**
 * MEASURED style profile of a reference video (real frame + audio analysis).
 * Style parameters only — it is never used as a shot-by-shot template.
 */
export interface ReferenceStyleProfile {
  version: string;
  source: string;
  resolution: string;
  aspect: string;
  fps: number;
  duration: number;
  shot_count: number;
  avg_shot_duration: number;
  median_shot_duration: number;
  shot_duration_p10: number;
  shot_duration_p90: number;
  cut_density: number;
  shot_type_weights: Record<string, number>;
  camera_weights: Record<string, number>;
  zoom_intensity: number;
  slow_motion_shot_ratio: number;
  subject_shot_ratio: number;
  text_per_shot: number;
  transition_weights: Record<string, number>;
  color: {
    contrast: number;
    saturation: number;
    skin_ratio: number;
    neon_grass_ratio: number;
    shadow_B_minus_R: number;
    highlight_B_minus_R: number;
  };
  audio: {
    impact_count: number;
    riser_count: number;
    silence_ratio: number;
    cut_impact_sync_ratio: number;
  };
  hero_structure: Array<Record<string, unknown>>;
  ending_structure: Array<Record<string, unknown>>;
}

export interface FootballVideoMetadata {
  id: string;
  title: string;
  description: string;
  duration: number;
  sourceUrl: string;
  thumbnail?: string;
  posterUrl?: string;
  localPath?: string;
  tags?: string[];
  defaultSubject?: string;
  isUserUploaded?: boolean;
}

export interface QCCorrection {
  timeline_index: number;
  change: string;
  reason: string;
  recommended_speed?: number;
  recommended_crop_x?: number;
  recommended_crop_y?: number;
  recommended_text?: string;
  recommended_transition?: string;
}

export interface QCReview {
  qc_verdict: string;
  overall_critique: string;
  pacing_score: number;
  cinematic_score: number;
  corrections: QCCorrection[];
}

export interface StyleProfile {
  average_shot_duration: number;
  zoom_intensity: number;
  transition_frequency: number;
  slow_motion_frequency: number;
  text_frequency: number;
  color_style: string;
  energy_curve: string;
  recommended_bpm?: number;
  cinematography_notes?: string;
}
