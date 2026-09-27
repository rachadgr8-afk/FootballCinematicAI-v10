export type AIStyle =
  | 'CINEMATIC SPORTS'
  | 'DARK FOOTBALL DOCUMENTARY'
  | 'HYPE / VIRAL FOOTBALL'
  | 'EMOTIONAL FOOTBALL STORY'
  | 'REFERENCE CINEMATIC REEL'
  | 'PSYCHOLOGICAL DRAMA';

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
  transition: 'hard_cut' | 'fade' | 'match_cut' | 'directional_blur' | 'flash';
  text: string;
  shot_type?: 'wide' | 'medium' | 'close_up' | 'extreme_close_up' | 'action' | 'reaction' | 'crowd' | 'detail' | 'eye_close_up';
  beat_role?: 'hook' | 'setup' | 'escalation' | 'impact' | 'reaction' | 'climax' | 'outro';
  veo_needed: boolean;
  veo_prompt?: string;
  veo_status?: 'not_requested' | 'pending' | 'ready' | 'fallback';
  veo_clip_url?: string | null;
  /** Optional spoken voice-over / inner-monologue line (psychological drama mode). */
  narration?: string;
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
}

export interface EditPlan {
  duration: number; // strictly 64
  aspect_ratio: '9:16';
  subject: {
    name: string;
    confidence: number;
  };
  timeline: TimelineClip[];
  music: MusicConfig;
  color_grade: ColorGrade;
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
