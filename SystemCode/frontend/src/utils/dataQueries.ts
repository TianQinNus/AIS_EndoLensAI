export interface Point { x: number; y: number }

export interface BackendVideo {
  video_id: string
  filename: string
  video_url: string
  width: number
  height: number
  duration_seconds: number
  fps: number | null
  total_frames: number | null
}

export interface BackendFrame {
  frame_id: string
  frame_index: number
  timestamp_seconds: number
  width: number
  height: number
  status: 'completed' | 'failed'
  is_candidate: boolean | null
  max_region_area_ratio: number | null
  source_frame_url: string | null
  thumbnail_url: string | null
  mask_url: string | null
  foreground_score: number | null
  contours: number[][][]
  review_status: 'unreviewed' | 'reviewed' | 'rejected'
  revision_id: string | null
  edited_mask_url?: string
  error?: string
}

export interface BackendKeyFrame {
  frame_id: string
  timestamp_seconds: number
  thumbnail_url: string | null
  is_candidate: boolean | null
}

export interface AnalysisStatus {
  task_id: string
  video_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  phase: string
  target_frames: number | null
  progress: {
    target_frames: number | null
    processed_frames: number
    failed_frames: number
    candidate_frames: number
    percent: number | null
  }
  error?: string | null
  performance?: {
    elapsed_seconds: number
    processing_fps: number
  }
}

export interface BackendReport {
  task_id: string
  video_id: string
  status: string
  video?: BackendVideo
  summary: {
    analyzed_frames: number
    candidate_frames: number
    failed_frames: number
    reviewed_frames: number
    candidate_frame_ratio: number | null
    processing_time_seconds?: number
  }
  warnings: string[]
}

export interface VideoMetadata {
  id: string
  name: string
  duration: number
  fps: number
  totalFrames: number | null
  streamUrl: string
  demo: boolean
}

export interface PolypDetection {
  id: string
  frameIndex: number
  timestamp: number
  confidence: number | null
  contour: Point[]
  status: string
  isCandidate: boolean | null
  frameStatus: 'completed' | 'failed'
  maskUrl?: string
  sourceUrl?: string
  thumbnailUrl?: string
  width: number
  height: number
}

export interface KeyFrame {
  id: string
  timestamp: number
  label: string
  thumbnailUrl?: string
  isCandidate: boolean | null
  detectionId?: string
}
