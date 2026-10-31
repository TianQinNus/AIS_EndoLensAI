import { useCallback, useEffect, useState } from 'react'
import { DetectionPanel } from '../components/DetectionPanel'
import { KeyFrames } from '../components/KeyFrames'
import { LiveAnalysis } from '../components/LiveAnalysis'
import { VideoPlayer } from '../components/VideoPlayer'
import { VideoTimeline } from '../components/VideoTimeline'
import {
  TASK_STORAGE_KEY,
  assetUrl,
  cancelAnalysis,
  followAnalysis,
  getKeyFrames,
  getReport,
  startAnalysis,
  uploadVideo,
} from '../utils/endpoints'
import type {
  AnalysisStatus,
  BackendFrame,
  BackendVideo,
  KeyFrame,
  PolypDetection,
  VideoMetadata,
} from '../utils/dataQueries'

function toVideo(value: BackendVideo): VideoMetadata {
  return {
    id: value.video_id,
    name: value.filename,
    duration: value.duration_seconds,
    fps: value.fps ?? 0,
    totalFrames: value.total_frames,
    streamUrl: assetUrl(value.video_url) ?? value.video_url,
    demo: false,
  }
}

function frameStatusLabel(frame: BackendFrame) {
  if (frame.status === 'failed') return 'Failed'
  if (frame.is_candidate === true) return 'Candidate frame'
  if (frame.is_candidate === false) return 'Not a candidate'
  return 'Unknown'
}

function toDetection(frame: BackendFrame): PolypDetection {
  return {
    id: frame.frame_id,
    frameIndex: frame.frame_index,
    timestamp: frame.timestamp_seconds,
    confidence: frame.foreground_score,
    contour: frame.contours[0]?.map(([x, y]) => ({ x, y })) ?? [],
    status: frameStatusLabel(frame),
    isCandidate: frame.is_candidate,
    frameStatus: frame.status,
    maskUrl: assetUrl(frame.mask_url),
    sourceUrl: assetUrl(frame.source_frame_url),
    thumbnailUrl: assetUrl(frame.thumbnail_url),
    width: frame.width,
    height: frame.height,
  }
}

function toKeyFrame(frame: {
  frame_id: string
  timestamp_seconds: number
  thumbnail_url: string | null
  is_candidate: boolean | null
}): KeyFrame {
  return {
    id: frame.frame_id,
    timestamp: frame.timestamp_seconds,
    label: formatTime(frame.timestamp_seconds),
    thumbnailUrl: assetUrl(frame.thumbnail_url),
    isCandidate: frame.is_candidate,
  }
}

function AnalysisPage() {
  const [video, setVideo] = useState<VideoMetadata>()
  const [taskId, setTaskId] = useState('')
  const [task, setTask] = useState<AnalysisStatus>()
  const [results, setResults] = useState<BackendFrame[]>([])
  const [frames, setFrames] = useState<KeyFrame[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [currentTime, setCurrentTime] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [mode, setMode] = useState<'saved' | 'live'>('saved')

  useEffect(() => {
    const saved = localStorage.getItem(TASK_STORAGE_KEY)
    if (!saved) return
    void getReport(saved)
      .then((report) => {
        if (report.video) setVideo(toVideo(report.video))
        setTaskId(saved)
      })
      .catch(() => localStorage.removeItem(TASK_STORAGE_KEY))
  }, [])

  useEffect(() => {
    if (!taskId) return
    const controller = new AbortController()
    void followAnalysis(
      taskId,
      (status, nextFrames) => {
        setTask(status)
        setResults(nextFrames)
        void getKeyFrames(taskId)
          .then((page) => setFrames(page.keyframes.map(toKeyFrame)))
          .catch((reason: Error) => setError(reason.message))
      },
      controller.signal,
    ).catch((reason: Error) => setError(reason.message))
    return () => controller.abort()
  }, [taskId])

  const handleUpload = async (file: File) => {
    setBusy(true)
    setError('')
    try {
      const uploaded = await uploadVideo(file)
      const created = await startAnalysis(uploaded.video_id, 1)
      localStorage.setItem(TASK_STORAGE_KEY, created.task_id)
      setVideo(toVideo(uploaded))
      setTaskId(created.task_id)
      setResults([])
      setFrames([])
      setSelectedId('')
      setCurrentTime(0)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to start analysis')
    } finally {
      setBusy(false)
    }
  }

  const handleSelect = useCallback((detection: PolypDetection) => {
    setSelectedId(detection.id)
    setCurrentTime(detection.timestamp)
  }, [])

  const detections = results.map(toDetection)
  const selected = results.find((item) => item.frame_id === selectedId) ?? results.find((item) => item.is_candidate)
  const progress =
    !task
      ? ''
      : task.progress.percent == null
        ? 'Indexing frames'
        : `${task.status} · ${task.progress.percent.toFixed(0)}%`

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="-mx-2.5 -mt-[9px] mb-2 flex min-h-10 shrink-0 items-end gap-1 bg-[#dee1e6] px-2 pt-1.5" role="tablist" aria-label="Analysis mode">
        {(
          [
            ['saved', 'Video Analysis'],
            ['live', 'Live Detection'],
          ] as const
        ).map(([id, label]) => {
          const selected = mode === id
          return (
            <button
              key={id}
              className={`whitespace-nowrap rounded-t-lg px-4 py-2 text-xs ${selected ? 'bg-slate-50 font-medium text-[#15243a]' : 'text-[#3c4043] hover:bg-[#e8eaed]'}`}
              aria-selected={selected}
              onClick={() => setMode(id)}
              role="tab"
              type="button"
            >
              {label}
            </button>
          )
        })}
      </div>
      <div className={`${mode === 'saved' ? 'flex' : 'hidden'} min-h-0 flex-1 flex-col`} role="tabpanel">
      <section className="mb-2 flex shrink-0 flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-white p-2 shadow-sm">
        <label className="cursor-pointer rounded bg-[#1677f2] px-3 py-2 text-xs font-medium text-white">
          {busy ? 'Uploading…' : 'Upload video'}
          <input
            className="hidden"
            type="file"
            accept=".mp4,.mov,.m4v,.avi,.mkv,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) void handleUpload(file)
              event.target.value = ''
            }}
          />
        </label>
        {video && <span className="text-xs text-slate-600">{video.name}</span>}
        {task && (
          <span className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">{progress}</span>
        )}
        {task && ['queued', 'running'].includes(task.status) && (
          <button
            className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-700"
            onClick={() => void cancelAnalysis(task.task_id).catch((reason: Error) => setError(reason.message))}
            type="button"
          >
            Cancel analysis
          </button>
        )}
        {error && <span className="text-xs text-red-600">{error}</span>}
      </section>
      {!video ? (
        <div className="rounded-md border border-dashed border-slate-300 bg-white p-12 text-center text-sm text-slate-500">
          Upload a video up to 512MB. Supported formats: MP4, MOV, M4V, AVI, and MKV.
        </div>
      ) : (
        <>
          <div className="grid min-h-0 flex-1 gap-[9px] lg:grid-cols-[minmax(450px,2fr)_minmax(330px,1fr)]">
            <VideoPlayer
              video={video}
              frame={selected}
              currentTime={currentTime}
              onTimeChange={setCurrentTime}
            />
            <DetectionPanel
              detections={detections}
              selectedId={selected?.frame_id ?? ''}
              candidateFrames={task?.progress.candidate_frames ?? 0}
              processingFps={task?.performance?.processing_fps ?? null}
              totalFrames={video.totalFrames}
              onSelect={handleSelect}
            />
          </div>
          <VideoTimeline
            currentTime={currentTime}
            detections={detections}
            candidateFrames={task?.progress.candidate_frames ?? 0}
            onSeek={setCurrentTime}
            duration={video.duration}
          />
          <KeyFrames
            frames={frames}
            selectedTime={currentTime}
            onSelect={(time) => {
              setCurrentTime(time)
              const match = results.find((item) => Math.abs(item.timestamp_seconds - time) < 0.05)
              if (match) setSelectedId(match.frame_id)
            }}
          />
        </>
      )}
      </div>
      <div className={`${mode === 'live' ? 'flex' : 'hidden'} min-h-0 flex-1 flex-col overflow-hidden`} role="tabpanel">
        <LiveAnalysis />
      </div>
    </div>
  )
}

function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0')}`
}

export default AnalysisPage
