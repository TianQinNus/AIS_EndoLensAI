import { useEffect, useRef, useState } from 'react'
import type { FrameResult } from '../utils/endpoints'
import type { VideoMetadata } from '../utils/dataQueries'
import { FrameOverlay } from './FrameOverlay'

interface VideoPlayerProps {
  video: VideoMetadata
  frame?: FrameResult
  currentTime: number
  onTimeChange: (time: number) => void
}

export function VideoPlayer({ video, frame, currentTime, onTimeChange }: VideoPlayerProps) {
  const media = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    setPlaying(false)
  }, [frame?.frame_id])

  useEffect(() => {
    const element = media.current
    if (!element) return
    if (playing) void element.play().catch(() => setPlaying(false))
    else element.pause()
  }, [playing])

  useEffect(() => {
    const element = media.current
    if (!element || playing) return
    if (Math.abs(element.currentTime - currentTime) > 0.2) element.currentTime = currentTime
  }, [currentTime, playing])

  const showExactFrame =
    !playing &&
    frame?.status === 'completed' &&
    Boolean(frame.source_frame_url) &&
    Math.abs(frame.timestamp_seconds - currentTime) < 0.05

  return (
    <section className="overflow-hidden rounded-md bg-[#0b1724] p-[9px]">
      <div className="relative aspect-[1.68] overflow-hidden rounded-md bg-black">
        <video
          ref={media}
          className={`absolute inset-0 h-full w-full object-contain ${showExactFrame ? 'invisible' : ''}`}
          src={video.streamUrl}
          muted
          playsInline
          aria-label={video.name}
          onTimeUpdate={(event) => {
            if (playing) onTimeChange(event.currentTarget.currentTime)
          }}
          onEnded={() => setPlaying(false)}
        />
        {showExactFrame && frame && (
          <div className="absolute inset-0 z-[6] flex items-center justify-center bg-black">
            <FrameOverlay frame={frame} />
          </div>
        )}
        <div className="absolute bottom-3 left-3 z-[7] rounded bg-[rgba(7,17,29,.72)] px-[11px] py-[7px] font-mono text-[11px] leading-[1.5] text-white">
          Frame: {frame ? frame.frame_index + 1 : '—'}
          <br />
          Time: {formatTime(currentTime)}
          <br />
          AI result, not a diagnosis
        </div>
      </div>
      <div className="flex items-center gap-3.5 px-2 pt-[9px] text-xs text-[#e9f1f8]">
        <button
          className="bg-transparent p-0 text-[21px] text-white"
          onClick={() => setPlaying((value) => !value)}
          aria-label={playing ? 'Pause' : 'Play'}
          type="button"
        >
          {playing ? 'Ⅱ' : '▶'}
        </button>
        <span>
          {formatTime(currentTime)} / {formatTime(video.duration)}
        </span>
        <input
          className="h-[5px] flex-1 accent-[#2e83f8]"
          type="range"
          min="0"
          max={video.duration || 0}
          value={Math.min(currentTime, video.duration || 0)}
          onChange={(event) => {
            setPlaying(false)
            onTimeChange(Number(event.target.value))
          }}
          aria-label="Video progress"
        />
        <span>{video.fps ? `${video.fps.toFixed(1)} playback FPS` : 'Playback FPS —'}</span>
      </div>
    </section>
  )
}

function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0')}`
}
