import type { PolypDetection } from '../utils/dataQueries'

interface VideoTimelineProps {
  currentTime: number
  detections: PolypDetection[]
  candidateFrames: number
  onSeek: (time: number) => void
  duration: number
}

export function VideoTimeline({
  currentTime,
  detections,
  candidateFrames,
  onSeek,
  duration,
}: VideoTimelineProps) {
  const safeDuration = duration || 1
  return (
    <section className="mt-[9px] rounded-md border border-slate-200 bg-white p-[11px_10px] shadow-sm">
      <div className="flex justify-between gap-3">
        <div className="mb-2 text-xs font-bold text-[#27384f]">Video Timeline</div>
        <div className="text-[10px] text-[#65758a]">Candidate frames: {candidateFrames}</div>
      </div>
      <button
        className="relative mx-4 my-1 block h-[17px] w-[calc(100%-2rem)] rounded-sm bg-[repeating-linear-gradient(90deg,#e4eef8_0_2px,transparent_2px_4px)]"
        onClick={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect()
          onSeek(((event.clientX - bounds.left) / bounds.width) * duration)
        }}
        aria-label="Seek video timeline"
        type="button"
      >
        {detections.map((detection) => (
          <i
            className={`absolute top-0 h-full w-[3px] ${markClass(detection)}`}
            key={detection.id}
            style={{ left: `${(detection.timestamp / safeDuration) * 100}%` }}
          />
        ))}
        <span
          className="absolute -top-1 h-[25px] w-0.5 bg-[#2879f1]"
          style={{ left: `${(currentTime / safeDuration) * 100}%` }}
        />
      </button>
      <div className="flex justify-between text-[9px] text-[#7a899c]">
        <span>00:00</span>
        <span>Pink: candidate. Gray: sampled, rule not met. Unsampled video is unknown.</span>
        <span>{formatTime(duration)}</span>
      </div>
    </section>
  )
}

function markClass(detection: PolypDetection) {
  if (detection.frameStatus === 'failed' || detection.isCandidate == null) return 'bg-[#f59e0b]'
  return detection.isCandidate ? 'bg-[#fb8296]' : 'bg-[#94a3b8]'
}

function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0')}`
}
