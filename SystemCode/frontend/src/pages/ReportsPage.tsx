import { useEffect, useRef, useState } from 'react'
import { ReportCard } from '../components/ReportCard'
import { TASK_STORAGE_KEY, assetUrl, exportCsv, getFrames, getReport } from '../utils/endpoints'
import type { BackendFrame, BackendReport } from '../utils/dataQueries'

function clockName(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(whole / 60)
  const secs = whole % 60
  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

function imageNames(frames: BackendFrame[]) {
  const used = new Map<string, number>()
  return frames.map((frame) => {
    const base = clockName(frame.timestamp_seconds)
    const count = used.get(base) ?? 0
    used.set(base, count + 1)
    return count === 0 ? base : `${base} (${count + 1})`
  })
}

function fileNameFor(label: string) {
  return `${label.replaceAll(':', '-')}.png`
}

async function loadCandidateFrames(taskId: string) {
  const frames: BackendFrame[] = []
  let after: string | null = null
  for (;;) {
    const page = await getFrames(taskId, after, true)
    frames.push(...page.items.filter((frame) => frame.is_candidate === true && frame.status === 'completed'))
    if (!page.has_more || !page.next_cursor) return frames
    after = page.next_cursor
  }
}

const paintQueue: Array<() => void> = []
let paintsRunning = 0

function enqueuePaint<T>(job: () => Promise<T>) {
  return new Promise<T>((resolve, reject) => {
    const start = () => {
      paintsRunning += 1
      job().then(resolve, reject).finally(() => {
        paintsRunning -= 1
        paintQueue.shift()?.()
      })
    }
    if (paintsRunning < 3) start()
    else paintQueue.push(start)
  })
}

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image()
    image.crossOrigin = 'anonymous'
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Image failed'))
    image.src = url
  })
}

async function paintModelMask(
  frame: BackendFrame,
  canvas: HTMLCanvasElement,
  cancelled: () => boolean,
  maxEdge = Math.max(frame.width, frame.height),
) {
  const sourceUrl = assetUrl(frame.source_frame_url)
  const maskUrl = assetUrl(frame.mask_url)
  if (!sourceUrl || !maskUrl) return false
  const scale = Math.min(1, maxEdge / Math.max(frame.width, frame.height))
  const width = Math.max(1, Math.round(frame.width * scale))
  const height = Math.max(1, Math.round(frame.height * scale))
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) return false
  const source = await loadImage(`${sourceUrl}${sourceUrl.includes('?') ? '&' : '?'}cors=1`)
  if (cancelled()) return false
  context.drawImage(source, 0, 0, width, height)
  const mask = await loadImage(`${maskUrl}${maskUrl.includes('?') ? '&' : '?'}cors=1`)
  if (cancelled()) return false
  const layer = document.createElement('canvas')
  layer.width = width
  layer.height = height
  const layerContext = layer.getContext('2d')
  if (!layerContext) return false
  layerContext.drawImage(mask, 0, 0, width, height)
  const pixels = layerContext.getImageData(0, 0, width, height)
  for (let index = 0; index < pixels.data.length; index += 4) {
    const foreground = pixels.data[index] > 127
    pixels.data[index] = 0
    pixels.data[index + 1] = 240
    pixels.data[index + 2] = 110
    pixels.data[index + 3] = foreground ? 85 : 0
  }
  layerContext.putImageData(pixels, 0, 0)
  context.drawImage(layer, 0, 0)
  return true
}

function CandidateFrame({
  frame,
  label,
  onError,
}: {
  frame: BackendFrame
  label: string
  onError: (message: string) => void
}) {
  const card = useRef<HTMLElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [visible, setVisible] = useState(false)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const node = card.current
    if (!node) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) setVisible(true)
      },
      { rootMargin: '240px' },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const target = canvas.current
    if (!visible || !target) return
    let cancelled = false
    setReady(false)
    void enqueuePaint(() => paintModelMask(frame, target, () => cancelled, 480))
      .then((painted) => {
        if (!cancelled) setReady(painted)
      })
      .catch(() => {
        if (!cancelled) setReady(false)
      })
    return () => {
      cancelled = true
    }
  }, [frame, visible])

  const download = () => {
    if (!ready) return
    const exportCanvas = document.createElement('canvas')
    void enqueuePaint(() => paintModelMask(frame, exportCanvas, () => false))
      .then((painted) => {
        if (!painted) throw new Error(`Could not download ${label}`)
        exportCanvas.toBlob((blob) => {
          if (!blob) {
            onError(`Could not download ${label}`)
            return
          }
          const objectUrl = URL.createObjectURL(blob)
          const link = document.createElement('a')
          link.href = objectUrl
          link.download = fileNameFor(label)
          link.click()
          URL.revokeObjectURL(objectUrl)
        }, 'image/png')
      })
      .catch((reason: Error) => onError(reason.message))
  }

  return (
    <figure ref={card} className="m-0 overflow-hidden rounded-md border border-slate-200">
      <figcaption className="flex items-center justify-between gap-2 border-b border-slate-200 bg-slate-50 px-3 py-2">
        <span className="text-sm font-semibold text-[#112a47]">{label}</span>
        <button
          className="text-[11px] font-medium text-[#1677f2] disabled:text-slate-400"
          disabled={!ready}
          onClick={download}
          type="button"
        >
          Download
        </button>
      </figcaption>
      <canvas
        ref={canvas}
        aria-label={`${label} with model mask`}
        className="block h-auto w-full bg-black"
      />
    </figure>
  )
}

function ReportsPage() {
  const [report, setReport] = useState<BackendReport>()
  const [frames, setFrames] = useState<BackendFrame[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const taskId = localStorage.getItem(TASK_STORAGE_KEY)
    if (!taskId) {
      setLoading(false)
      return
    }
    let cancelled = false
    void Promise.all([getReport(taskId), loadCandidateFrames(taskId)])
      .then(([saved, candidates]) => {
        if (cancelled) return
        setReport(saved)
        setFrames(candidates)
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(reason.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const download = async () => {
    const taskId = localStorage.getItem(TASK_STORAGE_KEY)
    if (!taskId) return
    const exported = await exportCsv(taskId)
    const url = assetUrl(exported.download_url)
    if (url) window.open(url, '_blank', 'noopener')
  }

  const names = imageNames(frames)
  const filename = report?.video?.filename

  return (
    <div className="mx-auto max-w-[1100px]">
      {filename && (
        <section className="mb-3 flex items-center gap-3 rounded-md border border-[#c5d4e8] bg-white px-4 py-3 shadow-sm">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-[#66758a]">Video file</span>
          <span className="min-w-0 truncate text-base font-semibold text-[#112a47]" title={filename}>
            {filename}
          </span>
        </section>
      )}
      <section className="mb-2 rounded-lg bg-gradient-to-r from-[#112a47] to-[#1d5f9b] p-5 text-white">
        <h1 className="mb-1 text-[22px]">Analysis Reports</h1>
        <p className="text-xs text-[#c8e0f9]">
          Frames judged to contain a mask are shown below. Each image is named by its time in the video.
        </p>
      </section>
      {loading ? (
        <section className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
          Loading the saved report.
        </section>
      ) : !report ? (
        <section className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
          {error || 'Complete a saved-video or live analysis to view its saved report.'}
        </section>
      ) : (
        <>
          <div className="mb-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <ReportCard label="Analyzed frames" value={String(report.summary.analyzed_frames)} />
            <ReportCard label="Candidate frames" value={String(report.summary.candidate_frames)} />
            <ReportCard label="Failed frames" value={String(report.summary.failed_frames)} />
            <ReportCard label="Reviewed frames" value={String(report.summary.reviewed_frames)} />
          </div>
          <section className="mb-3 rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                Status: {report.status}. Task {report.task_id}.
              </div>
              <button
                className="rounded bg-[#1677f2] px-3 py-2 text-xs font-medium text-white"
                onClick={() => void download().catch((reason: Error) => setError(reason.message))}
                type="button"
              >
                Export CSV
              </button>
            </div>
            {error && <p className="mb-2 text-xs text-red-600">{error}</p>}
            <ul className="list-disc pl-5 text-xs text-slate-500">
              {report.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </section>
          <section className="rounded-md border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 text-sm font-bold text-[#27384f]">
              Candidate frames ({frames.length})
            </h2>
            {frames.length === 0 ? (
              <p className="text-sm text-slate-500">No analyzed frame in this video was marked as a candidate.</p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {frames.map((frame, index) => (
                  <CandidateFrame
                    frame={frame}
                    key={frame.frame_id}
                    label={names[index]}
                    onError={setError}
                  />
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  )
}

export default ReportsPage
