import { useEffect, useRef, useState } from 'react'
import { assetUrl, inferLiveFrame, uploadVideo } from '../utils/endpoints'

const MAX_EDGE = 1280
const CANDIDATE_LIMIT = 16

type AnalyzedFrame = {
  frameIndex: number
  timestamp: number
  isCandidate: boolean
  foregroundScore: number | null
  inferenceMs: number
  roundTripMs: number
  snapshotUrl: string
  maskBase64: string
  width: number
  height: number
}

type LiveStats = {
  analyzed: number
  skipped: number
  candidates: number
  elapsedMs: number
  firstRoundTripMs: number | null
  firstInferenceMs: number | null
  lastRoundTripMs: number | null
  lastInferenceMs: number | null
  device: string
}

const emptyStats: LiveStats = {
  analyzed: 0,
  skipped: 0,
  candidates: 0,
  elapsedMs: 0,
  firstRoundTripMs: null,
  firstInferenceMs: null,
  lastRoundTripMs: null,
  lastInferenceMs: null,
  device: '',
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Image failed'))
    image.src = src
  })
}

function paintMask(
  ctx: CanvasRenderingContext2D,
  mask: CanvasImageSource,
  width: number,
  height: number,
) {
  const layer = document.createElement('canvas')
  layer.width = width
  layer.height = height
  const layerContext = layer.getContext('2d')
  if (!layerContext) return
  layerContext.drawImage(mask, 0, 0, width, height)
  const rgba = layerContext.getImageData(0, 0, width, height)
  for (let i = 0; i < rgba.data.length; i += 4) {
    const foreground = rgba.data[i] > 127
    rgba.data[i] = 0
    rgba.data[i + 1] = 240
    rgba.data[i + 2] = 110
    rgba.data[i + 3] = foreground ? 110 : 0
  }
  layerContext.putImageData(rgba, 0, 0)
  ctx.drawImage(layer, 0, 0)
}

function AnalyzedView({ frame, maxWidth }: { frame: AnalyzedFrame; maxWidth?: number }) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    const target = canvas.current
    if (!target) return
    const ctx = target.getContext('2d')
    if (!ctx) return
    void (async () => {
      const source = await loadImage(frame.snapshotUrl)
      const mask = await loadImage(`data:image/png;base64,${frame.maskBase64}`)
      if (cancelled) return
      const scale = maxWidth ? Math.min(1, maxWidth / frame.width) : 1
      const width = Math.max(1, Math.round(frame.width * scale))
      const height = Math.max(1, Math.round(frame.height * scale))
      target.width = width
      target.height = height
      ctx.clearRect(0, 0, width, height)
      ctx.drawImage(source, 0, 0, width, height)
      paintMask(ctx, mask, width, height)
    })().catch((error) => {
      if (!cancelled) console.error('Live overlay failed', error)
    })
    return () => {
      cancelled = true
    }
  }, [frame, maxWidth])

  return (
    <canvas
      ref={canvas}
      aria-label={`Analyzed frame ${frame.frameIndex} with its own mask`}
      className="block h-auto max-h-full w-full object-contain"
    />
  )
}

function grabFrame(video: HTMLVideoElement) {
  const srcW = video.videoWidth
  const srcH = video.videoHeight
  if (!srcW || !srcH) return null
  const scale = Math.min(1, MAX_EDGE / Math.max(srcW, srcH))
  const width = Math.max(1, Math.round(srcW * scale))
  const height = Math.max(1, Math.round(srcH * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(video, 0, 0, width, height)
  return { canvas, width, height }
}

function canvasToJpeg(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Could not encode frame'))),
      'image/jpeg',
      0.82,
    )
  })
}

function watchPlayback(
  video: HTMLVideoElement,
  onFrame: (frame: { frameIndex: number; timestamp: number }) => void,
) {
  const withCallback = video as HTMLVideoElement & {
    requestVideoFrameCallback?: (
      callback: (now: number, metadata: { mediaTime: number; presentedFrames: number }) => void,
    ) => number
    cancelVideoFrameCallback?: (handle: number) => void
  }
  if (withCallback.requestVideoFrameCallback && withCallback.cancelVideoFrameCallback) {
    let handle = 0
    const tick = (_now: number, metadata: { mediaTime: number; presentedFrames: number }) => {
      onFrame({ frameIndex: metadata.presentedFrames, timestamp: metadata.mediaTime })
      handle = withCallback.requestVideoFrameCallback!(tick)
    }
    handle = withCallback.requestVideoFrameCallback(tick)
    return () => withCallback.cancelVideoFrameCallback!(handle)
  }
  let index = 0
  let lastTime = -1
  let raf = 0
  const tick = () => {
    if (video.readyState >= 2 && video.currentTime !== lastTime) {
      lastTime = video.currentTime
      onFrame({ frameIndex: index, timestamp: video.currentTime })
      index += 1
    }
    raf = window.requestAnimationFrame(tick)
  }
  raf = window.requestAnimationFrame(tick)
  return () => window.cancelAnimationFrame(raf)
}

function formatClock(seconds: number) {
  const abs = Math.abs(seconds)
  const mins = Math.floor(abs / 60)
  const secs = (abs % 60).toFixed(1).padStart(4, '0')
  return `${seconds < 0 ? '-' : ''}${mins}:${secs}`
}

function formatMs(value: number | null) {
  return value == null ? '—' : `${Math.round(value)} ms`
}

export function LiveAnalysis() {
  const media = useRef<HTMLVideoElement>(null)
  const abortRef = useRef(new AbortController())
  const generation = useRef(0)
  const requestLock = useRef(false)
  const lastSent = useRef(-1)
  const skipped = useRef(0)
  const startedAt = useRef<number | null>(null)
  const retained = useRef(new Set<string>())
  const pinnedRef = useRef(false)
  const latestRef = useRef<AnalyzedFrame | null>(null)
  const currentRef = useRef<AnalyzedFrame | null>(null)
  const candidatesRef = useRef<AnalyzedFrame[]>([])
  const sourceRef = useRef('')
  const localFile = useRef<File | null>(null)
  const fileToken = useRef(0)
  const converting = useRef(false)
  const [source, setSource] = useState('')
  const [fileName, setFileName] = useState('')
  const [playing, setPlaying] = useState(false)
  const [playbackTime, setPlaybackTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [current, setCurrent] = useState<AnalyzedFrame | null>(null)
  const [pinned, setPinned] = useState(false)
  const [candidates, setCandidates] = useState<AnalyzedFrame[]>([])
  const [stats, setStats] = useState<LiveStats>(emptyStats)
  const [error, setError] = useState('')
  const [preparing, setPreparing] = useState('')

  useEffect(() => {
    pinnedRef.current = pinned
  }, [pinned])

  useEffect(() => {
    const urls = retained.current
    return () => {
      generation.current += 1
      abortRef.current.abort()
      urls.forEach((url) => URL.revokeObjectURL(url))
      urls.clear()
      if (sourceRef.current) URL.revokeObjectURL(sourceRef.current)
    }
  }, [])

  useEffect(() => {
    const video = media.current
    if (!video || !source) return
    if (playing) {
      video.playbackRate = 1
      void video.play().catch(() => setPlaying(false))
    } else {
      video.pause()
    }
  }, [playing, source])

  useEffect(() => {
    const video = media.current
    if (!video || !source || !playing) return
    let stopped = false
    let newest: { frameIndex: number; timestamp: number } | null = null
    const pump = () => {
      if (stopped || requestLock.current || !newest || newest.frameIndex === lastSent.current) return
      const grabbed = grabFrame(video)
      if (!grabbed) return
      const meta = newest
      if (lastSent.current >= 0) {
        const gap = meta.frameIndex - lastSent.current - 1
        if (gap > 0) skipped.current += gap
      }
      requestLock.current = true
      lastSent.current = meta.frameIndex
      const sentIndex = meta.frameIndex
      const sentTime = meta.timestamp
      const gen = generation.current
      const signal = abortRef.current.signal
      void (async () => {
        let previewUrl = ''
        try {
          const blob = await canvasToJpeg(grabbed.canvas)
          if (stopped || gen !== generation.current) return
          previewUrl = URL.createObjectURL(blob)
          if (startedAt.current == null) startedAt.current = performance.now()
          const started = performance.now()
          const result = await inferLiveFrame(blob, sentIndex, sentTime, signal)
          const roundTripMs = performance.now() - started
          const aligned =
            result.frame_index === sentIndex &&
            Math.abs(result.timestamp_seconds - Number(sentTime.toFixed(6))) < 1e-3 &&
            result.width === grabbed.width &&
            result.height === grabbed.height
          if (gen !== generation.current || !aligned) {
            if (gen === generation.current) {
              setError('Result did not match the submitted frame, so its mask was discarded.')
            }
            URL.revokeObjectURL(previewUrl)
            return
          }
          const analyzed: AnalyzedFrame = {
            frameIndex: result.frame_index,
            timestamp: result.timestamp_seconds,
            isCandidate: result.is_candidate,
            foregroundScore: result.foreground_score,
            inferenceMs: result.inference_time_ms,
            roundTripMs,
            snapshotUrl: previewUrl,
            maskBase64: result.mask_png_base64,
            width: result.width,
            height: result.height,
          }
          retained.current.add(previewUrl)
          const previousLatest = latestRef.current
          const nextCandidates = analyzed.isCandidate
            ? [...candidatesRef.current, analyzed].slice(-CANDIDATE_LIMIT)
            : candidatesRef.current
          const nextCurrent = pinnedRef.current && currentRef.current ? currentRef.current : analyzed
          const keep = new Set<string>([
            nextCurrent.snapshotUrl,
            analyzed.snapshotUrl,
            ...nextCandidates.map((item) => item.snapshotUrl),
          ])
          ;[previousLatest?.snapshotUrl, ...candidatesRef.current.map((item) => item.snapshotUrl)].forEach(
            (url) => {
              if (!url || keep.has(url)) return
              URL.revokeObjectURL(url)
              retained.current.delete(url)
            },
          )
          latestRef.current = analyzed
          currentRef.current = nextCurrent
          candidatesRef.current = nextCandidates
          setError('')
          setCurrent(nextCurrent)
          setCandidates(nextCandidates)
          setStats((previous) => ({
            analyzed: previous.analyzed + 1,
            skipped: skipped.current,
            candidates: previous.candidates + (analyzed.isCandidate ? 1 : 0),
            elapsedMs: performance.now() - (startedAt.current ?? performance.now()),
            firstRoundTripMs: previous.firstRoundTripMs ?? roundTripMs,
            firstInferenceMs: previous.firstInferenceMs ?? result.inference_time_ms,
            lastRoundTripMs: roundTripMs,
            lastInferenceMs: result.inference_time_ms,
            device: result.device,
          }))
        } catch (reason) {
          if (previewUrl) URL.revokeObjectURL(previewUrl)
          if (signal.aborted || gen !== generation.current) return
          setError(reason instanceof Error ? reason.message : 'Frame inference failed')
        } finally {
          if (gen === generation.current) {
            requestLock.current = false
            pump()
          }
        }
      })()
    }
    const stopWatch = watchPlayback(video, (meta) => {
      newest = meta
      pump()
    })
    return () => {
      stopped = true
      stopWatch()
    }
  }, [playing, source])

  const resetSession = () => {
    generation.current += 1
    abortRef.current.abort()
    abortRef.current = new AbortController()
    requestLock.current = false
    lastSent.current = -1
    skipped.current = 0
    startedAt.current = null
    pinnedRef.current = false
    latestRef.current = null
    currentRef.current = null
    candidatesRef.current = []
    retained.current.forEach((url) => URL.revokeObjectURL(url))
    retained.current.clear()
    setCurrent(null)
    setPinned(false)
    setCandidates([])
    setStats(emptyStats)
    setError('')
    setPreparing('')
    setPlaybackTime(0)
  }

  const convertLocalFile = async () => {
    const file = localFile.current
    const token = fileToken.current
    if (!file || converting.current) return
    converting.current = true
    setPlaying(false)
    setPreparing('Converting to MP4…')
    setError('')
    try {
      const uploaded = await uploadVideo(file)
      if (token !== fileToken.current) return
      const url = assetUrl(uploaded.video_url)
      if (!url) throw new Error('Converted video has no playback URL')
      sourceRef.current = url
      setDuration(uploaded.duration_seconds || 0)
      setSource(url)
      setPreparing('')
      setPlaying(true)
    } catch (reason) {
      if (token !== fileToken.current) return
      setPreparing('')
      setError(reason instanceof Error ? reason.message : 'Could not convert this recording')
    } finally {
      if (token === fileToken.current) converting.current = false
    }
  }

  const chooseFile = (file: File) => {
    fileToken.current += 1
    converting.current = false
    localFile.current = file
    resetSession()
    if (sourceRef.current.startsWith('blob:')) URL.revokeObjectURL(sourceRef.current)
    sourceRef.current = ''
    setFileName(file.name)
    setDuration(0)
    setSource('')
    void convertLocalFile()
  }

  const rate =
    stats.analyzed >= 2 && stats.elapsedMs > 0 ? stats.analyzed / (stats.elapsedMs / 1000) : null
  const lag = current ? playbackTime - current.timestamp : 0

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <section className="mb-2 flex flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-white p-3 shadow-sm">
        <label className="cursor-pointer rounded bg-[#1677f2] px-3 py-2 text-xs font-medium text-white">
          Upload video
          <input
            className="hidden"
            type="file"
            accept=".mp4,.mov,.m4v,.avi,.mkv,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) chooseFile(file)
              event.target.value = ''
            }}
          />
        </label>
        {fileName && <span className="text-xs text-slate-600">{fileName}</span>}
        {source && (
          <button
            className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 disabled:text-slate-400"
            disabled={Boolean(preparing)}
            onClick={() => setPlaying((value) => !value)}
            type="button"
          >
            {preparing ? 'Converting…' : playing ? 'Pause' : 'Play at original speed'}
          </button>
        )}
        {error && <span className="text-xs text-red-600">{error}</span>}
      </section>
      <p className="mb-2 shrink-0 rounded-md border border-[#d7e6f8] bg-[#f4f8fd] px-3 py-2 text-xs leading-5 text-[#3d5168]">
        Every recording is converted to MP4 before playback. It then plays at its original speed,
        and each request sends one frame together with its frame number and timestamp. Frames
        presented while a request is in flight are skipped, and the next request sends the newest
        frame. The mask is drawn on that analyzed frame.
      </p>
      {!source ? (
        <div className="rounded-md border border-dashed border-slate-300 bg-white p-12 text-center text-sm text-slate-500">
          {preparing ||
            'Choose a local recording. It is converted to MP4, then frames are submitted while it plays.'}
        </div>
      ) : (
        <>
          <div className="grid min-h-0 flex-1 gap-[9px] lg:grid-cols-2">
            <section className="overflow-hidden rounded-md border border-slate-200 bg-white p-[9px] shadow-sm">
              <div className="mb-2 text-[11px] font-medium text-[#526277]">Playback</div>
              <div className="relative aspect-[1.68] overflow-hidden rounded-md bg-white">
                <video
                  ref={media}
                  className="absolute inset-0 h-full w-full object-contain"
                  src={source}
                  crossOrigin="anonymous"
                  controls
                  playsInline
                  aria-label={fileName}
                  onRateChange={(event) => {
                    if (event.currentTarget.playbackRate !== 1) event.currentTarget.playbackRate = 1
                  }}
                  onTimeUpdate={(event) => setPlaybackTime(event.currentTarget.currentTime)}
                  onLoadedMetadata={(event) => {
                    const mediaElement = event.currentTarget
                    setDuration(Number.isFinite(mediaElement.duration) ? mediaElement.duration : 0)
                  }}
                  onError={() => {
                    if (!preparing) setError('The converted MP4 could not be played.')
                  }}
                  onEnded={() => setPlaying(false)}
                />
                {preparing && (
                  <div className="absolute inset-0 z-[8] flex items-center justify-center bg-white/85 px-6 text-center text-xs text-[#3d5168]">
                    {preparing}
                  </div>
                )}
                <div className="absolute bottom-3 left-3 z-[7] rounded bg-[rgba(7,17,29,.72)] px-[11px] py-[7px] font-mono text-[11px] leading-[1.5] text-white">
                  Playback {formatClock(playbackTime)} / {formatClock(duration)}
                  <br />
                  Original speed
                  <br />
                  AI result, not a diagnosis
                </div>
              </div>
            </section>
            <section className="overflow-hidden rounded-md border border-slate-200 bg-white shadow-sm">
              <div className="flex items-center justify-between border-b border-[#e3eaf2] px-3 py-2">
                <span className="text-xs font-bold text-[#1677f2]">Analyzed frame</span>
                {pinned && (
                  <button
                    className="rounded border border-slate-300 px-2 py-1 text-[11px] text-slate-700"
                    onClick={() => {
                      pinnedRef.current = false
                      setPinned(false)
                      if (latestRef.current) {
                        currentRef.current = latestRef.current
                        setCurrent(latestRef.current)
                      }
                    }}
                    type="button"
                  >
                    Follow latest
                  </button>
                )}
              </div>
              <div className="relative aspect-[1.68] bg-white">
                {current ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <AnalyzedView frame={current} />
                  </div>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-xs text-[#66758a]">
                    {preparing ||
                      (playing
                        ? 'Measuring the first single-frame round trip.'
                        : 'Press play to send the first frame.')}
                  </div>
                )}
              </div>
              <div className="space-y-1 px-3 py-2 text-[11px] leading-5 text-[#3d5168]">
                {current ? (
                  <>
                    <div>
                      Frame {current.frameIndex} · {formatClock(current.timestamp)} ·{' '}
                      {current.isCandidate ? 'Candidate' : 'No candidate region'} · score{' '}
                      {current.foregroundScore == null ? '—' : current.foregroundScore.toFixed(2)}
                    </div>
                    <div>
                      {lag > 0.2
                        ? `Playback is ${formatClock(lag)} ahead of this result.`
                        : 'This result matches the frame that was submitted.'}
                    </div>
                  </>
                ) : (
                  <div>Results stay paired with the frame that produced them.</div>
                )}
              </div>
            </section>
          </div>
          <section className="mt-[9px] grid gap-2 sm:grid-cols-4">
            {[
              [formatMs(stats.firstInferenceMs), 'First model time', stats.device || 'single inference'],
              [formatMs(stats.firstRoundTripMs), 'First round trip', 'upload + model + response'],
              [rate == null ? '—' : rate.toFixed(1), 'Achieved analysis rate', 'completed frames / elapsed'],
              [String(stats.skipped), 'Skipped frames', `${stats.analyzed} analyzed`],
            ].map(([value, label, sub]) => (
              <div
                className="rounded-md border border-[#e4ebf3] bg-white p-2 text-center shadow-sm"
                key={label}
              >
                <strong className="block text-base text-[#15243a]">{value}</strong>
                <span className="block text-[10px] text-[#66758a]">
                  {label}
                  <br />
                  {sub}
                </span>
              </div>
            ))}
          </section>
          <p className="mt-2 text-[11px] text-[#66758a]">
            Latest model time {formatMs(stats.lastInferenceMs)}, latest round trip{' '}
            {formatMs(stats.lastRoundTripMs)}. The analysis rate follows that single-frame latency.
            Candidate frames: {stats.candidates}.
          </p>
          <section className="mt-[9px] rounded-md border border-slate-200 bg-white p-[11px_10px] shadow-sm">
            <div className="mb-2 text-xs font-bold text-[#27384f]">Detected frames</div>
            {candidates.length === 0 ? (
              <p className="text-[11px] text-[#66758a]">Candidate frames appear here with their masks.</p>
            ) : (
              <div className="flex items-start gap-[11px] overflow-x-auto">
                {candidates.map((frame) => {
                  const selected = current?.snapshotUrl === frame.snapshotUrl
                  return (
                    <button
                      className={`w-[120px] flex-none text-center text-[9px] ${selected ? 'font-bold text-[#1474e9]' : 'text-[#63748a]'}`}
                      key={frame.snapshotUrl}
                      onClick={() => {
                        pinnedRef.current = true
                        setPinned(true)
                        currentRef.current = frame
                        setCurrent(frame)
                      }}
                      type="button"
                    >
                      <span
                        className={`mb-[3px] block overflow-hidden rounded-md border-2 ${selected ? 'border-[#1779fa]' : 'border-[#fb8296]'}`}
                      >
                        <AnalyzedView frame={frame} maxWidth={160} />
                      </span>
                      <span>
                        {formatClock(frame.timestamp)} · {frame.foregroundScore == null ? '—' : frame.foregroundScore.toFixed(2)}
                      </span>
                    </button>
                  )
                })}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  )
}
