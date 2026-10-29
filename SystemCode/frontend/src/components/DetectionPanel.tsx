import type { PolypDetection } from '../utils/dataQueries'

interface DetectionPanelProps {
  detections: PolypDetection[]
  selectedId: string
  candidateFrames: number
  processingFps: number | null
  totalFrames: number | null
  onSelect: (detection: PolypDetection) => void
}

export function DetectionPanel({
  detections,
  selectedId,
  candidateFrames,
  processingFps,
  totalFrames,
  onSelect,
}: DetectionPanelProps) {
  const selected = detections.find((item) => item.id === selectedId) ?? detections[0]
  const fpsLabel = processingFps == null || processingFps === 0 ? '—' : processingFps.toFixed(1)
  return (
    <section className="min-w-0 rounded-md border border-slate-200 bg-white p-[13px] shadow-sm">
      <div className="mb-3 -mx-[13px] flex h-[43px] items-center border-b border-[#e3eaf2] px-[13px]">
        <span className="flex h-full items-center border-b-2 border-[#1677f2] text-xs font-bold text-[#1677f2]">
          Sampled frames
        </span>
      </div>
      <div className="grid grid-cols-2 gap-[7px]">
        {[
          [String(candidateFrames), 'Candidate Frames', '(Saved count)'],
          [fpsLabel, 'Processing FPS', '(Sampled frames / time)'],
        ].map(([value, label, sub]) => (
          <div
            className="rounded-md border border-[#e4ebf3] bg-gradient-to-b from-[#f6f9fd] to-[#edf3fa] p-[7px_4px] text-center"
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
      </div>
      <div className="mt-2.5 max-h-[360px] overflow-auto">
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr className="bg-[#f8fafd] text-left font-medium text-[#526277]">
              <th className="p-1.5">#</th>
              <th className="p-1.5">Preview</th>
              <th className="p-1.5">Foreground Score</th>
              <th className="p-1.5">Time</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {detections.map((item, index) => (
              <tr className="border-t border-[#edf1f5]" key={item.id}>
                <td className="p-1.5">{index + 1}</td>
                <td className="p-1.5">
                  <button
                    className={`border-0 bg-transparent text-[#506782] ${item.id === selectedId ? 'outline-2 outline-[#1b78f3] outline-offset-1' : ''}`}
                    onClick={() => onSelect(item)}
                    type="button"
                  >
                    {item.thumbnailUrl ? (
                      <img className="h-[41px] w-[49px] rounded-md object-cover" src={item.thumbnailUrl} alt="" />
                    ) : (
                      <span className="inline-block h-[41px] w-[49px] rounded-md bg-slate-200" />
                    )}
                  </button>
                </td>
                <td className="p-1.5">
                  <span className="inline-block rounded bg-[#d9f8e6] px-[7px] py-1 text-[#137044]">
                    {scoreText(item.confidence)}
                  </span>
                </td>
                <td className="p-1.5">{formatTime(item.timestamp)}</td>
                <td className="p-1.5">
                  <button
                    className="border-0 bg-transparent text-[#506782]"
                    onClick={() => onSelect(item)}
                    aria-label={`Select frame ${item.frameIndex + 1}`}
                    type="button"
                  >
                    ›
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selected && (
        <div className="mt-2 border-t border-[#e7edf3] pt-2">
          <div className="flex items-center justify-between text-xs font-bold">
            Selected frame
          </div>
          <div className="mt-2 grid grid-cols-[1.25fr_1fr] gap-2.5">
            {selected.thumbnailUrl ? (
              <img className="min-h-[92px] w-full rounded border border-[#dbe4ec] object-cover" src={selected.thumbnailUrl} alt="" />
            ) : (
              <div className="min-h-[92px] rounded border border-[#dbe4ec] bg-slate-100" />
            )}
            <div className="grid content-center gap-1.5 text-[10px] text-[#64748b]">
              <div>
                Foreground Score <b className="float-right text-[#22344d]">{scoreText(selected.confidence)}</b>
              </div>
              <div>
                Frame{' '}
                <b className="float-right text-[#22344d]">
                  {selected.frameIndex + 1} / {totalFrames ?? 'Unknown'}
                </b>
              </div>
              <div>
                Time <b className="float-right text-[#22344d]">{formatTime(selected.timestamp)}</b>
              </div>
              <div>
                Status <span className="float-right rounded bg-[#e8eef6] px-1 py-0.5 text-[#3a5270]">{selected.status}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

function scoreText(value: number | null) {
  return value == null ? '—' : value.toFixed(2)
}

function formatTime(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0')}`
}
