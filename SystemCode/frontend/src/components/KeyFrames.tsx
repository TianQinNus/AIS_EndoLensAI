import type { KeyFrame } from '../utils/dataQueries'

interface KeyFramesProps {
  frames: KeyFrame[]
  selectedTime: number
  onSelect: (time: number) => void
}

export function KeyFrames({ frames, selectedTime, onSelect }: KeyFramesProps) {
  return (
    <section className="mt-[9px] rounded-md border border-slate-200 bg-white p-[11px_10px] shadow-sm">
      <div className="mb-2 text-xs font-bold text-[#27384f]">Key Frames</div>
      <div className="flex items-center gap-[11px] overflow-x-auto">
        {frames.map((frame) => {
          const selected = Math.abs(frame.timestamp - selectedTime) < 0.05
          return (
            <button
              className={`min-w-[70px] flex-1 cursor-pointer text-center text-[9px] ${selected ? 'font-bold text-[#1474e9]' : 'text-[#63748a]'}`}
              key={frame.id}
              onClick={() => onSelect(frame.timestamp)}
              type="button"
            >
              {frame.thumbnailUrl ? (
                <img
                  className={`mb-[3px] h-[59px] w-full rounded-md border-2 object-cover ${selected ? 'border-[#1779fa]' : frame.isCandidate ? 'border-[#fb8296]' : 'border-transparent'}`}
                  src={frame.thumbnailUrl}
                  alt=""
                />
              ) : (
                <div className="mb-[3px] h-[59px] rounded-md bg-slate-200" />
              )}
              <span>{frame.label}</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}
