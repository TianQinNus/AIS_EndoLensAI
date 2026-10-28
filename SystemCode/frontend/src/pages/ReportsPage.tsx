import { useEffect, useState } from 'react'
import { ReportCard } from '../components/ReportCard'
import { TASK_STORAGE_KEY, assetUrl, exportCsv, getReport } from '../utils/endpoints'
import type { BackendReport } from '../utils/dataQueries'

function ReportsPage() {
  const [report, setReport] = useState<BackendReport>()
  const [error, setError] = useState('')

  useEffect(() => {
    const taskId = localStorage.getItem(TASK_STORAGE_KEY)
    if (!taskId) return
    void getReport(taskId)
      .then(setReport)
      .catch((reason: Error) => setError(reason.message))
  }, [])

  const download = async () => {
    const taskId = localStorage.getItem(TASK_STORAGE_KEY)
    if (!taskId) return
    const exported = await exportCsv(taskId)
    const url = assetUrl(exported.download_url)
    if (url) window.open(url, '_blank', 'noopener')
  }

  return (
    <div className="mx-auto max-w-[1100px]">
      <section className="mb-2 rounded-lg bg-gradient-to-r from-[#112a47] to-[#1d5f9b] p-5 text-white">
        <h1 className="mb-1 text-[22px]">Analysis Reports</h1>
        <p className="text-xs text-[#c8e0f9]">
          Counts come from the saved analysis report. They are candidate frames, not distinct polyps.
        </p>
      </section>
      {!report ? (
        <section className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
          {error || 'Complete an analysis on the Recorded Video Analysis page to view its saved report.'}
        </section>
      ) : (
        <>
          <div className="mb-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <ReportCard label="Analyzed frames" value={String(report.summary.analyzed_frames)} />
            <ReportCard label="Candidate frames" value={String(report.summary.candidate_frames)} />
            <ReportCard label="Failed frames" value={String(report.summary.failed_frames)} />
            <ReportCard label="Reviewed frames" value={String(report.summary.reviewed_frames)} />
          </div>
          <section className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600 shadow-sm">
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
        </>
      )}
    </div>
  )
}

export default ReportsPage
