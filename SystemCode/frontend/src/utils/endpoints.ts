import type {
  AnalysisStatus,
  BackendFrame,
  BackendKeyFrame,
  BackendReport,
  BackendVideo,
} from './dataQueries'

export const MODEL_VERSION = 'e13-r50-joint-r94-r4-region080-v2'
export const TASK_STORAGE_KEY = 'endolensTaskId'

export const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1'
).replace(/\/$/, '')

export type FrameResult = BackendFrame

export function assetUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined
  if (path.startsWith('http')) return path
  return new URL(path, new URL(apiBaseUrl, window.location.origin).origin).href
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function uploadVideo(file: File): Promise<BackendVideo> {
  const form = new FormData()
  form.append('file', file)
  return request<BackendVideo>('/videos', { method: 'POST', body: form })
}

export function startAnalysis(videoId: string, intervalSeconds = 1): Promise<{ task_id: string }> {
  return request<{ task_id: string }>('/analyses', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_id: videoId,
      sampling_mode: 'interval',
      interval_seconds: intervalSeconds,
      model_version: MODEL_VERSION,
    }),
  })
}

export const createAnalysis = (videoId: string) => startAnalysis(videoId, 1)

export const getAnalysis = (taskId: string, signal?: AbortSignal) =>
  request<AnalysisStatus>(`/analyses/${taskId}`, { signal })

export const getAnalysisStatus = getAnalysis

export const cancelAnalysis = (taskId: string) =>
  request<{ task_id: string; status: string }>(`/analyses/${taskId}/cancel`, { method: 'POST' })

export function getFrames(
  taskId: string,
  after: string | null = null,
  candidateOnly = false,
  signal?: AbortSignal,
) {
  const query = new URLSearchParams({ limit: '100', candidate_only: String(candidateOnly) })
  if (after !== null) query.set('after', after)
  return request<{
    items: FrameResult[]
    next_cursor: string | null
    last_cursor: string | null
    has_more: boolean
  }>(`/analyses/${taskId}/frames?${query}`, { signal })
}

export const getAnalysisFrames = (taskId: string) => getFrames(taskId)

export const getKeyframes = (taskId: string) =>
  request<{ keyframes: BackendKeyFrame[] }>(`/analyses/${taskId}/keyframes`)

export const getKeyFrames = getKeyframes

export const getReport = (taskId: string) => request<BackendReport>(`/analyses/${taskId}/report`)

export function getModels() {
  return request<{ items: Array<{ model_version: string; device?: string }> }>('/models')
}

export function saveReview(frame: FrameResult, mask: Blob, notes = '') {
  const body = new FormData()
  body.append('edited_mask', mask, 'mask.png')
  body.append('review_status', 'reviewed')
  body.append('notes', notes)
  if (frame.revision_id) body.append('base_revision_id', frame.revision_id)
  return request(`/frames/${frame.frame_id}/review`, { method: 'PUT', body })
}

export function exportCsv(taskId: string) {
  return request<{ download_url: string }>(`/analyses/${taskId}/exports`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format: 'csv', include_reviewed_masks: false }),
  })
}

export async function followAnalysis(
  taskId: string,
  onUpdate: (status: AnalysisStatus, frames: FrameResult[]) => void,
  signal: AbortSignal,
) {
  let cursor: string | null = null
  const saved = new Map<string, FrameResult>()
  try {
    while (!signal.aborted) {
      const status = await getAnalysis(taskId, signal)
      while (!signal.aborted) {
        const page = await getFrames(taskId, cursor, false, signal)
        page.items.forEach((frame) => saved.set(frame.frame_id, frame))
        if (page.last_cursor !== null) cursor = page.last_cursor
        if (!page.has_more) break
      }
      if (signal.aborted) return
      onUpdate(
        status,
        [...saved.values()].sort((a, b) => a.frame_index - b.frame_index),
      )
      if (['completed', 'failed', 'cancelled'].includes(status.status)) return
      await new Promise<void>((resolve) => {
        const done = () => {
          window.clearTimeout(timer)
          signal.removeEventListener('abort', done)
          resolve()
        }
        const timer = window.setTimeout(done, 1500)
        signal.addEventListener('abort', done, { once: true })
      })
    }
  } catch (error) {
    if (!signal.aborted) throw error
  }
}
