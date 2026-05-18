/**
 * Типизированный клиент к FastAPI backend.
 * Все пути относительные — в dev Vite проксирует /api → http://localhost:8000,
 * на проде frontend и backend должны жить под одним origin.
 */

export type JobStatus =
  | 'queued'
  | 'preprocessing'
  | 'transcribing'
  | 'done'
  | 'failed'

export type WhisperModelName =
  | 'large-v3'
  | 'large-v3-turbo'
  | 'medium'
  | 'small'
  | 'base'

export type DownloadFormat = 'txt' | 'srt' | 'json'

export interface JobCreateResponse {
  id: string
  filename: string
  status: JobStatus
  created_at: string
}

export interface JobSummary {
  id: string
  filename: string
  status: JobStatus
  duration_sec: number | null
  model: string
  detected_language: string | null
  progress: number
  created_at: string
  finished_at: string | null
}

export interface Segment {
  start: number
  end: number
  text: string
}

export interface JobDetail {
  id: string
  filename: string
  status: JobStatus
  progress: number
  duration_sec: number | null
  model: string
  language: string | null
  detected_language: string | null
  transcript_text: string | null
  segments: Segment[] | null
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface HealthResponse {
  status: 'ok'
  worker_alive: boolean
  gpu_available: boolean
  queue_size: number
}

const API_BASE = '/api'

export class ApiError extends Error {
  status: number
  body: unknown
  constructor(message: string, status: number, body: unknown) {
    super(message)
    this.status = status
    this.body = body
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    let body: unknown = null
    try {
      body = await res.json()
    } catch {
      try {
        body = await res.text()
      } catch {
        body = null
      }
    }
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : `HTTP ${res.status}`
    throw new ApiError(detail, res.status, body)
  }
  if (res.status === 204) {
    return undefined as T
  }
  return (await res.json()) as T
}

export interface CreateJobParams {
  file: File
  language?: string | null
  model?: WhisperModelName
  /** v1: backend accepts but ignores */
  diarize?: boolean
}

export async function createJob({
  file,
  language,
  model,
  diarize,
}: CreateJobParams): Promise<JobCreateResponse> {
  const fd = new FormData()
  fd.append('file', file)
  if (language) fd.append('language', language)
  if (model) fd.append('model', model)
  if (diarize === true) {
    fd.append('diarize', 'true')
  }
  return request<JobCreateResponse>('/jobs', { method: 'POST', body: fd })
}

export interface ListJobsParams {
  limit?: number
  status?: JobStatus
}

export async function listJobs(
  params: ListJobsParams = {},
): Promise<JobSummary[]> {
  const qs = new URLSearchParams()
  if (params.limit) qs.set('limit', String(params.limit))
  if (params.status) qs.set('status', params.status)
  const query = qs.toString()
  const data = await request<{ jobs: JobSummary[] }>(
    `/jobs${query ? `?${query}` : ''}`,
  )
  return data.jobs
}

export function getJob(id: string): Promise<JobDetail> {
  return request<JobDetail>(`/jobs/${encodeURIComponent(id)}`)
}

export function deleteJob(id: string): Promise<void> {
  return request<void>(`/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export function downloadUrl(id: string, format: DownloadFormat): string {
  return `${API_BASE}/jobs/${encodeURIComponent(id)}/download?format=${format}`
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}
