import { ExternalLink } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import type { JobDetail, JobStatus } from '@/api'
import { Button } from '@/components/ui/button'

function formatDateTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: 'short',
      timeStyle: 'medium',
    })
  } catch {
    return iso
  }
}

function statusLabel(status: JobStatus): string {
  const map: Record<JobStatus, string> = {
    queued: 'В очереди',
    preprocessing: 'Подготовка',
    transcribing: 'Распознавание',
    done: 'Готово',
    failed: 'Ошибка',
  }
  return map[status]
}

function statusBadgeClass(status: JobStatus): string {
  const base =
    'inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium'
  const variants: Record<JobStatus, string> = {
    queued:
      'border-border bg-muted text-muted-foreground',
    preprocessing:
      'border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-200',
    transcribing:
      'border-blue-500/30 bg-blue-500/10 text-blue-800 dark:text-blue-200',
    done: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-200',
    failed: 'border-red-500/30 bg-red-500/10 text-red-800 dark:text-red-200',
  }
  return `${base} ${variants[status]}`
}

export interface JobCardProps {
  job: JobDetail
}

export function JobCard({ job }: JobCardProps) {
  const navigate = useNavigate()
  const showProgress =
    job.status === 'preprocessing' || job.status === 'transcribing'

  return (
    <article className="rounded-xl border border-border p-4 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <h2 className="truncate font-medium leading-tight">{job.filename}</h2>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{job.id}</p>
        </div>
        <span className={statusBadgeClass(job.status)}>{statusLabel(job.status)}</span>
      </div>

      {showProgress && (
        <div className="space-y-1">
          <div className="h-2 w-full overflow-hidden rounded bg-muted">
            <div
              className="h-full bg-primary transition-[width] duration-300 ease-out"
              style={{ width: `${Math.min(100, Math.max(0, job.progress * 100))}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            {Math.round(job.progress * 100)}%
          </p>
        </div>
      )}

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>Создано: {formatDateTime(job.created_at)}</span>
        {job.finished_at && (
          <span>Завершено: {formatDateTime(job.finished_at)}</span>
        )}
      </div>

      {job.status === 'failed' && job.error && (
        <p className="rounded-md border border-destructive/30 bg-destructive/5 px-2 py-1.5 text-sm text-destructive">
          {job.error}
        </p>
      )}

      <div className="flex justify-end pt-1">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => navigate(`/jobs/${job.id}`)}
        >
          <ExternalLink className="mr-1.5 size-3.5" />
          Открыть
        </Button>
      </div>
    </article>
  )
}
