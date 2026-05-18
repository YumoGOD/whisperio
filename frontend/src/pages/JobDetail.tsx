import { ArrowLeft } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { ApiError, getJob, type JobDetail, type JobStatus } from '@/api'
import { TranscriptView } from '@/components/TranscriptView'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { formatDurationSec, formatElapsedMs } from '@/lib/formatTime'

function isTerminal(status: JobStatus): boolean {
  return status === 'done' || status === 'failed'
}

function statusLabelRu(status: JobStatus): string {
  const m: Record<JobStatus, string> = {
    queued: 'В очереди',
    preprocessing: 'Подготовка',
    transcribing: 'Распознавание',
    done: 'Готово',
    failed: 'Ошибка',
  }
  return m[status]
}

export function JobDetail() {
  const { id } = useParams<{ id: string }>()
  const [job, setJob] = useState<JobDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    setJob(null)
    setError(null)

    let cancelled = false
    let intervalId: ReturnType<typeof setInterval> | undefined

    void (async () => {
      try {
        const j = await getJob(id)
        if (cancelled) return
        setJob(j)
        setError(null)
        if (isTerminal(j.status)) return

        intervalId = window.setInterval(() => {
          void (async () => {
            try {
              const u = await getJob(id)
              if (cancelled) return
              setJob(u)
              if (isTerminal(u.status) && intervalId) {
                window.clearInterval(intervalId)
                intervalId = undefined
              }
            } catch {
              /* опрос: не затираем экран ошибкой */
            }
          })()
        }, 2000)
      } catch (e) {
        if (cancelled) return
        const msg =
          e instanceof ApiError && e.status === 404
            ? 'Задача не найдена'
            : e instanceof ApiError
              ? e.message
              : 'Не удалось загрузить задачу'
        setError(msg)
        setJob(null)
      }
    })()

    return () => {
      cancelled = true
      if (intervalId) window.clearInterval(intervalId)
    }
  }, [id])

  if (!id) {
    return (
      <p className="text-sm text-muted-foreground">Некорректный адрес.</p>
    )
  }

  if (error && !job) {
    return (
      <div className="space-y-4">
        <Link
          to="/history"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-4" />
          К истории
        </Link>
        <p className="text-sm text-destructive">{error}</p>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="space-y-4">
        <Link
          to="/history"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-4" />
          К истории
        </Link>
        <p className="text-sm text-muted-foreground">Загрузка…</p>
      </div>
    )
  }

  const inProgress = !isTerminal(job.status)
  const showProgress =
    job.status === 'preprocessing' || job.status === 'transcribing'

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Link
          to="/history"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-4" />
          К истории
        </Link>
        <Link to="/">
          <Button type="button" variant="outline" size="sm">
            Новая загрузка
          </Button>
        </Link>
      </div>

      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">{job.filename}</h1>
        <p className="font-mono text-xs text-muted-foreground break-all">{job.id}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Статус</CardTitle>
          <CardDescription>
            {statusLabelRu(job.status)}
            {showProgress && (
              <span className="ml-2 text-foreground">
                {Math.round(job.progress * 100)}%
              </span>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {showProgress && (
            <div className="h-2 w-full overflow-hidden rounded-md bg-muted">
              <div
                className="h-full bg-primary transition-[width] duration-300"
                style={{
                  width: `${Math.min(100, Math.max(0, job.progress * 100))}%`,
                }}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Метаданные</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-3 sm:grid-cols-2 text-sm">
            <div>
              <dt className="text-muted-foreground">Модель</dt>
              <dd className="font-medium">{job.model}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Длительность</dt>
              <dd className="font-medium">{formatDurationSec(job.duration_sec)}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Язык (запрос)</dt>
              <dd className="font-medium">{job.language ?? 'Авто'}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Язык (определён)</dt>
              <dd className="font-medium">{job.detected_language ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Создано</dt>
              <dd className="font-medium tabular-nums">
                {new Date(job.created_at).toLocaleString()}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Время обработки</dt>
              <dd className="font-medium">
                {formatElapsedMs(job.started_at, job.finished_at, inProgress)}
              </dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      {job.status === 'failed' && job.error && (
        <div
          className="rounded-xl border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive"
          role="alert"
        >
          {job.error}
        </div>
      )}

      {job.status === 'done' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Транскрипт</CardTitle>
            <CardDescription>Сегменты и файлы результата</CardDescription>
          </CardHeader>
          <CardContent>
            <TranscriptView
              transcriptText={job.transcript_text}
              segments={job.segments}
              jobId={job.id}
              downloadsAvailable
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
