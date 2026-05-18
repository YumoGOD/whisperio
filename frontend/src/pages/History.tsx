import { Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { ApiError, deleteJob, listJobs, type JobStatus, type JobSummary } from '@/api'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { formatDurationSec } from '@/lib/formatTime'
import { cn } from '@/lib/utils'

type StatusFilter = 'all' | JobStatus

const FILTER_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: 'all', label: 'Все статусы' },
  { value: 'queued', label: 'В очереди' },
  { value: 'preprocessing', label: 'Подготовка' },
  { value: 'transcribing', label: 'Распознавание' },
  { value: 'done', label: 'Готово' },
  { value: 'failed', label: 'Ошибка' },
]

function statusLabel(status: JobStatus): string {
  const m: Record<JobStatus, string> = {
    queued: 'В очереди',
    preprocessing: 'Подготовка',
    transcribing: 'Распознавание',
    done: 'Готово',
    failed: 'Ошибка',
  }
  return m[status]
}

function statusBadgeClass(status: JobStatus): string {
  const base =
    'inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium'
  const variants: Record<JobStatus, string> = {
    queued: 'border-border bg-muted text-muted-foreground',
    preprocessing:
      'border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-200',
    transcribing:
      'border-blue-500/30 bg-blue-500/10 text-blue-800 dark:text-blue-200',
    done: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-200',
    failed: 'border-red-500/30 bg-red-500/10 text-red-800 dark:text-red-200',
  }
  return cn(base, variants[status])
}

function formatDateShort(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: 'short',
      timeStyle: 'short',
    })
  } catch {
    return iso
  }
}

export function History() {
  const [filter, setFilter] = useState<StatusFilter>('all')
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const [toDelete, setToDelete] = useState<JobSummary | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setLoadError(null)

    void (async () => {
      try {
        const rows = await listJobs({
          limit: 100,
          status: filter === 'all' ? undefined : filter,
        })
        if (!cancelled) setJobs(rows)
      } catch (e) {
        if (!cancelled) {
          setLoadError(
            e instanceof ApiError ? e.message : 'Не удалось загрузить список',
          )
          setJobs([])
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [filter])

  const confirmDelete = async () => {
    if (!toDelete) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await deleteJob(toDelete.id)
      setJobs((prev) => prev.filter((j) => j.id !== toDelete.id))
      setToDelete(null)
    } catch (e) {
      setDeleteError(
        e instanceof ApiError ? e.message : 'Не удалось удалить задачу',
      )
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight">История</h1>
          <p className="text-sm text-muted-foreground">
            До {100} последних задач. Фильтр по статусу — без пагинации.
          </p>
        </div>
        <div className="space-y-1.5">
          <label htmlFor="history-status" className="text-sm font-medium">
            Статус
          </label>
          <Select
            value={filter}
            onValueChange={(v) => setFilter(v as StatusFilter)}
          >
            <SelectTrigger id="history-status" className="w-[min(100%,220px)]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {FILTER_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value}>
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {loadError && (
        <p className="text-sm text-destructive" role="alert">
          {loadError}
        </p>
      )}

      {loading ? (
        <p className="text-sm text-muted-foreground">Загрузка…</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Файл</TableHead>
              <TableHead>Статус</TableHead>
              <TableHead>Создано</TableHead>
              <TableHead>Длительность</TableHead>
              <TableHead className="w-[1%] text-right">Действия</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-muted-foreground">
                  Нет задач с выбранным фильтром.
                </TableCell>
              </TableRow>
            ) : (
              jobs.map((j) => (
                <TableRow key={j.id}>
                  <TableCell className="max-w-[min(40vw,280px)]">
                    <div className="truncate font-medium">{j.filename}</div>
                    <div className="truncate font-mono text-xs text-muted-foreground">
                      {j.id}
                    </div>
                  </TableCell>
                  <TableCell>
                    <span className={statusBadgeClass(j.status)}>
                      {statusLabel(j.status)}
                    </span>
                  </TableCell>
                  <TableCell className="tabular-nums text-muted-foreground">
                    {formatDateShort(j.created_at)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {formatDurationSec(j.duration_sec)}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Link
                        to={`/jobs/${j.id}`}
                        className={buttonVariants({
                          variant: 'outline',
                          size: 'sm',
                        })}
                      >
                        Открыть
                      </Link>
                      <Button
                        type="button"
                        variant="destructive"
                        size="icon-sm"
                        title="Удалить"
                        onClick={() => {
                          setDeleteError(null)
                          setToDelete(j)
                        }}
                      >
                        <Trash2 className="size-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      )}

      <Dialog
        open={toDelete !== null}
        onOpenChange={(open) => {
          if (!open) {
            setToDelete(null)
            setDeleteError(null)
          }
        }}
      >
        <DialogContent showCloseButton>
          <DialogHeader>
            <DialogTitle>Удалить задачу?</DialogTitle>
            <DialogDescription>
              {toDelete
                ? `Файл «${toDelete.filename}» и результаты будут удалены безвозвратно.`
                : null}
            </DialogDescription>
          </DialogHeader>
          {deleteError && (
            <p className="text-sm text-destructive">{deleteError}</p>
          )}
          <DialogFooter className="flex-row justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => setToDelete(null)}
              disabled={deleting}
            >
              Отмена
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => void confirmDelete()}
              disabled={deleting}
            >
              {deleting ? 'Удаление…' : 'Удалить'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
