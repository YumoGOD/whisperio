import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, createJob, getJob, type JobDetail, type JobStatus } from '@/api'
import { JobCard } from '@/components/JobCard'
import { UploadZone, type UploadFormValues } from '@/components/UploadZone'

function isTerminal(status: JobStatus): boolean {
  return status === 'done' || status === 'failed'
}

export function Home() {
  const [jobs, setJobs] = useState<JobDetail[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [uploadErrors, setUploadErrors] = useState<string[]>([])

  const activeIds = useMemo(
    () => jobs.filter((j) => !isTerminal(j.status)).map((j) => j.id),
    [jobs],
  )
  const activeKey = activeIds.join(',')
  const activeIdsRef = useRef(activeIds)
  activeIdsRef.current = activeIds

  const handleUpload = useCallback(
    async (files: File[], values: UploadFormValues) => {
      setUploadErrors([])
      setSubmitting(true)
      const errors: string[] = []
      const added: JobDetail[] = []
      for (const file of files) {
        try {
          const res = await createJob({
            file,
            language: values.language,
            model: values.model,
            diarize: values.diarize,
          })
          const detail = await getJob(res.id)
          added.push(detail)
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : 'Не удалось создать задачу'
          errors.push(`${file.name}: ${msg}`)
        }
      }
      if (added.length > 0) {
        setJobs((prev) => {
          const ids = new Set(added.map((j) => j.id))
          const rest = prev.filter((j) => !ids.has(j.id))
          return [...added, ...rest]
        })
      }
      if (errors.length > 0) setUploadErrors(errors)
      setSubmitting(false)
    },
    [],
  )

  useEffect(() => {
    if (!activeKey) return

    const poll = async () => {
      const ids = activeIdsRef.current
      if (ids.length === 0) return
      try {
        const updates = await Promise.all(ids.map((id) => getJob(id)))
        setJobs((prev) =>
          prev.map((j) => {
            const u = updates.find((x) => x.id === j.id)
            return u ?? j
          }),
        )
      } catch {
        // сеть / бэкенд — оставляем последнее известное состояние
      }
    }

    void poll()
    const id = window.setInterval(() => void poll(), 2000)
    return () => window.clearInterval(id)
  }, [activeKey])

  return (
    <div className="space-y-10">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Загрузка</h1>
        <p className="text-sm text-muted-foreground">
          Аудио и видео отправляются на транскрибацию. Несколько файлов — по
          очереди в одну партию.
        </p>
      </div>

      {uploadErrors.length > 0 && (
        <div
          role="alert"
          className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive"
        >
          <ul className="list-inside list-disc space-y-1">
            {uploadErrors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      <UploadZone onSubmit={handleUpload} submitting={submitting} />

      {jobs.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-lg font-semibold tracking-tight">Текущая сессия</h2>
          <ul className="space-y-4">
            {jobs.map((job) => (
              <li key={job.id}>
                <JobCard job={job} />
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
