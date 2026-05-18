import { Upload } from 'lucide-react'
import { useCallback, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { WhisperModelName } from '@/api'

const ACCEPT =
  '.mp3,.wav,.m4a,.ogg,.flac,.mp4,.mov,.mkv,.webm,.avi,audio/*,video/*'

const MODEL_OPTIONS: { value: WhisperModelName; label: string }[] = [
  { value: 'large-v3', label: 'large-v3' },
  { value: 'large-v3-turbo', label: 'large-v3-turbo' },
  { value: 'medium', label: 'medium' },
  { value: 'small', label: 'small' },
  { value: 'base', label: 'base' },
]

const LANGUAGE_OPTIONS: { value: string; label: string }[] = [
  { value: '', label: 'Авто' },
  { value: 'ru', label: 'Русский' },
  { value: 'en', label: 'English' },
  { value: 'uk', label: 'Українська' },
  { value: 'de', label: 'Deutsch' },
  { value: 'fr', label: 'Français' },
]

export interface UploadFormValues {
  model: WhisperModelName
  language: string | null
  diarize: boolean
}

export interface UploadZoneProps {
  onSubmit: (
    files: File[],
    values: UploadFormValues,
  ) => void | Promise<void>
  submitting?: boolean
}

export function UploadZone({ onSubmit, submitting = false }: UploadZoneProps) {
  const inputId = useId()
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [pending, setPending] = useState<File[]>([])
  const [model, setModel] = useState<WhisperModelName>('large-v3')
  const [language, setLanguage] = useState('')

  const addFiles = useCallback((list: FileList | File[]) => {
    const next = Array.from(list).filter((f) => f.size > 0)
    if (next.length === 0) return
    setPending((prev) => {
      const seen = new Set(prev.map((p) => `${p.name}:${p.size}:${p.lastModified}`))
      const merged = [...prev]
      for (const f of next) {
        const key = `${f.name}:${f.size}:${f.lastModified}`
        if (!seen.has(key)) {
          seen.add(key)
          merged.push(f)
        }
      }
      return merged
    })
  }, [])

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files)
  }

  const handleSubmit = async () => {
    if (pending.length === 0 || submitting) return
    const values: UploadFormValues = {
      model,
      language: language.trim() === '' ? null : language.trim(),
      diarize: false,
    }
    await onSubmit(pending, values)
    setPending([])
    if (fileRef.current) fileRef.current.value = ''
  }

  const removeAt = (index: number) => {
    setPending((prev) => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="space-y-1.5">
          <label htmlFor={`${inputId}-model`} className="text-sm font-medium">
            Модель
          </label>
          <select
            id={`${inputId}-model`}
            className="flex h-9 w-full rounded-lg border border-input bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
            value={model}
            onChange={(e) => setModel(e.target.value as WhisperModelName)}
            disabled={submitting}
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-1.5">
          <label htmlFor={`${inputId}-lang`} className="text-sm font-medium">
            Язык
          </label>
          <select
            id={`${inputId}-lang`}
            className="flex h-9 w-full rounded-lg border border-input bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            disabled={submitting}
          >
            {LANGUAGE_OPTIONS.map((o) => (
              <option key={o.value || 'auto'} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-end pb-1">
          <label className="flex cursor-not-allowed items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              className="size-4 rounded border-input"
              checked={false}
              disabled
            />
            Диаризация (скоро)
          </label>
        </div>
      </div>

      <div
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            fileRef.current?.click()
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={(e) => {
          e.preventDefault()
          if (!e.currentTarget.contains(e.relatedTarget as Node)) {
            setDragOver(false)
          }
        }}
        onDragOver={(e) => {
          e.preventDefault()
          e.dataTransfer.dropEffect = 'copy'
        }}
        onDrop={handleDrop}
        className={[
          'rounded-xl border-2 border-dashed p-10 text-center transition-colors',
          dragOver
            ? 'border-ring bg-muted/40'
            : 'border-border hover:border-muted-foreground/40',
        ].join(' ')}
      >
        <Upload className="mx-auto mb-3 size-8 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">
          Перетащите файлы сюда или выберите с диска
        </p>
        <input
          ref={fileRef}
          id={`${inputId}-file`}
          type="file"
          multiple
          accept={ACCEPT}
          className="sr-only"
          disabled={submitting}
          onChange={(e) => {
            if (e.target.files?.length) addFiles(e.target.files)
          }}
        />
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={submitting}
            onClick={() => fileRef.current?.click()}
          >
            Выбрать файлы
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={submitting || pending.length === 0}
            onClick={() => void handleSubmit()}
          >
            Отправить
          </Button>
        </div>
      </div>

      {pending.length > 0 && (
        <ul className="space-y-2 text-sm">
          {pending.map((f, i) => (
            <li
              key={`${f.name}-${f.size}-${f.lastModified}-${i}`}
              className="flex items-center justify-between gap-2 rounded-lg border border-border px-3 py-2"
            >
              <span className="truncate font-medium">{f.name}</span>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                className="shrink-0"
                disabled={submitting}
                onClick={() => removeAt(i)}
                aria-label={`Убрать ${f.name}`}
              >
                ×
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
