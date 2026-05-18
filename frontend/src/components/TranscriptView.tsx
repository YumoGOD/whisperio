import { useState } from 'react'

import type { Segment } from '@/api'
import { Button, buttonVariants } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { formatTimecode } from '@/lib/formatTime'

export interface TranscriptViewProps {
  transcriptText: string | null
  segments: Segment[] | null
  jobId: string
  downloadsAvailable: boolean
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

export function TranscriptView({
  transcriptText,
  segments,
  jobId,
  downloadsAvailable,
}: TranscriptViewProps) {
  const [selected, setSelected] = useState<number | null>(null)
  const [copyOk, setCopyOk] = useState(false)

  const hasSegments = segments && segments.length > 0
  const text =
    transcriptText?.trim() ??
    (hasSegments ? segments!.map((s) => s.text).join('\n') : '')

  const handleCopy = async () => {
    if (!text) return
    const ok = await copyText(text)
    setCopyOk(ok)
    window.setTimeout(() => setCopyOk(false), 2000)
  }

  const base = `/api/jobs/${encodeURIComponent(jobId)}/download`

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!text}
          onClick={() => void handleCopy()}
        >
          {copyOk ? 'Скопировано' : 'Копировать'}
        </Button>
        <a
          href={`${base}?format=txt`}
          download
          className={cn(
            buttonVariants({ variant: 'outline', size: 'sm' }),
            !downloadsAvailable && 'pointer-events-none opacity-50',
          )}
          aria-disabled={!downloadsAvailable}
          onClick={(e) => {
            if (!downloadsAvailable) e.preventDefault()
          }}
        >
          TXT
        </a>
        <a
          href={`${base}?format=srt`}
          download
          className={cn(
            buttonVariants({ variant: 'outline', size: 'sm' }),
            !downloadsAvailable && 'pointer-events-none opacity-50',
          )}
          aria-disabled={!downloadsAvailable}
          onClick={(e) => {
            if (!downloadsAvailable) e.preventDefault()
          }}
        >
          SRT
        </a>
        <a
          href={`${base}?format=json`}
          download
          className={cn(
            buttonVariants({ variant: 'outline', size: 'sm' }),
            !downloadsAvailable && 'pointer-events-none opacity-50',
          )}
          aria-disabled={!downloadsAvailable}
          onClick={(e) => {
            if (!downloadsAvailable) e.preventDefault()
          }}
        >
          JSON
        </a>
      </div>

      {hasSegments ? (
        <div className="divide-y divide-border rounded-xl border border-border">
          {segments!.map((seg, i) => (
            <div
              key={`${seg.start}-${seg.end}-${i}`}
              className={cn(
                'flex gap-3 px-3 py-2 transition-colors',
                selected === i && 'bg-muted/80',
              )}
            >
              <button
                type="button"
                className={cn(
                  'shrink-0 cursor-pointer font-mono text-xs tabular-nums tracking-tight',
                  'text-muted-foreground underline-offset-2 hover:text-foreground hover:underline',
                )}
                onClick={() => setSelected((prev) => (prev === i ? null : i))}
              >
                {formatTimecode(seg.start)}
              </button>
              <p className="min-w-0 flex-1 text-sm leading-relaxed">{seg.text}</p>
            </div>
          ))}
        </div>
      ) : text ? (
        <pre className="max-h-[min(60vh,32rem)] overflow-auto rounded-xl border border-border bg-muted/30 p-4 text-sm leading-relaxed whitespace-pre-wrap font-sans">
          {text}
        </pre>
      ) : (
        <p className="text-sm text-muted-foreground">Текст транскрипта недоступен.</p>
      )}
    </div>
  )
}
