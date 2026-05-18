/** Длительность файла / сегмента в таймкоде (до часов — mm:ss, иначе h:mm:ss). */
export function formatTimecode(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const frac = seconds - Math.floor(seconds)
  const cs = Math.round(frac * 100)
  const pad = (n: number, w: number) => String(n).padStart(w, '0')
  if (h > 0) {
    return `${h}:${pad(m, 2)}:${pad(s, 2)}.${pad(cs, 2)}`
  }
  return `${m}:${pad(s, 2)}.${pad(cs, 2)}`
}

/** Продолжительность медиа для метаданных. */
export function formatDurationSec(sec: number | null): string {
  if (sec === null || !Number.isFinite(sec)) return '—'
  if (sec < 60) return `${Math.round(sec)} с`
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  if (h > 0) return `${h} ч ${m} мин ${s} с`
  return `${m} мин ${s} с`
}

/** Длительность обработки (между started и finished). */
export function formatElapsedMs(
  startedAt: string | null,
  finishedAt: string | null,
  inProgress: boolean,
): string {
  if (!startedAt) return '—'
  const start = new Date(startedAt).getTime()
  if (Number.isNaN(start)) return '—'
  const end = finishedAt ? new Date(finishedAt).getTime() : Date.now()
  if (Number.isNaN(end)) return '—'
  const ms = Math.max(0, end - start)
  const sec = ms / 1000
  if (inProgress) return `${sec.toFixed(1)} с (ещё идёт)`
  if (sec < 60) return `${sec.toFixed(1)} с`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m} мин ${s} с`
}
