import { useEffect, useState } from 'react'

import { getHealth, type HealthResponse } from '@/api'

const POLL_INTERVAL_MS = 5000

export interface WorkerStatus {
  health: HealthResponse | null
  error: boolean
}

export function useWorkerStatus(): WorkerStatus {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function tick(): Promise<void> {
      try {
        const data = await getHealth()
        if (cancelled) return
        setHealth(data)
        setError(false)
      } catch {
        if (cancelled) return
        setError(true)
      }
    }

    tick()
    const id = window.setInterval(tick, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  return { health, error }
}
