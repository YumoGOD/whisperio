import { useWorkerStatus } from '@/hooks/useWorkerStatus'

export function WorkerStatusDot() {
  const { health, error } = useWorkerStatus()

  const alive = !error && health?.worker_alive === true
  const queueSize = health?.queue_size ?? 0
  const gpu = health?.gpu_available === true

  const title = error
    ? 'Backend недоступен'
    : alive
      ? `Воркер активен • очередь: ${queueSize} • ${gpu ? 'GPU' : 'CPU'}`
      : 'Воркер не отвечает'

  return (
    <div
      className="flex items-center gap-2 text-xs text-muted-foreground"
      title={title}
    >
      <span
        className={[
          'inline-block h-2 w-2 rounded-full',
          alive ? 'bg-emerald-500' : 'bg-red-500',
        ].join(' ')}
        aria-hidden
      />
      <span className="hidden sm:inline">
        {alive
          ? gpu
            ? 'worker • GPU'
            : 'worker • CPU'
          : error
            ? 'offline'
            : 'idle'}
      </span>
    </div>
  )
}
