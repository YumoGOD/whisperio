import { Link } from 'react-router-dom'

export function NotFound() {
  return (
    <div className="flex flex-col items-start gap-4">
      <h1 className="text-2xl font-semibold tracking-tight">404</h1>
      <p className="text-sm text-muted-foreground">Страница не найдена.</p>
      <Link
        to="/"
        className="text-sm underline-offset-4 hover:underline text-foreground"
      >
        На главную
      </Link>
    </div>
  )
}
