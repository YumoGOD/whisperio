import { AudioLines } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'

import { ThemeToggle } from '@/components/ThemeToggle'
import { WorkerStatusDot } from '@/components/WorkerStatusDot'

const navItems: { to: string; label: string; end?: boolean }[] = [
  { to: '/', label: 'Загрузка', end: true },
  { to: '/history', label: 'История' },
]

function navLinkClass({ isActive }: { isActive: boolean }): string {
  return [
    'text-sm transition-colors',
    isActive
      ? 'text-foreground font-medium'
      : 'text-muted-foreground hover:text-foreground',
  ].join(' ')
}

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto max-w-5xl px-4 h-14 flex items-center justify-between gap-4">
          <NavLink to="/" className="flex items-center gap-2">
            <AudioLines className="size-5 text-foreground" />
            <span className="font-semibold tracking-tight">Whisper</span>
          </NavLink>
          <nav className="flex items-center gap-5">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={navLinkClass}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-3">
            <WorkerStatusDot />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="flex-1">
        <div className="mx-auto max-w-5xl px-4 py-8">
          <Outlet />
        </div>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-5xl px-4 h-10 flex items-center justify-between text-xs text-muted-foreground">
          <span>Whisper Transcription Service</span>
          <span className="font-mono">v0.1</span>
        </div>
      </footer>
    </div>
  )
}
