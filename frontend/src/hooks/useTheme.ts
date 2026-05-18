import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

const STORAGE_KEY = 'whisper.theme'

function readInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  const saved = window.localStorage.getItem(STORAGE_KEY)
  if (saved === 'light' || saved === 'dark') return saved
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}

export function useTheme(): {
  theme: Theme
  toggle: () => void
  setTheme: (t: Theme) => void
} {
  const [theme, setTheme] = useState<Theme>(readInitialTheme)

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    window.localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  return {
    theme,
    setTheme,
    toggle: () => setTheme((t) => (t === 'light' ? 'dark' : 'light')),
  }
}
