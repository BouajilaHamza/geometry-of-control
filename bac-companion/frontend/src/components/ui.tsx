import { useCallback, useEffect, useState } from 'react'
import type { StatusKind } from '../api'
import { useLang } from '../i18n'

// --- Data fetching hook --------------------------------------------------- //
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)

  const run = useCallback(() => {
    setLoading(true)
    fn()
      .then((d) => { setData(d); setError(null) })
      .catch((e) => setError(e as Error))
      .finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(run, [run])
  return { data, error, loading, reload: run }
}

// --- Mastery bar ---------------------------------------------------------- //
export function MasteryBar({ value, color = '#0d9488' }: { value: number; color?: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100)
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
      <div
        className="h-full rounded-full transition-all duration-700 ease-out"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  )
}

export function masteryColor(value: number): string {
  if (value >= 0.72) return '#0d9488'  // strong — brand teal
  if (value >= 0.45) return '#d97706'  // ok — amber
  return '#f43f5e'                     // fragile — rose
}

// --- Status pill ---------------------------------------------------------- //
export function StatusPill({ status }: { status: StatusKind }) {
  const { t } = useLang()
  const map = {
    on_track:      { label: 'onTrack',      cls: 'bg-brand-50 text-brand-700 border border-brand-100' },
    mild_backlog:  { label: 'mildBacklog',  cls: 'bg-amber-50 text-amber-700 border border-amber-100' },
    major_backlog: { label: 'majorBacklog', cls: 'bg-sky-50 text-sky-700 border border-sky-100' },
    re_entry:      { label: 'reEntry',      cls: 'bg-violet-50 text-violet-700 border border-violet-100' },
  } as const
  const { label, cls } = map[status]
  return <span className={`chip shrink-0 ${cls}`}>{t(label)}</span>
}

// --- Spinner -------------------------------------------------------------- //
export function Spinner() {
  const { t } = useLang()
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-24 text-ink-400">
      <div className="h-9 w-9 animate-spin rounded-full border-[3px] border-slate-200 border-t-brand-500" />
      <span className="text-sm">{t('loading')}</span>
    </div>
  )
}

// --- Helpers -------------------------------------------------------------- //
export function SubjectDot({ color }: { color: string }) {
  return <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
}

export function SectionTitle({ children, hint }: { children: React.ReactNode; hint?: string }) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <h2 className="text-[11px] font-bold uppercase tracking-widest text-ink-400">{children}</h2>
      {hint && <span className="text-xs font-semibold text-brand-600">{hint}</span>}
    </div>
  )
}
