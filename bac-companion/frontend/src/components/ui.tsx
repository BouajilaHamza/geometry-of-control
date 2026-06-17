import { useCallback, useEffect, useState } from 'react'
import type { StatusKind } from '../api'
import { useLang } from '../i18n'

// --- Data fetching hook ---------------------------------------------------- //
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)

  const run = useCallback(() => {
    setLoading(true)
    fn()
      .then((d) => {
        setData(d)
        setError(null)
      })
      .catch((e) => setError(e as Error))
      .finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(run, [run])
  return { data, error, loading, reload: run }
}

// --- Circular progress ring ------------------------------------------------ //
export function RingProgress({
  value,
  size = 120,
  stroke = 10,
  color = '#0d9488',
  children,
}: {
  value: number // 0..1
  size?: number
  stroke?: number
  color?: string
  children?: React.ReactNode
}) {
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const offset = c * (1 - Math.max(0, Math.min(1, value)))
  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e2e8f0" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">{children}</div>
    </div>
  )
}

// --- Mastery bar ----------------------------------------------------------- //
export function MasteryBar({ value, color = '#0d9488' }: { value: number; color?: string }) {
  const pct = Math.round(value * 100)
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  )
}

export function masteryColor(value: number): string {
  if (value >= 0.75) return '#0d9488' // strong — brand teal
  if (value >= 0.5) return '#eab308' // ok — amber
  return '#f43f5e' // fragile — rose (used sparingly, never as "failure")
}

// --- Status pill ----------------------------------------------------------- //
export function StatusPill({ status }: { status: StatusKind }) {
  const { t } = useLang()
  const map = {
    on_track: { label: 'onTrack', cls: 'bg-brand-50 text-brand-700' },
    mild_backlog: { label: 'mildBacklog', cls: 'bg-amber-50 text-amber-700' },
    major_backlog: { label: 'majorBacklog', cls: 'bg-sky-50 text-sky-700' },
    re_entry: { label: 'reEntry', cls: 'bg-violet-50 text-violet-700' },
  } as const
  const { label, cls } = map[status]
  return <span className={`chip ${cls}`}>{t(label)}</span>
}

// --- Misc ------------------------------------------------------------------ //
export function Spinner() {
  const { t } = useLang()
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20 text-ink-400">
      <div className="h-8 w-8 animate-spin rounded-full border-[3px] border-slate-200 border-t-brand-500" />
      <span className="text-sm">{t('loading')}</span>
    </div>
  )
}

export function SubjectDot({ color }: { color: string }) {
  return <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
}

export function SectionTitle({ children, hint }: { children: React.ReactNode; hint?: string }) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <h2 className="text-sm font-bold uppercase tracking-wide text-ink-400">{children}</h2>
      {hint && <span className="text-xs text-ink-400">{hint}</span>}
    </div>
  )
}
