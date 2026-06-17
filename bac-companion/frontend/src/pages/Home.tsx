import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useLang } from '../i18n'
import {
  MasteryBar,
  SectionTitle,
  Spinner,
  masteryColor,
  useAsync,
} from '../components/ui'
import TaskCard from '../components/TaskCard'
import type { Status } from '../api'

export default function Home() {
  const { t, pick } = useLang()
  const nav = useNavigate()

  const status = useAsync(() => api.status(), [])
  const today = useAsync(() => api.today(), [])
  const weak = useAsync(() => api.weakPoints(5), [])

  if (status.loading || today.loading) return <Spinner />
  if (status.error || !status.data || !today.data) {
    return <ErrorState onRetry={status.reload} />
  }

  const s = status.data
  const plan = today.data
  const isRecovery = s.status === 'major_backlog' || s.status === 're_entry'
  const reviewCount = plan.tasks.filter((t) => t.task_type === 'review').length
  const drillCount = plan.tasks.filter((t) => t.task_type === 'drill').length
  const newCount = plan.tasks.filter((t) => t.task_type === 'new').length
  const visibleTasks = plan.tasks.slice(0, 4)

  return (
    <div className="space-y-7">
      {/* Greeting + streak pill */}
      <div className="pt-1 animate-fade-up">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-sm text-ink-400">{t('greetingMorning')} 👋</p>
            <h1 className="text-[1.65rem] font-extrabold leading-tight tracking-tight">
              {pick(s.headline_fr, s.headline_ar)}
            </h1>
          </div>
          <StreakBadge streak={s.streak} />
        </div>
        <StatusBar status={s} />
      </div>

      {/* Recovery banner */}
      {isRecovery && (
        <button
          onClick={() => nav('/recovery')}
          className="card w-full overflow-hidden p-0 text-start animate-scale-in"
        >
          <div className="bg-gradient-to-br from-violet-500 to-brand-600 p-5 text-white">
            <p className="text-sm font-medium opacity-90">
              {t('reEntry')} · {s.days_inactive} {t('day')}{s.days_inactive !== 1 ? 's' : ''}
            </p>
            <p className="mt-1 text-xl font-extrabold">{t('lightRestart')}</p>
            <p className="mt-1 text-sm leading-relaxed opacity-90">
              {pick(
                'On a allégé ton plan. Rien à rattraper en force.',
                'خفّفنا برنامجك. ما فماش علاش تلهث.',
              )}
            </p>
            <span className="mt-3 inline-flex items-center gap-1.5 rounded-xl bg-white/20 px-4 py-2 text-sm font-bold backdrop-blur">
              {t('startRecovery')} →
            </span>
          </div>
        </button>
      )}

      {/* Q1: What to do today */}
      <section className="animate-fade-up">
        <SectionTitle>{t('todayQuestion')}</SectionTitle>

        <div className="card overflow-hidden">
          {/* Session summary row */}
          <div className="flex items-stretch gap-0">
            {/* Left: task count + type breakdown */}
            <div className="flex flex-1 flex-col justify-center gap-1 p-5">
              {plan.tasks.length === 0 ? (
                <p className="font-semibold text-ink-700">{t('allCaughtUp')}</p>
              ) : (
                <>
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-extrabold tabular-nums text-brand-600">
                      {plan.tasks.length}
                    </span>
                    <span className="text-sm font-semibold text-ink-400">
                      {plan.tasks.length === 1 ? t('task') : t('tasks')}
                    </span>
                  </div>
                  <p className="text-base font-bold text-ink-700">
                    ~{plan.est_minutes} {t('minutes')}
                  </p>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {reviewCount > 0 && (
                      <TaskTypePill color="brand" count={reviewCount} label={t('review')} />
                    )}
                    {drillCount > 0 && (
                      <TaskTypePill color="rose" count={drillCount} label={t('drill')} />
                    )}
                    {newCount > 0 && (
                      <TaskTypePill color="violet" count={newCount} label={t('new')} />
                    )}
                  </div>
                </>
              )}
            </div>

            {/* Right: visual donut */}
            {plan.tasks.length > 0 && (
              <div className="flex items-center justify-center border-l border-slate-100 px-5">
                <SessionDonut
                  review={reviewCount}
                  drill={drillCount}
                  newItem={newCount}
                />
              </div>
            )}
          </div>

          {plan.tasks.length > 0 && (
            <Link
              to="/session"
              className="btn-primary flex items-center justify-center rounded-none rounded-b-2xl py-4 text-base"
            >
              {t('startSession')} →
            </Link>
          )}
        </div>

        {/* Task previews */}
        {visibleTasks.length > 0 && (
          <div className="mt-3 space-y-2.5">
            {visibleTasks.map((task, i) => (
              <TaskCard key={`${task.concept_id}-${task.task_type}`} task={task} index={i} />
            ))}
            {plan.tasks.length > visibleTasks.length && (
              <p className="pt-1 text-center text-xs text-ink-400">
                +{plan.tasks.length - visibleTasks.length}{' '}
                {pick('autres dans la séance', 'أخرى في الجلسة')}
              </p>
            )}
          </div>
        )}
      </section>

      {/* Q2: What is weak */}
      <section className="animate-fade-up">
        <SectionTitle>{t('weakQuestion')}</SectionTitle>
        {weak.loading ? (
          <div className="card h-20 animate-pulse bg-slate-100" />
        ) : weak.data && weak.data.length > 0 ? (
          <div className="space-y-2.5">
            {weak.data.slice(0, 3).map((w) => (
              <Link
                to="/mastery"
                key={w.concept_id}
                className="card flex items-center gap-3 p-3.5 transition-transform active:scale-[0.99]"
              >
                <div
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-xl text-white text-xs font-bold"
                  style={{ backgroundColor: w.subject_color }}
                >
                  {Math.round(w.mastery * 100)}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-semibold leading-tight">
                    {pick(w.concept_name_fr, w.concept_name_ar)}
                  </p>
                  <p className="truncate text-xs text-ink-400">
                    {w.subject_name_fr} · {w.chapter_name_fr}
                  </p>
                  <MasteryBar value={w.mastery} color={masteryColor(w.mastery)} />
                  <p className="mt-1 text-xs text-ink-500">{w.reason_fr}</p>
                </div>
                <span className="shrink-0 text-ink-300">›</span>
              </Link>
            ))}
          </div>
        ) : (
          <div className="card p-5 text-center text-sm text-ink-500">{t('noWeakPoints')}</div>
        )}
      </section>

      {/* Q3: On track this week */}
      <section className="animate-fade-up">
        <SectionTitle>{t('trackQuestion')}</SectionTitle>
        <Link to="/week" className="card block p-5">
          <div className="mb-4 flex items-center justify-between">
            <p className="font-bold">{t('weeklyPlanTitle')}</p>
            <span className="text-xs font-semibold text-brand-600">{pick('voir →', 'شاهد →')}</span>
          </div>
          <WeekMiniBar />
        </Link>
      </section>
    </div>
  )
}

// --- Sub-components ------------------------------------------------------- //

function StreakBadge({ streak }: { streak: number }) {
  if (streak === 0) return null
  return (
    <div className="shrink-0 flex flex-col items-center justify-center rounded-2xl bg-orange-50 px-3.5 py-2.5 text-center">
      <span className="text-xl leading-none">🔥</span>
      <span className="mt-0.5 text-lg font-extrabold leading-none text-orange-500">{streak}</span>
    </div>
  )
}

function StatusBar({ status }: { status: Status }) {
  const colorMap = {
    on_track: 'bg-brand-500',
    mild_backlog: 'bg-amber-400',
    major_backlog: 'bg-sky-500',
    re_entry: 'bg-violet-500',
  }
  const widthMap = {
    on_track: 'w-full',
    mild_backlog: 'w-3/4',
    major_backlog: 'w-1/2',
    re_entry: 'w-1/4',
  }
  return (
    <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
      <div
        className={`h-full rounded-full transition-all duration-700 ${colorMap[status.status]} ${widthMap[status.status]}`}
      />
    </div>
  )
}

function TaskTypePill({
  color,
  count,
  label,
}: {
  color: 'brand' | 'rose' | 'violet'
  count: number
  label: string
}) {
  const cls = {
    brand: 'bg-brand-50 text-brand-700',
    rose: 'bg-rose-50 text-rose-600',
    violet: 'bg-violet-50 text-violet-700',
  }[color]
  return (
    <span className={`chip ${cls}`}>
      {count} {label.toLowerCase()}
    </span>
  )
}

/** Circular badge + segmented ring showing task type mix. */
function SessionDonut({
  review,
  drill,
  newItem,
}: {
  review: number
  drill: number
  newItem: number
}) {
  const total = review + drill + newItem || 1
  const size = 76
  const stroke = 7
  const r = (size - stroke) / 2
  const circ = 2 * Math.PI * r

  // Arcs drawn using transform rotation so each starts where the last ended
  const segments = [
    { count: review, color: '#0d9488' },
    { count: drill, color: '#f43f5e' },
    { count: newItem, color: '#7c3aed' },
  ].filter((s) => s.count > 0)

  let cumFrac = 0
  const arcs = segments.map((s) => {
    const frac = s.count / total
    const startAngle = cumFrac * 360 - 90          // degrees from top
    cumFrac += frac
    return { ...s, frac, startAngle }
  })

  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} style={{ position: 'absolute', top: 0, left: 0 }}>
        {/* Track */}
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#f1f5f9" strokeWidth={stroke} />
        {arcs.map((a, i) => (
          <circle
            key={i}
            cx={size/2}
            cy={size/2}
            r={r}
            fill="none"
            stroke={a.color}
            strokeWidth={stroke}
            strokeLinecap="butt"
            strokeDasharray={`${a.frac * circ} ${circ}`}
            strokeDashoffset={circ * 0.25}  /* start from top (-90°) */
            transform={`rotate(${a.startAngle + 90} ${size/2} ${size/2})`}
          />
        ))}
      </svg>
      {/* Centre number */}
      <div className="relative z-10 text-center">
        <span className="block text-lg font-extrabold leading-none text-brand-600">{total}</span>
        <span className="block text-[9px] font-semibold uppercase tracking-wide text-ink-400">tâches</span>
      </div>
    </div>
  )
}

/** Mini 7-bar chart showing this week's daily planned load. */
function WeekMiniBar() {
  const { data, loading } = useAsync(() => api.week(), [])
  const DAY_ABBR = ['L', 'M', 'M', 'J', 'V', 'S', 'D']
  const todayIdx = (new Date().getDay() + 6) % 7

  if (loading || !data) {
    return <div className="h-12 animate-pulse rounded-lg bg-slate-100" />
  }

  const max = Math.max(...data.days.map((d) => d.est_minutes), 1)
  return (
    <div className="flex items-end gap-1.5">
      {data.days.map((d, i) => {
        const h = Math.max(6, (d.est_minutes / max) * 40)
        const isToday = i === todayIdx
        return (
          <div key={i} className="flex flex-1 flex-col items-center gap-1">
            <div
              className="w-full rounded-t-sm transition-all duration-500"
              style={{
                height: h,
                backgroundColor: isToday ? '#0d9488' : d.est_minutes > 0 ? '#ccfbf1' : '#f1f5f9',
              }}
            />
            <span
              className={`text-[10px] font-bold ${isToday ? 'text-brand-600' : 'text-ink-300'}`}
            >
              {DAY_ABBR[i]}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function ErrorState({ onRetry }: { onRetry: () => void }) {
  const { pick } = useLang()
  return (
    <div className="flex flex-col items-center gap-4 py-20 text-center">
      <div className="text-4xl">🔌</div>
      <p className="font-semibold text-ink-700">
        {pick('Impossible de joindre le serveur.', 'تعذّر الاتصال بالخادم.')}
      </p>
      <p className="max-w-xs rounded-lg bg-slate-100 px-4 py-2 font-mono text-xs text-ink-500">
        uvicorn app.main:app --reload
      </p>
      <button onClick={onRetry} className="btn-ghost px-5 py-2.5">
        {pick('Réessayer', 'حاول مجددا')}
      </button>
    </div>
  )
}

