import { api } from '../api'
import { useLang } from '../i18n'
import { Spinner, StatusPill, useAsync } from '../components/ui'
import TaskCard from '../components/TaskCard'

const WEEKDAY_ABBR = ['L', 'M', 'M', 'J', 'V', 'S', 'D']

export default function WeeklyPlan() {
  const { t, pick } = useLang()
  const { data, loading, error } = useAsync(() => api.week(), [])

  if (loading) return <Spinner />
  if (error || !data) {
    return (
      <div className="py-20 text-center text-ink-500">
        {pick('Erreur de chargement.', 'خطأ في التحميل.')}
      </div>
    )
  }

  const todayIdx = (new Date().getDay() + 6) % 7
  const totalTasks = data.days.reduce((n, d) => n + d.tasks.length, 0)
  const totalMin = data.days.reduce((n, d) => n + d.est_minutes, 0)
  const maxMin = Math.max(...data.days.map((d) => d.est_minutes), 1)

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between pt-1">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight">{t('weeklyPlanTitle')}</h1>
          <p className="mt-0.5 text-sm text-ink-400">
            {totalTasks} {t('tasks')} · ~{totalMin} {t('minutes')}
          </p>
        </div>
        <StatusPill status={data.status} />
      </div>

      {/* Mini week overview bar */}
      <WeekOverview days={data.days} todayIdx={todayIdx} maxMin={maxMin} />

      {/* Day sections */}
      <div className="space-y-5">
        {data.days.map((d, di) => {
          const isToday = di === todayIdx
          const isPast = di < todayIdx

          return (
            <section
              key={d.weekday}
              className="animate-fade-up"
              style={{ animationDelay: `${di * 40}ms` }}
            >
              {/* Day header */}
              <div
                className={`mb-2.5 flex items-center gap-2 rounded-xl px-3 py-2 ${
                  isToday
                    ? 'bg-brand-600 text-white'
                    : isPast
                    ? 'bg-slate-100 text-ink-400'
                    : 'bg-slate-50 text-ink-700'
                }`}
              >
                <div
                  className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${
                    isToday
                      ? 'bg-white/20 text-white'
                      : isPast
                      ? 'bg-white text-ink-400'
                      : 'bg-white text-ink-700'
                  }`}
                >
                  {WEEKDAY_ABBR[di]}
                </div>
                <span className="flex-1 font-bold">{d.weekday}</span>
                {isToday && (
                  <span className="rounded-lg bg-white/20 px-2 py-0.5 text-xs font-semibold backdrop-blur">
                    {t('navToday')}
                  </span>
                )}
                {d.tasks.length > 0 && (
                  <span className={`text-xs font-semibold ${isToday ? 'text-white/80' : 'text-ink-400'}`}>
                    ~{d.est_minutes} {t('minutes')}
                  </span>
                )}
              </div>

              {/* Tasks */}
              {d.tasks.length === 0 ? (
                <div className="flex items-center gap-2 rounded-xl border border-dashed border-slate-200 px-4 py-3 text-sm text-ink-400">
                  <span>🌿</span>
                  <span>{pick('Jour léger — repos ou révision libre.', 'يوم خفيف — راحة أو مراجعة حرة.')}</span>
                </div>
              ) : (
                <div className="space-y-2.5">
                  {d.tasks.slice(0, 5).map((task, i) => (
                    <TaskCard key={`${task.concept_id}-${i}`} task={task} index={i} />
                  ))}
                  {d.tasks.length > 5 && (
                    <p className="text-center text-xs text-ink-400">
                      +{d.tasks.length - 5} {pick('autres', 'أخرى')}
                    </p>
                  )}
                </div>
              )}
            </section>
          )
        })}
      </div>
    </div>
  )
}

function WeekOverview({
  days,
  todayIdx,
  maxMin,
}: {
  days: { est_minutes: number }[]
  todayIdx: number
  maxMin: number
}) {
  return (
    <div className="card flex items-end gap-1.5 px-4 py-4">
      {days.map((d, i) => {
        const h = Math.max(4, (d.est_minutes / maxMin) * 40)
        const isToday = i === todayIdx
        const isPast = i < todayIdx
        const ABBR = ['L', 'M', 'M', 'J', 'V', 'S', 'D']
        return (
          <div key={i} className="flex flex-1 flex-col items-center gap-1">
            <div
              className="w-full rounded-t-sm transition-all duration-500"
              style={{
                height: h,
                backgroundColor: isToday
                  ? '#0d9488'
                  : isPast
                  ? '#e2e8f0'
                  : d.est_minutes > 0
                  ? '#99f6e4'
                  : '#f8fafc',
              }}
            />
            <span
              className={`text-[10px] font-bold ${
                isToday ? 'text-brand-600' : isPast ? 'text-ink-300' : 'text-ink-400'
              }`}
            >
              {ABBR[i]}
            </span>
          </div>
        )
      })}
    </div>
  )
}
