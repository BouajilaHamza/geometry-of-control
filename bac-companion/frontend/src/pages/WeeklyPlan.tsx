import { api } from '../api'
import { useLang } from '../i18n'
import { Spinner, StatusPill, useAsync } from '../components/ui'
import TaskCard from '../components/TaskCard'

export default function WeeklyPlan() {
  const { t, pick } = useLang()
  const { data, loading, error } = useAsync(() => api.week(), [])

  if (loading) return <Spinner />
  if (error || !data) return <div className="py-20 text-center text-ink-500">{pick('Erreur.', 'خطأ.')}</div>

  const todayIdx = (new Date().getDay() + 6) % 7 // Mon=0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between pt-1">
        <h1 className="text-2xl font-extrabold tracking-tight">{t('weeklyPlanTitle')}</h1>
        <StatusPill status={data.status} />
      </div>

      <div className="space-y-5">
        {data.days.map((d, di) => {
          const isToday = di === todayIdx
          return (
            <section key={d.weekday} className="animate-fade-up" style={{ animationDelay: `${di * 40}ms` }}>
              <div className="mb-2.5 flex items-center justify-between">
                <h2 className={`font-bold ${isToday ? 'text-brand-600' : ''}`}>
                  {d.weekday}
                  {isToday && <span className="ms-2 chip bg-brand-50 text-brand-700">{t('navToday')}</span>}
                </h2>
                {d.tasks.length > 0 && (
                  <span className="text-xs text-ink-400">
                    ~{d.est_minutes} {t('minutes')}
                  </span>
                )}
              </div>
              {d.tasks.length === 0 ? (
                <div className="card p-4 text-center text-sm text-ink-400">
                  {pick('Jour léger — repos ou révision libre.', 'يوم خفيف — راحة أو مراجعة حرة.')}
                </div>
              ) : (
                <div className="space-y-2.5">
                  {d.tasks.map((task, i) => (
                    <TaskCard key={`${task.concept_id}-${i}`} task={task} index={i} />
                  ))}
                </div>
              )}
            </section>
          )
        })}
      </div>
    </div>
  )
}
