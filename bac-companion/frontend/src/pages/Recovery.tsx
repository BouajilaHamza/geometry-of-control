import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useLang } from '../i18n'
import { Spinner, useAsync } from '../components/ui'
import TaskCard from '../components/TaskCard'

export default function Recovery() {
  const { t, pick } = useLang()
  const nav = useNavigate()
  const { data, loading, error } = useAsync(() => api.recovery(), [])

  if (loading) return <Spinner />
  if (error || !data) return <div className="py-20 text-center text-ink-500">{pick('Erreur.', 'خطأ.')}</div>

  return (
    <div className="space-y-6">
      <button onClick={() => nav('/')} className="flex items-center gap-1 text-sm font-semibold text-ink-400">
        ← {t('backHome')}
      </button>

      {/* Supportive hero — no guilt language, no giant counts */}
      <div className="card overflow-hidden animate-scale-in">
        <div className="bg-gradient-to-br from-violet-500 to-brand-600 p-6 text-white">
          <span className="text-4xl">👋</span>
          <h1 className="mt-2 text-2xl font-extrabold leading-tight">
            {pick(data.headline_fr, data.headline_ar)}
          </h1>
          <p className="mt-2 text-sm leading-relaxed opacity-95">
            {pick(data.message_fr, data.message_ar)}
          </p>
        </div>
        {data.hidden_count > 0 && (
          <div className="flex items-center gap-2 bg-white px-5 py-3 text-sm text-ink-500">
            <span className="grid h-7 w-7 place-items-center rounded-full bg-slate-100 text-xs">✓</span>
            <span>
              <b className="text-ink-700">{data.hidden_count}</b> {pick('éléments', 'عناصر')} {t('hiddenSetAside')}
            </span>
          </div>
        )}
      </div>

      {/* The gentle multi-day restart */}
      <div className="space-y-5">
        {data.plan_days.map((d, di) => (
          <section key={d.weekday} className="animate-fade-up" style={{ animationDelay: `${di * 60}ms` }}>
            <div className="mb-2.5 flex items-center justify-between">
              <h2 className="font-bold">{d.weekday}</h2>
              <span className="chip bg-slate-100 text-ink-500">
                ~{d.est_minutes} {t('minutes')} · {d.tasks.length} {d.tasks.length === 1 ? t('task') : t('tasks')}
              </span>
            </div>
            <div className="space-y-2.5">
              {d.tasks.map((task, i) => (
                <TaskCard key={`${task.concept_id}-${i}`} task={task} index={i} />
              ))}
            </div>
            {di === 0 && (
              <Link to="/session" className="btn-primary mt-3 w-full py-4 text-base">
                {t('startRecovery')} →
              </Link>
            )}
          </section>
        ))}
      </div>
    </div>
  )
}
