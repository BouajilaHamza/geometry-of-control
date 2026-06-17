import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useLang } from '../i18n'
import { RingProgress, Spinner, StatusPill, SectionTitle, MasteryBar, masteryColor } from '../components/ui'
import TaskCard from '../components/TaskCard'

export default function Home() {
  const { t, pick } = useLang()
  const nav = useNavigate()

  const status = useAsyncStatus()
  const today = useAsyncToday()
  const weak = useAsyncWeak()

  if (status.loading || today.loading) return <Spinner />
  if (status.error || !status.data || !today.data) return <ErrorState onRetry={status.reload} />

  const s = status.data
  const plan = today.data
  const isRecovery = s.status === 'major_backlog' || s.status === 're_entry'
  const visibleTasks = plan.tasks.slice(0, 4)

  return (
    <div className="space-y-7">
      {/* Greeting */}
      <div className="flex items-center justify-between pt-1 animate-fade-up">
        <div>
          <p className="text-sm text-ink-400">{t('greetingMorning')} 👋</p>
          <h1 className="text-2xl font-extrabold tracking-tight">{pick(s.headline_fr, s.headline_ar)}</h1>
        </div>
        <StatusPill status={s.status} />
      </div>

      {/* Recovery banner (supportive, never punitive) */}
      {isRecovery && (
        <button
          onClick={() => nav('/recovery')}
          className="card w-full overflow-hidden p-0 text-start animate-scale-in"
        >
          <div className="bg-gradient-to-br from-violet-500 to-brand-600 p-5 text-white">
            <p className="text-sm font-medium opacity-90">
              {t('reEntry')} · {s.days_inactive} {t('day')}{s.days_inactive > 1 ? 's' : ''}
            </p>
            <p className="mt-1 text-xl font-extrabold">{t('lightRestart')}</p>
            <p className="mt-1 text-sm opacity-90">
              {pick(
                'On a allégé ton plan. Rien à rattraper en force.',
                'خفّفنا برنامجك. ما فماش علاش تلهث.',
              )}
            </p>
            <span className="mt-3 inline-flex items-center gap-1 rounded-lg bg-white/20 px-3 py-1.5 text-sm font-semibold">
              {t('startRecovery')} →
            </span>
          </div>
        </button>
      )}

      {/* Q1: What to do today */}
      <section className="animate-fade-up">
        <SectionTitle>{t('todayQuestion')}</SectionTitle>
        <div className="card overflow-hidden">
          <div className="flex items-center gap-4 p-5">
            <RingProgress value={plan.tasks.length ? 0 : 1} size={86} stroke={9}>
              <span className="text-xl font-extrabold leading-none">{plan.tasks.length}</span>
              <span className="text-[10px] font-semibold text-ink-400">
                {plan.tasks.length === 1 ? t('task') : t('tasks')}
              </span>
            </RingProgress>
            <div className="flex-1">
              {plan.tasks.length === 0 ? (
                <p className="font-semibold text-ink-700">{t('allCaughtUp')}</p>
              ) : (
                <>
                  <p className="text-sm text-ink-500">{t('todayQuestion')}</p>
                  <p className="text-lg font-bold">
                    ~{plan.est_minutes} {t('minutes')}
                  </p>
                  <p className="mt-0.5 text-xs text-ink-400">
                    {countBy(plan.tasks, 'review')} {t('review').toLowerCase()} ·{' '}
                    {countBy(plan.tasks, 'drill')} {t('drill').toLowerCase()}
                    {countBy(plan.tasks, 'new') > 0 && ` · ${countBy(plan.tasks, 'new')} ${t('new').toLowerCase()}`}
                  </p>
                </>
              )}
            </div>
          </div>
          {plan.tasks.length > 0 && (
            <Link
              to="/session"
              className="btn-primary block rounded-none rounded-b-2xl py-4 text-base"
            >
              {t('startSession')} →
            </Link>
          )}
        </div>

        {/* Preview of first tasks */}
        {visibleTasks.length > 0 && (
          <div className="mt-3 space-y-2.5">
            {visibleTasks.map((task, i) => (
              <TaskCard key={`${task.concept_id}-${task.task_type}`} task={task} index={i} />
            ))}
            {plan.tasks.length > visibleTasks.length && (
              <p className="pt-1 text-center text-xs text-ink-400">
                +{plan.tasks.length - visibleTasks.length} {pick('autres dans la séance', 'أخرى في الجلسة')}
              </p>
            )}
          </div>
        )}
      </section>

      {/* Q2: What is weak */}
      <section className="animate-fade-up">
        <SectionTitle hint={pick('voir tout', 'الكل')}>{t('weakQuestion')}</SectionTitle>
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
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <p className="font-semibold leading-tight">{pick(w.concept_name_fr, w.concept_name_ar)}</p>
                    <span className="text-sm font-bold" style={{ color: masteryColor(w.mastery) }}>
                      {Math.round(w.mastery * 100)}%
                    </span>
                  </div>
                  <p className="mb-2 text-xs text-ink-400">{w.subject_name_fr} · {w.chapter_name_fr}</p>
                  <MasteryBar value={w.mastery} color={masteryColor(w.mastery)} />
                  <p className="mt-1.5 text-xs text-ink-500">{w.reason_fr}</p>
                </div>
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
        <div className="grid grid-cols-2 gap-3">
          <div className="card flex flex-col items-center justify-center gap-1 p-5">
            <div className="flex items-center gap-1 text-3xl font-extrabold text-brand-600">
              {s.streak}
              <FlameIcon />
            </div>
            <p className="text-center text-xs font-medium text-ink-400">{t('streak')}</p>
          </div>
          <Link to="/week" className="card flex flex-col items-center justify-center gap-1 p-5">
            <CalendarMini />
            <p className="text-center text-sm font-bold">{t('navWeek')}</p>
            <p className="text-center text-xs text-ink-400">{t('weeklyPlanTitle')}</p>
          </Link>
        </div>
      </section>
    </div>
  )
}

function countBy(tasks: { task_type: string }[], type: string) {
  return tasks.filter((x) => x.task_type === type).length
}

function FlameIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="#f97316" stroke="#f97316" strokeWidth="1">
      <path d="M12 2c1 3-1 4-2 6-1 2 0 4 2 4 1.5 0 2-1 2-2 2 1.5 3 3.5 3 5.5A5.5 5.5 0 0 1 6.5 16c0-3 2.5-5 3.5-8 .7-2 .5-4 2-6Z" />
    </svg>
  )
}
function CalendarMini() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0d9488" strokeWidth="2" strokeLinecap="round">
      <rect x="3" y="4.5" width="18" height="17" rx="3" /><path d="M3 9h18M8 3v3M16 3v3" />
    </svg>
  )
}

function ErrorState({ onRetry }: { onRetry: () => void }) {
  const { pick } = useLang()
  return (
    <div className="flex flex-col items-center gap-4 py-20 text-center">
      <p className="text-ink-500">
        {pick('Impossible de joindre le serveur.', 'تعذّر الاتصال بالخادم.')}
      </p>
      <p className="max-w-xs text-xs text-ink-400">
        {pick(
          "Démarre le backend : uvicorn app.main:app --reload",
          'شغّل الخادم: uvicorn app.main:app --reload',
        )}
      </p>
      <button onClick={onRetry} className="btn-ghost px-5 py-2.5">
        {pick('Réessayer', 'حاول مجددا')}
      </button>
    </div>
  )
}

// Local hook wrappers to keep imports tidy.
import { useAsync } from '../components/ui'
function useAsyncStatus() {
  return useAsync(() => api.status(), [])
}
function useAsyncToday() {
  return useAsync(() => api.today(), [])
}
function useAsyncWeak() {
  return useAsync(() => api.weakPoints(5), [])
}
