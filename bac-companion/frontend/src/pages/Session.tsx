import { useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type SessionResult, type SessionSubmitResponse, type Task } from '../api'
import { useLang } from '../i18n'
import { Spinner, SubjectDot, useAsync } from '../components/ui'

const GRADES = [
  { quality: 1, key: 'again', cls: 'bg-rose-500 hover:bg-rose-600' },
  { quality: 2, key: 'hard', cls: 'bg-amber-500 hover:bg-amber-600' },
  { quality: 4, key: 'good', cls: 'bg-brand-500 hover:bg-brand-600' },
  { quality: 5, key: 'easy', cls: 'bg-emerald-600 hover:bg-emerald-700' },
] as const

export default function Session() {
  const { t, pick } = useLang()
  const nav = useNavigate()
  const { data, loading, error } = useAsync(() => api.today(), [])

  const tasks = useMemo(() => data?.tasks ?? [], [data])
  const [idx, setIdx] = useState(0)
  const [revealed, setRevealed] = useState(false)
  const [results, setResults] = useState<SessionResult[]>([])
  const [done, setDone] = useState<SessionSubmitResponse | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const startRef = useRef(Date.now())
  const cardStartRef = useRef(Date.now())

  if (loading) return <Spinner />
  if (error || !data) return <Centered>{pick('Erreur de chargement.', 'خطأ في التحميل.')}</Centered>
  if (tasks.length === 0) return <Centered>{t('allCaughtUp')}</Centered>

  if (done) return <Summary res={done} onHome={() => nav('/')} />

  const task = tasks[idx]
  const progress = (idx + (revealed ? 0.5 : 0)) / tasks.length

  async function grade(quality: number) {
    const seconds = Math.round((Date.now() - cardStartRef.current) / 1000)
    const next = [...results, { concept_id: task.concept_id, quality, seconds }]
    setResults(next)

    if (idx + 1 < tasks.length) {
      setIdx(idx + 1)
      setRevealed(false)
      cardStartRef.current = Date.now()
    } else {
      setSubmitting(true)
      try {
        const duration = Math.round((Date.now() - startRef.current) / 1000)
        const res = await api.submitSession(next, duration, data!.is_recovery ? 'recovery' : 'daily')
        setDone(res)
      } finally {
        setSubmitting(false)
      }
    }
  }

  return (
    <div className="flex min-h-[80vh] flex-col">
      {/* Progress header */}
      <div className="flex items-center gap-3 pb-5 pt-1">
        <button onClick={() => nav('/')} className="text-ink-400" aria-label="close">
          <CloseIcon />
        </button>
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200">
          <div className="h-full rounded-full bg-brand-500 transition-all duration-300" style={{ width: `${progress * 100}%` }} />
        </div>
        <span className="text-sm font-semibold text-ink-400">
          {idx + 1}/{tasks.length}
        </span>
      </div>

      {/* Flashcard */}
      <div className="flex flex-1 flex-col">
        <FlashCard task={task} revealed={revealed} />

        <div className="mt-auto pt-6">
          {!revealed ? (
            <button
              onClick={() => setRevealed(true)}
              className="btn-primary w-full py-4 text-base"
            >
              {pick('Afficher', 'إظهار')}
            </button>
          ) : (
            <div>
              <p className="mb-3 text-center text-sm font-semibold text-ink-500">{t('howWasIt')}</p>
              <div className="grid grid-cols-4 gap-2">
                {GRADES.map((g) => (
                  <button
                    key={g.quality}
                    disabled={submitting}
                    onClick={() => grade(g.quality)}
                    className={`btn py-3 text-xs font-bold text-white ${g.cls} disabled:opacity-60`}
                  >
                    {t(g.key)}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function FlashCard({ task, revealed }: { task: Task; revealed: boolean }) {
  const { pick, t } = useLang()
  return (
    <div className="card flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center animate-scale-in">
      <div className="flex items-center gap-2">
        <SubjectDot color={task.subject_color} />
        <span className="text-sm font-semibold text-ink-500">{task.subject_name_fr}</span>
      </div>
      <h2 className="text-2xl font-extrabold leading-tight">{pick(task.concept_name_fr, task.concept_name_ar)}</h2>
      <span className="chip bg-slate-100 text-ink-500">
        {task.kind === 'memory' ? t('memory') : t('problem')} · {task.chapter_name_fr}
      </span>

      {revealed && (
        <div className="mt-2 w-full rounded-xl bg-slate-50 p-4 text-sm text-ink-700 animate-fade-up">
          {task.kind === 'memory'
            ? pick(
                'Récite la règle / définition, puis évalue ta réponse.',
                'سمّع القاعدة / التعريف، ثمّ قيّم إجابتك.',
              )
            : pick(
                'Résous un exercice type sur ce concept, puis évalue-toi.',
                'حلّ تمرين نموذجي على هذا المفهوم، ثمّ قيّم نفسك.',
              )}
        </div>
      )}
    </div>
  )
}

function Summary({ res, onHome }: { res: SessionSubmitResponse; onHome: () => void }) {
  const { t, pick } = useLang()
  return (
    <div className="flex min-h-[80vh] flex-col items-center justify-center gap-6 text-center animate-scale-in">
      <div className="grid h-24 w-24 place-items-center rounded-full bg-brand-50 text-5xl">🎉</div>
      <div>
        <h1 className="text-2xl font-extrabold">{t('sessionDone')}</h1>
        <p className="mt-1 text-ink-500">{res.message_fr}</p>
      </div>
      <div className="grid w-full max-w-xs grid-cols-2 gap-3">
        <Stat big={`${res.items_correct}/${res.items_total}`} label={pick('réussies', 'صحيحة')} />
        <Stat big={`${res.streak} 🔥`} label={t('streak')} />
      </div>
      {res.mastery_delta > 0 && (
        <p className="chip bg-brand-50 text-brand-700">
          {t('masteryUp')} +{Math.round(res.mastery_delta * 100)}%
        </p>
      )}
      <button onClick={onHome} className="btn-primary w-full max-w-xs py-4">
        {t('backHome')}
      </button>
    </div>
  )
}

function Stat({ big, label }: { big: string; label: string }) {
  return (
    <div className="card p-4">
      <p className="text-2xl font-extrabold">{big}</p>
      <p className="text-xs text-ink-400">{label}</p>
    </div>
  )
}

function Centered({ children }: { children: React.ReactNode }) {
  const nav = useNavigate()
  const { t } = useLang()
  return (
    <div className="flex min-h-[70vh] flex-col items-center justify-center gap-5 text-center">
      <p className="text-lg font-semibold text-ink-700">{children}</p>
      <button onClick={() => nav('/')} className="btn-ghost px-5 py-2.5">
        {t('backHome')}
      </button>
    </div>
  )
}

function CloseIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  )
}
