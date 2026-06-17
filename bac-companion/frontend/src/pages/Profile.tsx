import { useState } from 'react'
import { api } from '../api'
import { useLang } from '../i18n'
import { Spinner, StatusPill, useAsync } from '../components/ui'

export default function Profile() {
  const { t, pick, lang, setLang } = useLang()
  const student = useAsync(() => api.student(), [])
  const status = useAsync(() => api.status(), [])
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  if (student.loading || status.loading) return <Spinner />
  if (!student.data || !status.data)
    return <div className="py-20 text-center text-ink-500">{pick('Erreur.', 'خطأ.')}</div>

  async function simulate(days: number) {
    setBusy(true)
    setNote(null)
    try {
      await api.simulateInactivity(days)
      setNote(pick(`Absence de ${days} jours simulée. Va à l'accueil.`, `تمّت محاكاة غياب ${days} أيام. روح للرئيسية.`))
      status.reload()
    } finally {
      setBusy(false)
    }
  }
  async function reset() {
    setBusy(true)
    setNote(null)
    try {
      await api.reseed()
      setNote(pick('Démo réinitialisée.', 'تمت إعادة الضبط.'))
      status.reload()
      student.reload()
    } finally {
      setBusy(false)
    }
  }

  const s = student.data
  return (
    <div className="space-y-7">
      <h1 className="pt-1 text-2xl font-extrabold tracking-tight">{t('profile')}</h1>

      {/* Identity */}
      <div className="card flex items-center gap-4 p-5">
        <div className="grid h-16 w-16 place-items-center rounded-full bg-brand-100 text-2xl font-extrabold text-brand-700">
          {s.name.charAt(0)}
        </div>
        <div className="flex-1">
          <p className="text-lg font-bold">{s.name}</p>
          <p className="text-sm text-ink-400">
            {t('section')}: {s.section === 'math' ? pick('Mathématiques', 'الرياضيات') : s.section}
          </p>
        </div>
        <StatusPill status={status.data.status} />
      </div>

      {/* Streak + counts */}
      <div className="grid grid-cols-3 gap-3">
        <Mini value={`${status.data.streak}`} label={t('streak')} />
        <Mini value={`${status.data.due_today_count}`} label={pick("aujourd'hui", 'اليوم')} />
        <Mini value={`${status.data.days_inactive}`} label={pick('jours absent', 'أيام غياب')} />
      </div>

      {/* Language */}
      <section>
        <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-ink-400">{t('language')}</h2>
        <div className="grid grid-cols-2 gap-3">
          {(['fr', 'ar'] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className={`card p-4 font-bold transition-all ${
                lang === l ? 'ring-2 ring-brand-500' : 'text-ink-500'
              }`}
            >
              {l === 'fr' ? 'Français' : 'العربية'}
            </button>
          ))}
        </div>
      </section>

      {/* Demo tools */}
      <section>
        <h2 className="mb-1 text-sm font-bold uppercase tracking-wide text-ink-400">{t('demoTools')}</h2>
        <p className="mb-3 text-xs text-ink-400">{t('demoHint')}</p>
        <div className="card space-y-3 p-4">
          <div className="grid grid-cols-3 gap-2">
            {[3, 10, 20].map((d) => (
              <button
                key={d}
                disabled={busy}
                onClick={() => simulate(d)}
                className="btn-ghost flex-col py-3 disabled:opacity-50"
              >
                <span className="text-lg font-extrabold">{d}</span>
                <span className="text-[10px] text-ink-400">{t('daysAway')}</span>
              </button>
            ))}
          </div>
          <button disabled={busy} onClick={reset} className="btn-ghost w-full py-3 disabled:opacity-50">
            {t('reset')}
          </button>
          {note && <p className="text-center text-sm font-medium text-brand-700">{note}</p>}
        </div>
      </section>
    </div>
  )
}

function Mini({ value, label }: { value: string; label: string }) {
  return (
    <div className="card flex flex-col items-center justify-center p-4">
      <span className="text-2xl font-extrabold text-brand-600">{value}</span>
      <span className="text-center text-[11px] text-ink-400">{label}</span>
    </div>
  )
}
