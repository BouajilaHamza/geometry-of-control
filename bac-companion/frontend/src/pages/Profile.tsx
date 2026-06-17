import { useState } from 'react'
import { api } from '../api'
import { useLang } from '../i18n'
import { Spinner, masteryColor, useAsync } from '../components/ui'
import type { StatusKind } from '../api'

const STATUS_CONFIG: Record<StatusKind, { label: string; color: string; bgCls: string }> = {
  on_track: { label: 'Sur la bonne voie', color: '#0d9488', bgCls: 'from-brand-500 to-brand-600' },
  mild_backlog: { label: 'Léger retard', color: '#d97706', bgCls: 'from-amber-400 to-amber-500' },
  major_backlog: { label: 'Mode récupération', color: '#0284c7', bgCls: 'from-sky-500 to-sky-600' },
  re_entry: { label: 'Nouveau départ', color: '#7c3aed', bgCls: 'from-violet-500 to-violet-600' },
}

export default function Profile() {
  const { t, pick, lang, setLang } = useLang()
  const student = useAsync(() => api.student(), [])
  const status = useAsync(() => api.status(), [])
  const mastery = useAsync(() => api.mastery(), [])
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  if (student.loading || status.loading) return <Spinner />
  if (!student.data || !status.data) {
    return (
      <div className="py-20 text-center text-ink-500">
        {pick('Erreur de chargement.', 'خطأ في التحميل.')}
      </div>
    )
  }

  async function simulate(days: number) {
    setBusy(true)
    setNote(null)
    try {
      await api.simulateInactivity(days)
      setNote(
        pick(
          `Absence de ${days} jours simulée. Va à l'accueil.`,
          `تمّت محاكاة غياب ${days} أيام. روح للرئيسية.`,
        ),
      )
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
      await Promise.all([status.reload(), student.reload(), mastery.reload()])
    } finally {
      setBusy(false)
    }
  }

  const s = student.data
  const st = status.data
  const cfg = STATUS_CONFIG[st.status]
  const avgMastery =
    mastery.data && mastery.data.length > 0
      ? mastery.data.reduce((sum, m) => sum + m.mastery, 0) / mastery.data.length
      : null

  return (
    <div className="space-y-6">
      {/* Hero card */}
      <div className={`card overflow-hidden animate-scale-in`}>
        <div className={`bg-gradient-to-br ${cfg.bgCls} p-6 text-white`}>
          <div className="flex items-center gap-4">
            <div className="grid h-16 w-16 place-items-center rounded-2xl bg-white/20 text-2xl font-extrabold backdrop-blur">
              {s.name.charAt(0)}
            </div>
            <div>
              <p className="text-2xl font-extrabold">{s.name}</p>
              <p className="text-sm opacity-80">
                {pick('Section', 'الشعبة')}:{' '}
                {s.section === 'math' ? pick('Mathématiques', 'الرياضيات') : s.section}
              </p>
              <span className="mt-1 inline-block rounded-lg bg-white/20 px-2.5 py-0.5 text-xs font-bold backdrop-blur">
                {cfg.label}
              </span>
            </div>
          </div>

          {/* Overall mastery bar */}
          {avgMastery !== null && (
            <div className="mt-5">
              <div className="mb-1 flex justify-between text-xs font-semibold opacity-80">
                <span>{pick('Maîtrise globale', 'الإتقان الكلي')}</span>
                <span>{Math.round(avgMastery * 100)}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-white/20">
                <div
                  className="h-full rounded-full bg-white transition-all duration-700"
                  style={{ width: `${avgMastery * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 divide-x divide-slate-100">
          <StatCell big={`${st.streak}🔥`} label={pick('jours de suite', 'أيام متتالية')} />
          <StatCell big={String(st.due_today_count)} label={pick("tâches aujourd'hui", 'مهام اليوم')} />
          <StatCell big={String(st.days_inactive)} label={pick('jours absent', 'أيام غياب')} />
        </div>
      </div>

      {/* Subject mastery quick view */}
      {mastery.data && mastery.data.length > 0 && (
        <section>
          <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-ink-400">
            {pick('Maîtrise par matière', 'الإتقان حسب المادة')}
          </h2>
          <div className="card divide-y divide-slate-100">
            {mastery.data.map((m) => (
              <div key={m.subject_id} className="flex items-center gap-3 px-4 py-3">
                <div
                  className="h-3 w-3 shrink-0 rounded-full"
                  style={{ backgroundColor: m.color }}
                />
                <span className="min-w-0 flex-1 truncate text-sm font-medium text-ink-700">
                  {pick(m.name_fr, m.name_ar)}
                </span>
                <div className="flex w-24 items-center gap-2 shrink-0">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${m.mastery * 100}%`,
                        backgroundColor: masteryColor(m.mastery),
                      }}
                    />
                  </div>
                  <span
                    className="w-8 text-right text-xs font-bold tabular-nums"
                    style={{ color: masteryColor(m.mastery) }}
                  >
                    {Math.round(m.mastery * 100)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Language */}
      <section>
        <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-ink-400">{t('language')}</h2>
        <div className="grid grid-cols-2 gap-3">
          {(['fr', 'ar'] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className={`card p-4 font-bold transition-all ${
                lang === l
                  ? 'border-2 border-brand-500 text-brand-700 shadow-lift'
                  : 'text-ink-500'
              }`}
            >
              {l === 'fr' ? '🇫🇷 Français' : '🇹🇳 العربية'}
            </button>
          ))}
        </div>
      </section>

      {/* Demo tools */}
      <section className="pb-2">
        <h2 className="mb-1 text-sm font-bold uppercase tracking-wide text-ink-400">{t('demoTools')}</h2>
        <p className="mb-3 text-xs text-ink-400">{t('demoHint')}</p>
        <div className="card p-4">
          <p className="mb-3 text-xs font-semibold text-ink-500">
            {pick('Simuler une absence de…', 'حاكي غياباً من…')}
          </p>
          <div className="grid grid-cols-3 gap-2">
            {[3, 10, 20].map((d) => (
              <button
                key={d}
                disabled={busy}
                onClick={() => simulate(d)}
                className="flex flex-col items-center rounded-xl border border-slate-200 py-3.5 transition-colors hover:bg-slate-50 disabled:opacity-50"
              >
                <span className="text-2xl font-extrabold text-ink-700">{d}</span>
                <span className="mt-0.5 text-[10px] font-medium text-ink-400">
                  {pick('jours', 'أيام')}
                </span>
              </button>
            ))}
          </div>
          <button
            disabled={busy}
            onClick={reset}
            className="btn-ghost mt-3 w-full py-3 disabled:opacity-50"
          >
            {busy ? '⏳' : '↩'} {t('reset')}
          </button>
          {note && (
            <div className="mt-3 rounded-xl bg-brand-50 p-3 text-center text-sm font-medium text-brand-700">
              {note}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

function StatCell({ big, label }: { big: string; label: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-0.5 py-4">
      <span className="text-xl font-extrabold text-ink-800">{big}</span>
      <span className="text-center text-[10px] font-medium leading-tight text-ink-400">{label}</span>
    </div>
  )
}
