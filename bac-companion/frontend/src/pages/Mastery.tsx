import { useState } from 'react'
import { api, type SubjectMastery } from '../api'
import { useLang } from '../i18n'
import { MasteryBar, SectionTitle, Spinner, masteryColor, useAsync } from '../components/ui'

export default function Mastery() {
  const { t, pick } = useLang()
  const overview = useAsync(() => api.mastery(), [])
  const weak = useAsync(() => api.weakPoints(6), [])

  if (overview.loading) return <Spinner />
  if (overview.error || !overview.data) {
    return (
      <div className="py-20 text-center text-ink-500">
        {pick('Erreur de chargement.', 'خطأ في التحميل.')}
      </div>
    )
  }

  return (
    <div className="space-y-7">
      <h1 className="pt-1 text-2xl font-extrabold tracking-tight">{t('navMastery')}</h1>

      {/* Mastery summary */}
      <MasterySummary subjects={overview.data} />

      {/* Weak points */}
      <section>
        <SectionTitle>{t('weakPointsTitle')}</SectionTitle>
        {weak.loading ? (
          <div className="space-y-2.5">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="card h-24 animate-pulse bg-slate-100" />
            ))}
          </div>
        ) : weak.data && weak.data.length > 0 ? (
          <div className="space-y-2.5">
            {weak.data.map((w, i) => {
              const errorPct = Math.min(100, Math.round(w.error_rate * 100))
              return (
                <div key={w.concept_id} className="card overflow-hidden animate-fade-up" style={{ animationDelay: `${i * 40}ms` }}>
                  <div
                    className="h-1"
                    style={{ backgroundColor: masteryColor(w.mastery) }}
                  />
                  <div className="p-3.5">
                    <div className="mb-2 flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="font-bold leading-snug text-ink-900">
                          {pick(w.concept_name_fr, w.concept_name_ar)}
                        </p>
                        <p className="text-xs text-ink-400">
                          {w.subject_name_fr} · {w.chapter_name_fr}
                        </p>
                      </div>
                      <div className="shrink-0 text-right">
                        <span
                          className="text-lg font-extrabold tabular-nums"
                          style={{ color: masteryColor(w.mastery) }}
                        >
                          {Math.round(w.mastery * 100)}%
                        </span>
                      </div>
                    </div>
                    <MasteryBar value={w.mastery} color={masteryColor(w.mastery)} />
                    <div className="mt-2 flex items-center justify-between gap-2">
                      <p className="text-xs text-ink-500">{w.reason_fr}</p>
                      {errorPct > 0 && (
                        <span className="chip shrink-0 bg-rose-50 text-rose-600">
                          {errorPct}% {t('errorRate')}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="card p-6 text-center">
            <p className="text-2xl">🎉</p>
            <p className="mt-2 font-semibold text-ink-700">{t('noWeakPoints')}</p>
          </div>
        )}
      </section>

      {/* By subject */}
      <section>
        <SectionTitle>{t('bySubject')}</SectionTitle>
        <div className="space-y-3">
          {overview.data.map((s) => (
            <SubjectRow key={s.subject_id} subject={s} />
          ))}
        </div>
      </section>
    </div>
  )
}

/** Horizontal scroll row of subject mastery rings. */
function MasterySummary({ subjects }: { subjects: SubjectMastery[] }) {
  const { pick } = useLang()
  return (
    <div className="flex gap-3 overflow-x-auto pb-1 no-scrollbar">
      {subjects.map((s) => {
        const pct = Math.round(s.mastery * 100)
        const color = masteryColor(s.mastery)
        return (
          <div key={s.subject_id} className="card flex shrink-0 flex-col items-center gap-2 px-4 py-3.5 min-w-[96px]">
            <MiniRing value={s.mastery} color={s.color} />
            <p className="text-center text-[11px] font-semibold leading-tight text-ink-700">
              {pick(s.name_fr, s.name_ar).split(' ')[0]}
            </p>
            <span className="text-sm font-extrabold" style={{ color }}>
              {pct}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

function MiniRing({ value, color }: { value: number; color: string }) {
  const size = 44
  const stroke = 5
  const r = (size - stroke) / 2
  const circ = 2 * Math.PI * r
  const clamped = Math.max(0, Math.min(1, value))
  // strokeDashoffset: circ = empty, 0 = full
  const dashOffset = circ * (1 - clamped)
  return (
    <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e2e8f0" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={`${circ}`}
        strokeDashoffset={`${dashOffset}`}
        style={{ transition: 'stroke-dashoffset 0.8s ease-out' }}
      />
    </svg>
  )
}

function SubjectRow({ subject }: { subject: SubjectMastery }) {
  const { t, pick } = useLang()
  const [open, setOpen] = useState(false)
  const pct = Math.round(subject.mastery * 100)

  return (
    <div className="card overflow-hidden">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-3 p-4 text-start">
        {/* Colour swatch */}
        <div
          className="h-10 w-10 shrink-0 rounded-xl"
          style={{ backgroundColor: `${subject.color}20`, border: `2px solid ${subject.color}40` }}
        >
          <div className="flex h-full items-center justify-center">
            <span
              className="text-sm font-extrabold"
              style={{ color: subject.color }}
            >
              {pct}
            </span>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <span className="font-bold">{pick(subject.name_fr, subject.name_ar)}</span>
            <span className="shrink-0 text-xs text-ink-400">
              {subject.chapters.length} {t('bySubject').includes('chapitre') ? '' : pick('chapitres', 'فصول')}
            </span>
          </div>
          <MasteryBar value={subject.mastery} color={subject.color} />
        </div>
        <span
          className={`shrink-0 text-ink-300 transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
        >
          ›
        </span>
      </button>

      {open && (
        <div className="space-y-3 border-t border-slate-100 px-4 py-3 animate-fade-up">
          {subject.chapters.map((c) => (
            <div key={c.chapter_id} className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="mb-1 flex items-center justify-between text-sm">
                  <span className="font-medium text-ink-700">{pick(c.name_fr, c.name_ar)}</span>
                  <span
                    className="ml-2 shrink-0 text-xs font-bold"
                    style={{ color: masteryColor(c.mastery) }}
                  >
                    {Math.round(c.mastery * 100)}%
                  </span>
                </div>
                <MasteryBar value={c.mastery} color={masteryColor(c.mastery)} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
