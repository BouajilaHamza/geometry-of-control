import { useState } from 'react'
import { api, type SubjectMastery } from '../api'
import { useLang } from '../i18n'
import { MasteryBar, SectionTitle, Spinner, SubjectDot, masteryColor, useAsync } from '../components/ui'

export default function Mastery() {
  const { t, pick } = useLang()
  const overview = useAsync(() => api.mastery(), [])
  const weak = useAsync(() => api.weakPoints(6), [])

  if (overview.loading) return <Spinner />
  if (overview.error || !overview.data)
    return <div className="py-20 text-center text-ink-500">{pick('Erreur.', 'خطأ.')}</div>

  return (
    <div className="space-y-7">
      <h1 className="pt-1 text-2xl font-extrabold tracking-tight">{t('navMastery')}</h1>

      {/* Weak points */}
      <section>
        <SectionTitle>{t('weakPointsTitle')}</SectionTitle>
        {weak.data && weak.data.length > 0 ? (
          <div className="space-y-2.5">
            {weak.data.map((w) => (
              <div key={w.concept_id} className="card p-3.5">
                <div className="flex items-center justify-between">
                  <p className="font-semibold leading-tight">{pick(w.concept_name_fr, w.concept_name_ar)}</p>
                  <span className="text-sm font-bold" style={{ color: masteryColor(w.mastery) }}>
                    {Math.round(w.mastery * 100)}%
                  </span>
                </div>
                <p className="mb-2 text-xs text-ink-400">
                  {w.subject_name_fr} · {w.chapter_name_fr}
                </p>
                <MasteryBar value={w.mastery} color={masteryColor(w.mastery)} />
                <div className="mt-1.5 flex items-center justify-between text-xs text-ink-500">
                  <span>{w.reason_fr}</span>
                  {w.error_rate > 0 && (
                    <span className="text-ink-400">
                      {Math.round(w.error_rate * 100)}% {t('errorRate')}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="card p-5 text-center text-sm text-ink-500">{t('noWeakPoints')}</div>
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

function SubjectRow({ subject }: { subject: SubjectMastery }) {
  const { t, pick } = useLang()
  const [open, setOpen] = useState(false)
  return (
    <div className="card overflow-hidden">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-3 p-4 text-start">
        <SubjectDot color={subject.color} />
        <div className="flex-1">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="font-bold">{pick(subject.name_fr, subject.name_ar)}</span>
            <span className="text-sm font-bold" style={{ color: subject.color }}>
              {Math.round(subject.mastery * 100)}%
            </span>
          </div>
          <MasteryBar value={subject.mastery} color={subject.color} />
        </div>
        <span className={`text-ink-400 transition-transform ${open ? 'rotate-90' : ''}`}>›</span>
      </button>

      {open && (
        <div className="space-y-3 border-t border-slate-100 bg-slate-50/60 px-4 py-3 animate-fade-up">
          {subject.chapters.map((c) => (
            <div key={c.chapter_id}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="font-medium text-ink-700">{pick(c.name_fr, c.name_ar)}</span>
                <span className="text-xs text-ink-400">
                  {c.concept_count} {t('concepts')} · {Math.round(c.mastery * 100)}%
                </span>
              </div>
              <MasteryBar value={c.mastery} color={masteryColor(c.mastery)} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
