import type { Task, TaskType } from '../api'
import { useLang } from '../i18n'

const typeConfig: Record<TaskType, { cls: string; label: (t: (k: string) => string) => string }> = {
  review: {
    cls: 'bg-brand-50 text-brand-700 border border-brand-100',
    label: (t) => t('review'),
  },
  drill: {
    cls: 'bg-rose-50 text-rose-600 border border-rose-100',
    label: (t) => t('drill'),
  },
  new: {
    cls: 'bg-violet-50 text-violet-700 border border-violet-100',
    label: (t) => t('new'),
  },
}

export default function TaskCard({ task, index }: { task: Task; index?: number }) {
  const { t, pick } = useLang()
  const { cls, label } = typeConfig[task.task_type]

  return (
    <div
      className="card flex items-stretch gap-0 overflow-hidden animate-fade-up"
      style={index !== undefined ? { animationDelay: `${index * 40}ms` } : undefined}
    >
      {/* Left colour stripe */}
      <div className="w-1 shrink-0" style={{ backgroundColor: task.subject_color }} />

      {/* Icon */}
      <div
        className="flex w-12 shrink-0 items-center justify-center"
        style={{ backgroundColor: `${task.subject_color}10` }}
      >
        {task.kind === 'memory' ? (
          <CardsIcon color={task.subject_color} />
        ) : (
          <PenIcon color={task.subject_color} />
        )}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1 py-3 pl-3 pr-2">
        <p className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: task.subject_color }}>
          {task.subject_name_fr}
        </p>
        <p className="font-bold leading-snug text-ink-900">
          {pick(task.concept_name_fr, task.concept_name_ar)}
        </p>
        <p className="mt-0.5 text-xs text-ink-400">{task.chapter_name_fr}</p>
      </div>

      {/* Right: type + time */}
      <div className="flex shrink-0 flex-col items-end justify-center gap-1.5 py-3 pr-3.5">
        <span className={`chip text-[11px] ${cls}`}>{label(t)}</span>
        <span className="text-xs text-ink-400">~{task.est_minutes} {t('minutes')}</span>
      </div>
    </div>
  )
}

function CardsIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="5" width="13" height="16" rx="2" />
      <path d="M8 5V3h13v16h-2" />
    </svg>
  )
}

function PenIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5Z" />
    </svg>
  )
}
