import type { Task, TaskType } from '../api'
import { useLang } from '../i18n'
import { SubjectDot } from './ui'

const typeStyle: Record<TaskType, string> = {
  review: 'bg-brand-50 text-brand-700',
  drill: 'bg-rose-50 text-rose-600',
  new: 'bg-violet-50 text-violet-700',
}

export default function TaskCard({ task, index }: { task: Task; index?: number }) {
  const { t, pick } = useLang()
  const typeLabel = task.task_type === 'review' ? t('review') : task.task_type === 'drill' ? t('drill') : t('new')

  return (
    <div
      className="card flex items-center gap-3 p-3.5 animate-fade-up"
      style={index !== undefined ? { animationDelay: `${index * 40}ms` } : undefined}
    >
      <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl" style={{ backgroundColor: `${task.subject_color}15` }}>
        {task.kind === 'memory' ? <CardsIcon color={task.subject_color} /> : <PenIcon color={task.subject_color} />}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <SubjectDot color={task.subject_color} />
          <span className="truncate text-xs font-semibold text-ink-500">{task.subject_name_fr}</span>
        </div>
        <p className="truncate font-semibold leading-tight">{pick(task.concept_name_fr, task.concept_name_ar)}</p>
        <p className="truncate text-xs text-ink-400">{task.chapter_name_fr}</p>
      </div>
      <div className="flex shrink-0 flex-col items-end gap-1">
        <span className={`chip ${typeStyle[task.task_type]}`}>{typeLabel}</span>
        <span className="text-xs text-ink-400">~{task.est_minutes} {t('minutes')}</span>
      </div>
    </div>
  )
}

function CardsIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="5" width="13" height="16" rx="2" /><path d="M8 5V3h13v16h-2" />
    </svg>
  )
}
function PenIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5Z" />
    </svg>
  )
}
