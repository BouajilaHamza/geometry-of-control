// Typed client for the Bac Companion FastAPI backend.

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

// ---- Types (mirror backend/app/schemas.py) ------------------------------- //
export interface Student {
  id: number
  name: string
  section: string
  language: string
  streak: number
  last_active_at: string
}

export type StatusKind = 'on_track' | 'mild_backlog' | 'major_backlog' | 're_entry'

export interface Status {
  status: StatusKind
  days_inactive: number
  overdue_count: number
  due_today_count: number
  visible_cap: number
  allows_new_material: boolean
  headline_fr: string
  headline_ar: string
  streak: number
}

export type TaskType = 'review' | 'drill' | 'new'

export interface Task {
  concept_id: number
  concept_name_fr: string
  concept_name_ar: string
  chapter_name_fr: string
  subject_name_fr: string
  subject_color: string
  kind: 'memory' | 'problem'
  task_type: TaskType
  est_minutes: number
  mastery: number
  priority: number
}

export interface DailyPlan {
  date: string
  status: StatusKind
  is_recovery: boolean
  est_minutes: number
  headline_fr: string
  headline_ar: string
  tasks: Task[]
  hidden_count: number
}

export interface WeekdayPlan {
  weekday: string
  est_minutes: number
  tasks: Task[]
}

export interface WeeklyPlan {
  week_start: string
  status: StatusKind
  days: WeekdayPlan[]
}

export interface RecoveryPlan {
  days_inactive: number
  headline_fr: string
  headline_ar: string
  message_fr: string
  message_ar: string
  plan_days: WeekdayPlan[]
  hidden_count: number
}

export interface ChapterMastery {
  chapter_id: number
  name_fr: string
  name_ar: string
  mastery: number
  concept_count: number
}

export interface SubjectMastery {
  subject_id: number
  name_fr: string
  name_ar: string
  color: string
  mastery: number
  chapters: ChapterMastery[]
}

export interface WeakPoint {
  concept_id: number
  concept_name_fr: string
  concept_name_ar: string
  chapter_name_fr: string
  subject_name_fr: string
  subject_color: string
  mastery: number
  error_rate: number
  reason_fr: string
}

export interface SessionResult {
  concept_id: number
  quality: number
  seconds?: number
}

export interface SessionSubmitResponse {
  session_id: number
  items_total: number
  items_correct: number
  streak: number
  mastery_delta: number
  message_fr: string
}

// ---- Fetch helpers -------------------------------------------------------- //
async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  student: () => get<Student>('/api/student'),
  status: () => get<Status>('/api/student/status'),
  today: () => get<DailyPlan>('/api/plan/today'),
  week: () => get<WeeklyPlan>('/api/plan/week'),
  recovery: () => get<RecoveryPlan>('/api/plan/recovery'),
  mastery: () => get<SubjectMastery[]>('/api/mastery'),
  weakPoints: (limit = 5) => get<WeakPoint[]>(`/api/mastery/weak-points?limit=${limit}`),
  submitSession: (results: SessionResult[], duration: number, kind = 'daily') =>
    post<SessionSubmitResponse>('/api/sessions', { kind, duration_seconds: duration, results }),
  // demo helpers
  simulateInactivity: (days: number) =>
    post<{ ok: boolean; days_inactive: number }>(`/api/dev/simulate-inactivity?days=${days}`),
  reseed: () => post<{ ok: boolean }>('/api/dev/reseed'),
}
