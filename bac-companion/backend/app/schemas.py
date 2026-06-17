"""Pydantic response/request schemas for the API layer."""

from datetime import datetime

from pydantic import BaseModel


# --- Curriculum ------------------------------------------------------------ #
class SubSkillOut(BaseModel):
    id: int
    name_fr: str
    name_ar: str


class ConceptOut(BaseModel):
    id: int
    name_fr: str
    name_ar: str
    kind: str
    subskills: list[SubSkillOut] = []


class ChapterOut(BaseModel):
    id: int
    name_fr: str
    name_ar: str
    exam_weight: float
    concepts: list[ConceptOut] = []


class SubjectOut(BaseModel):
    id: int
    section: str
    name_fr: str
    name_ar: str
    color: str


class SubjectTreeOut(SubjectOut):
    chapters: list[ChapterOut] = []


# --- Student & status ------------------------------------------------------ #
class StudentOut(BaseModel):
    id: int
    name: str
    section: str
    language: str
    streak: int
    last_active_at: datetime


class StatusOut(BaseModel):
    status: str
    days_inactive: int
    overdue_count: int
    due_today_count: int
    visible_cap: int
    allows_new_material: bool
    # A supportive, non-punitive headline for the home / recovery screen.
    headline_fr: str
    headline_ar: str
    streak: int


# --- Mastery & weak points ------------------------------------------------- #
class ConceptMasteryOut(BaseModel):
    concept_id: int
    concept_name_fr: str
    concept_name_ar: str
    chapter_name_fr: str
    subject_name_fr: str
    subject_color: str
    mastery: float
    due_at: datetime | None = None


class ChapterMasteryOut(BaseModel):
    chapter_id: int
    name_fr: str
    name_ar: str
    mastery: float
    concept_count: int


class SubjectMasteryOut(BaseModel):
    subject_id: int
    name_fr: str
    name_ar: str
    color: str
    mastery: float
    chapters: list[ChapterMasteryOut] = []


class WeakPointOut(BaseModel):
    concept_id: int
    concept_name_fr: str
    concept_name_ar: str
    chapter_name_fr: str
    subject_name_fr: str
    subject_color: str
    mastery: float
    error_rate: float
    reason_fr: str


# --- Plans & tasks --------------------------------------------------------- #
class TaskOut(BaseModel):
    concept_id: int
    concept_name_fr: str
    concept_name_ar: str
    chapter_name_fr: str
    subject_name_fr: str
    subject_color: str
    kind: str            # "memory" | "problem"
    task_type: str       # "review" | "drill" | "new"
    est_minutes: int
    mastery: float
    priority: float


class DailyPlanOut(BaseModel):
    date: datetime
    status: str
    is_recovery: bool
    est_minutes: int
    headline_fr: str
    headline_ar: str
    tasks: list[TaskOut]
    hidden_count: int    # tasks deliberately kept off the visible queue


class WeekdayPlanOut(BaseModel):
    weekday: str
    est_minutes: int
    tasks: list[TaskOut]


class WeeklyPlanOut(BaseModel):
    week_start: datetime
    status: str
    days: list[WeekdayPlanOut]


class RecoveryPlanOut(BaseModel):
    days_inactive: int
    headline_fr: str
    headline_ar: str
    message_fr: str
    message_ar: str
    plan_days: list[WeekdayPlanOut]
    hidden_count: int


# --- Sessions -------------------------------------------------------------- #
class SessionResultIn(BaseModel):
    concept_id: int
    quality: int          # 0..5 (SM-2 grade)
    seconds: int = 0


class SessionSubmitIn(BaseModel):
    kind: str = "daily"
    results: list[SessionResultIn]
    duration_seconds: int = 0


class SessionSubmitOut(BaseModel):
    session_id: int
    items_total: int
    items_correct: int
    streak: int
    mastery_delta: float
    message_fr: str
