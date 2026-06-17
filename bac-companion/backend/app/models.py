"""Database models for the Tunisian Bac Companion.

The curriculum is modelled as a competency graph:

    Subject -> Chapter -> Concept -> SubSkill

Mastery is tracked per Concept for the student, together with the
spaced-repetition scheduling state used by the planning engine.
"""

from datetime import datetime

from sqlmodel import Field, Relationship, SQLModel


# --------------------------------------------------------------------------- #
#  Curriculum graph                                                           #
# --------------------------------------------------------------------------- #
class Subject(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    section: str = Field(index=True)  # e.g. "math", "sciences", "info"
    name_fr: str
    name_ar: str
    color: str = "#0d9488"  # accent colour used by the UI
    order: int = 0

    chapters: list["Chapter"] = Relationship(back_populates="subject")


class Chapter(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    subject_id: int = Field(foreign_key="subject.id", index=True)
    name_fr: str
    name_ar: str
    order: int = 0
    # How frequently this chapter shows up in past bac exams (0..1). Used to
    # prioritise both planning and recovery ranking.
    exam_weight: float = 0.5

    subject: Subject | None = Relationship(back_populates="chapters")
    concepts: list["Concept"] = Relationship(back_populates="chapter")


class Concept(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    chapter_id: int = Field(foreign_key="chapter.id", index=True)
    name_fr: str
    name_ar: str
    order: int = 0
    # "memory" -> spaced repetition, "problem" -> adaptive practice.
    kind: str = "memory"
    # Comma-separated concept ids that should ideally be mastered first.
    prerequisites: str = ""

    chapter: Chapter | None = Relationship(back_populates="concepts")
    subskills: list["SubSkill"] = Relationship(back_populates="concept")


class SubSkill(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    concept_id: int = Field(foreign_key="concept.id", index=True)
    name_fr: str
    name_ar: str

    concept: Concept | None = Relationship(back_populates="subskills")


# --------------------------------------------------------------------------- #
#  Student profile & mastery                                                  #
# --------------------------------------------------------------------------- #
class Student(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    section: str = "math"
    language: str = "fr"  # "fr" or "ar"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    # Length of the current consecutive study-day streak.
    streak: int = 0


class Mastery(SQLModel, table=True):
    """Per-concept mastery + spaced-repetition scheduling state (SM-2 style)."""

    id: int | None = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id", index=True)
    concept_id: int = Field(foreign_key="concept.id", index=True)

    # Mastery estimate in [0, 1].
    mastery: float = 0.0
    # Confidence the student self-reports, in [0, 1].
    confidence: float = 0.0

    # Spaced-repetition state.
    ease: float = 2.5
    interval_days: float = 0.0
    repetitions: int = 0
    last_reviewed_at: datetime | None = None
    due_at: datetime | None = None

    # Behavioural signals feeding weak-point detection.
    avg_seconds: float = 0.0
    error_count: int = 0
    attempts: int = 0


class StudySession(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id", index=True)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    kind: str = "daily"  # "daily" | "recovery" | "mock"
    items_total: int = 0
    items_correct: int = 0
    duration_seconds: int = 0
