"""The planning + backlog-forgiveness engine.

This module is deliberately framework-free so the logic can be unit tested in
isolation. It implements four things described in the PRD:

1. Spaced repetition (an SM-2 variant) for memory-heavy concepts.
2. Student status detection (on_track / mild / major backlog / re-entry).
3. Visible workload caps per status, so the home screen never overwhelms.
4. Recovery ranking, which orders overdue work by a blend of exam importance,
   weakness severity, recency and forgetting risk.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

from .config import settings
from .models import Chapter, Concept, Mastery


class Status(str, Enum):
    ON_TRACK = "on_track"
    MILD_BACKLOG = "mild_backlog"
    MAJOR_BACKLOG = "major_backlog"
    RE_ENTRY = "re_entry"


# --------------------------------------------------------------------------- #
#  Spaced repetition (SM-2 variant)                                           #
# --------------------------------------------------------------------------- #
def update_schedule(m: Mastery, quality: int, now: datetime | None = None) -> Mastery:
    """Update a Mastery row after a review.

    `quality` is 0..5 (0 = total blackout, 5 = perfect recall), matching the
    classic SM-2 grading scale. We also fold the result into the 0..1 mastery
    estimate so the dashboards have a smooth signal to display.
    """
    now = now or datetime.utcnow()
    quality = max(0, min(5, quality))

    m.attempts += 1
    if quality < 3:
        m.error_count += 1
        # Failed recall: reset repetitions, review again soon.
        m.repetitions = 0
        m.interval_days = 1.0
    else:
        m.repetitions += 1
        if m.repetitions == 1:
            m.interval_days = 1.0
        elif m.repetitions == 2:
            m.interval_days = 6.0
        else:
            m.interval_days = round(m.interval_days * m.ease, 2)

    # Update ease factor (bounded below at 1.3 as in SM-2).
    m.ease = max(1.3, m.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))

    # Smooth mastery estimate toward the normalised quality.
    target = quality / 5.0
    m.mastery = round(m.mastery + 0.4 * (target - m.mastery), 3)
    m.confidence = m.mastery

    m.last_reviewed_at = now
    m.due_at = now + timedelta(days=m.interval_days)
    return m


def forgetting_risk(m: Mastery, now: datetime | None = None) -> float:
    """Estimate recall-loss risk in [0, 1].

    Combines how overdue the item is with how shaky mastery already is. Newer,
    weaker, more-overdue items score higher and should resurface first.
    """
    now = now or datetime.utcnow()
    if m.due_at is None:
        # Never scheduled -> treat as moderately at risk so it gets surfaced.
        return 0.6 * (1.0 - m.mastery) + 0.2
    overdue_days = max(0.0, (now - m.due_at).total_seconds() / 86400.0)
    horizon = max(1.0, m.interval_days)
    overdue_ratio = min(1.0, overdue_days / horizon)
    return round(0.6 * overdue_ratio + 0.4 * (1.0 - m.mastery), 3)


# --------------------------------------------------------------------------- #
#  Status detection + workload caps                                           #
# --------------------------------------------------------------------------- #
def days_inactive(last_active_at: datetime, now: datetime | None = None) -> int:
    now = now or datetime.utcnow()
    return max(0, (now - last_active_at).days)


def detect_status(
    last_active_at: datetime,
    overdue_count: int,
    now: datetime | None = None,
) -> Status:
    """Classify the student so the UI can pick tone, caps and mode.

    Inactivity is the dominant signal (the PRD's defining concern), with the
    size of the overdue queue used as a secondary nudge.
    """
    now = now or datetime.utcnow()
    inactive = days_inactive(last_active_at, now)

    if inactive >= settings.reentry_threshold_days:
        return Status.RE_ENTRY
    if inactive >= settings.recovery_threshold_days or overdue_count >= 40:
        return Status.MAJOR_BACKLOG
    if overdue_count >= 15:
        return Status.MILD_BACKLOG
    return Status.ON_TRACK


def visible_cap(status: Status) -> int:
    return {
        Status.ON_TRACK: settings.cap_on_track,
        Status.MILD_BACKLOG: settings.cap_mild_backlog,
        Status.MAJOR_BACKLOG: settings.cap_major_backlog,
        Status.RE_ENTRY: settings.cap_reentry,
    }[status]


def allows_new_material(status: Status) -> bool:
    # New concepts only flow when the student is on track or mildly behind.
    return status in (Status.ON_TRACK, Status.MILD_BACKLOG)


# --------------------------------------------------------------------------- #
#  Recovery ranking                                                           #
# --------------------------------------------------------------------------- #
def recovery_score(
    m: Mastery,
    chapter: Chapter,
    concept: Concept,
    now: datetime | None = None,
) -> float:
    """Blend the four PRD signals into a single priority score in ~[0, 1].

    exam importance + weakness severity + recency + forgetting risk.
    """
    now = now or datetime.utcnow()
    importance = chapter.exam_weight
    weakness = 1.0 - m.mastery
    risk = forgetting_risk(m, now)

    # Recency: more-recently-due items are slightly favoured so the comeback
    # rebuilds the most relevant recent material first.
    if m.due_at is not None:
        overdue_days = max(0.0, (now - m.due_at).total_seconds() / 86400.0)
        recency = 1.0 / (1.0 + overdue_days / 7.0)
    else:
        recency = 0.5

    score = 0.35 * importance + 0.30 * weakness + 0.20 * risk + 0.15 * recency
    return round(score, 4)
