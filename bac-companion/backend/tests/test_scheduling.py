"""Unit tests for the framework-free scheduling engine.

Run with:  .venv/bin/python -m pytest   (after `pip install pytest`)
or simply: .venv/bin/python tests/test_scheduling.py
"""

from datetime import datetime, timedelta

from app.models import Chapter, Concept, Mastery
from app.scheduling import (
    Status,
    allows_new_material,
    detect_status,
    forgetting_risk,
    recovery_score,
    update_schedule,
    visible_cap,
)

NOW = datetime(2026, 1, 15, 12, 0, 0)


def test_successful_review_grows_interval():
    m = Mastery(student_id=1, concept_id=1, repetitions=2, interval_days=6.0, ease=2.5)
    update_schedule(m, quality=5, now=NOW)
    assert m.repetitions == 3
    assert m.interval_days > 6.0
    assert m.due_at > NOW
    assert m.mastery > 0


def test_failed_review_resets_and_reviews_soon():
    m = Mastery(student_id=1, concept_id=1, repetitions=4, interval_days=30.0, ease=2.5, mastery=0.8)
    update_schedule(m, quality=1, now=NOW)
    assert m.repetitions == 0
    assert m.interval_days == 1.0
    assert m.error_count == 1
    assert m.mastery < 0.8  # mastery dropped


def test_status_transitions_with_inactivity():
    assert detect_status(NOW, overdue_count=0, now=NOW) == Status.ON_TRACK
    assert detect_status(NOW, overdue_count=20, now=NOW) == Status.MILD_BACKLOG
    assert detect_status(NOW - timedelta(days=8), 5, now=NOW) == Status.MAJOR_BACKLOG
    assert detect_status(NOW - timedelta(days=15), 5, now=NOW) == Status.RE_ENTRY


def test_caps_shrink_as_backlog_grows():
    assert visible_cap(Status.ON_TRACK) > visible_cap(Status.MILD_BACKLOG)
    assert visible_cap(Status.MILD_BACKLOG) > visible_cap(Status.MAJOR_BACKLOG)
    assert visible_cap(Status.MAJOR_BACKLOG) >= visible_cap(Status.RE_ENTRY)


def test_new_material_paused_in_recovery():
    assert allows_new_material(Status.ON_TRACK)
    assert not allows_new_material(Status.MAJOR_BACKLOG)
    assert not allows_new_material(Status.RE_ENTRY)


def test_forgetting_risk_higher_when_more_overdue():
    weak_old = Mastery(student_id=1, concept_id=1, mastery=0.2, interval_days=6,
                       due_at=NOW - timedelta(days=10))
    strong_fresh = Mastery(student_id=1, concept_id=2, mastery=0.9, interval_days=6,
                           due_at=NOW + timedelta(days=2))
    assert forgetting_risk(weak_old, NOW) > forgetting_risk(strong_fresh, NOW)


def test_recovery_score_prioritises_important_weak_overdue():
    chapter_hi = Chapter(subject_id=1, name_fr="x", name_ar="x", exam_weight=0.95)
    chapter_lo = Chapter(subject_id=1, name_fr="y", name_ar="y", exam_weight=0.4)
    concept = Concept(chapter_id=1, name_fr="c", name_ar="c")
    important_weak = Mastery(student_id=1, concept_id=1, mastery=0.2,
                             interval_days=6, due_at=NOW - timedelta(days=3))
    minor_strong = Mastery(student_id=1, concept_id=2, mastery=0.9,
                           interval_days=6, due_at=NOW - timedelta(days=3))
    assert recovery_score(important_weak, chapter_hi, concept, NOW) > \
        recovery_score(minor_strong, chapter_lo, concept, NOW)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
