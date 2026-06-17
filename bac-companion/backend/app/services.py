"""Business logic shared by the routers.

Keeps the routers thin: they translate HTTP <-> these functions, which own the
planning, status and recovery rules built on top of `scheduling.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import Session, select

from . import scheduling
from .models import Chapter, Concept, Mastery, Student, Subject
from .scheduling import Status

WEEKDAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

EST_MINUTES = {"review": 2, "drill": 4, "new": 6}


def get_student(session: Session) -> Student:
    student = session.exec(select(Student)).first()
    if student is None:
        raise LookupError("No student seeded. Call POST /api/dev/reseed.")
    return student


def _concept_context(session: Session) -> dict[int, tuple[Concept, Chapter, Subject]]:
    """Map concept_id -> (concept, chapter, subject) for cheap lookups."""
    concepts = session.exec(select(Concept)).all()
    chapters = {c.id: c for c in session.exec(select(Chapter)).all()}
    subjects = {s.id: s for s in session.exec(select(Subject)).all()}
    ctx: dict[int, tuple[Concept, Chapter, Subject]] = {}
    for concept in concepts:
        chapter = chapters[concept.chapter_id]
        subject = subjects[chapter.subject_id]
        ctx[concept.id] = (concept, chapter, subject)
    return ctx


def _masteries(session: Session, student: Student) -> list[Mastery]:
    return session.exec(select(Mastery).where(Mastery.student_id == student.id)).all()


def _counts(masteries: list[Mastery], now: datetime) -> tuple[int, int]:
    """Return (overdue_count, due_today_count)."""
    overdue = 0
    due_today = 0
    for m in masteries:
        if m.due_at is None:
            continue
        if m.due_at < now - timedelta(hours=12):
            overdue += 1
        elif m.due_at <= now + timedelta(hours=12):
            due_today += 1
    return overdue, due_today


# --------------------------------------------------------------------------- #
#  Status                                                                     #
# --------------------------------------------------------------------------- #
HEADLINES = {
    Status.ON_TRACK: ("Tu es sur la bonne voie 💪", "أنت على الطريق الصحيح 💪"),
    Status.MILD_BACKLOG: ("Petit retard, rien de grave 🙂", "تأخّر بسيط، لا داعي للقلق 🙂"),
    Status.MAJOR_BACKLOG: ("On reprend en douceur cette semaine 🌱", "نعاودوا بالشوية هذا الأسبوع 🌱"),
    Status.RE_ENTRY: ("Content de te revoir — on repart léger ✨", "فرحانين بيك — نبداو من جديد بخفّة ✨"),
}


def build_status(session: Session, now: datetime | None = None) -> dict:
    now = now or datetime.utcnow()
    student = get_student(session)
    masteries = _masteries(session, student)
    overdue, due_today = _counts(masteries, now)
    status = scheduling.detect_status(student.last_active_at, overdue, now)
    headline_fr, headline_ar = HEADLINES[status]
    return {
        "status": status.value,
        "days_inactive": scheduling.days_inactive(student.last_active_at, now),
        "overdue_count": overdue,
        "due_today_count": due_today,
        "visible_cap": scheduling.visible_cap(status),
        "allows_new_material": scheduling.allows_new_material(status),
        "headline_fr": headline_fr,
        "headline_ar": headline_ar,
        "streak": student.streak,
    }


# --------------------------------------------------------------------------- #
#  Task assembly                                                              #
# --------------------------------------------------------------------------- #
def _task(m: Mastery, ctx, task_type: str, priority: float) -> dict:
    concept, chapter, subject = ctx[m.concept_id]
    return {
        "concept_id": concept.id,
        "concept_name_fr": concept.name_fr,
        "concept_name_ar": concept.name_ar,
        "chapter_name_fr": chapter.name_fr,
        "subject_name_fr": subject.name_fr,
        "subject_color": subject.color,
        "kind": concept.kind,
        "task_type": task_type,
        "est_minutes": EST_MINUTES[task_type],
        "mastery": m.mastery,
        "priority": priority,
    }


def _ranked_pool(session: Session, now: datetime):
    """Return (student, status, ctx, ranked_due, weak, fresh).

    ranked_due : overdue/due items sorted by recovery priority (desc)
    weak       : weakest non-due concepts (drill candidates)
    fresh      : concepts never scheduled (new-material candidates)
    """
    student = get_student(session)
    masteries = _masteries(session, student)
    ctx = _concept_context(session)
    overdue, _ = _counts(masteries, now)
    status = scheduling.detect_status(student.last_active_at, overdue, now)

    due, weak, fresh = [], [], []
    for m in masteries:
        concept, chapter, _ = ctx[m.concept_id]
        if m.due_at is None or m.repetitions == 0 and m.last_reviewed_at is None:
            fresh.append(m)
        if m.due_at is not None and m.due_at <= now + timedelta(hours=12):
            score = scheduling.recovery_score(m, chapter, concept, now)
            due.append((score, m))
        elif m.mastery < 0.55:
            weak.append(m)

    due.sort(key=lambda t: t[0], reverse=True)
    weak.sort(key=lambda m: m.mastery)
    fresh.sort(key=lambda m: ctx[m.concept_id][1].exam_weight, reverse=True)
    return student, status, ctx, due, weak, fresh


def build_daily_plan(session: Session, now: datetime | None = None) -> dict:
    now = now or datetime.utcnow()
    student, status, ctx, due, weak, fresh = _ranked_pool(session, now)
    cap = scheduling.visible_cap(status)
    is_recovery = status in (Status.MAJOR_BACKLOG, Status.RE_ENTRY)

    tasks: list[dict] = []
    # Due reviews first (most important for memory + recovery).
    for score, m in due:
        if len(tasks) >= cap:
            break
        tasks.append(_task(m, ctx, "review", score))

    # One or two weak-point drills if there's room.
    drill_budget = min(2, cap - len(tasks))
    for m in weak[:drill_budget]:
        concept, chapter, _ = ctx[m.concept_id]
        tasks.append(_task(m, ctx, "drill", scheduling.recovery_score(m, chapter, concept, now)))

    # Optionally one new concept, only when status allows it.
    if scheduling.allows_new_material(status) and len(tasks) < cap and fresh:
        m = fresh[0]
        tasks.append(_task(m, ctx, "new", 0.4))

    total_due = len(due)
    hidden = max(0, total_due - sum(1 for t in tasks if t["task_type"] == "review"))
    est_minutes = sum(t["est_minutes"] for t in tasks)
    headline_fr, headline_ar = HEADLINES[status]

    return {
        "date": now,
        "status": status.value,
        "is_recovery": is_recovery,
        "est_minutes": est_minutes,
        "headline_fr": headline_fr,
        "headline_ar": headline_ar,
        "tasks": tasks,
        "hidden_count": hidden,
    }


def build_weekly_plan(session: Session, now: datetime | None = None) -> dict:
    now = now or datetime.utcnow()
    student, status, ctx, due, weak, fresh = _ranked_pool(session, now)
    cap = scheduling.visible_cap(status)

    # Spread the ranked due pool + weak drills across the week, capped per day.
    pool: list[dict] = []
    for score, m in due:
        pool.append(_task(m, ctx, "review", score))
    for m in weak:
        concept, chapter, _ = ctx[m.concept_id]
        pool.append(_task(m, ctx, "drill", scheduling.recovery_score(m, chapter, concept, now)))
    if scheduling.allows_new_material(status):
        for m in fresh:
            pool.append(_task(m, ctx, "new", 0.4))

    per_day = max(3, cap)
    days = []
    monday = now - timedelta(days=now.weekday())
    idx = 0
    for d in range(7):
        day_tasks = pool[idx: idx + per_day]
        idx += per_day
        days.append({
            "weekday": WEEKDAYS_FR[d],
            "est_minutes": sum(t["est_minutes"] for t in day_tasks),
            "tasks": day_tasks,
        })
    return {"week_start": monday, "status": status.value, "days": days}


def build_recovery_plan(session: Session, now: datetime | None = None) -> dict:
    """A gentle, capped restart spread over a few days — never a backlog wall."""
    now = now or datetime.utcnow()
    student, status, ctx, due, weak, fresh = _ranked_pool(session, now)
    cap = scheduling.visible_cap(status)
    days_inactive = scheduling.days_inactive(student.last_active_at, now)

    ranked = [_task(m, ctx, "review", score) for score, m in due]
    total = len(ranked)

    # Reintroduce gradually over 5 days, only the highest-priority items, and
    # never more than the (small) cap per day.
    plan_days = []
    idx = 0
    for d in range(5):
        day_tasks = ranked[idx: idx + cap]
        idx += cap
        if not day_tasks:
            break
        plan_days.append({
            "weekday": f"Jour {d + 1}",
            "est_minutes": sum(t["est_minutes"] for t in day_tasks),
            "tasks": day_tasks,
        })

    shown = sum(len(d["tasks"]) for d in plan_days)
    hidden = max(0, total - shown)

    return {
        "days_inactive": days_inactive,
        "headline_fr": "Reprenons cette semaine, en douceur",
        "headline_ar": "نعاودوا هذا الأسبوع، بالشوية",
        "message_fr": (
            "Pas de panique : on a mis de côté ce qui peut attendre. "
            f"Voici un plan léger sur {len(plan_days)} jours pour te remettre dans le rythme."
        ),
        "message_ar": (
            "ما تخمّمش برشا: حطّينا على جنب اللي ينجّم يستنّى. "
            f"هاو برنامج خفيف على {len(plan_days)} أيام باش ترجع للنسق."
        ),
        "plan_days": plan_days,
        "hidden_count": hidden,
    }


# --------------------------------------------------------------------------- #
#  Mastery dashboards & weak points                                          #
# --------------------------------------------------------------------------- #
def mastery_overview(session: Session) -> list[dict]:
    student = get_student(session)
    masteries = {m.concept_id: m for m in _masteries(session, student)}
    subjects = sorted(session.exec(select(Subject)).all(), key=lambda s: s.order)
    out = []
    for subject in subjects:
        chapters = sorted(
            [c for c in session.exec(select(Chapter)).all() if c.subject_id == subject.id],
            key=lambda c: c.order,
        )
        ch_out = []
        subj_scores = []
        for chapter in chapters:
            concepts = [c for c in session.exec(select(Concept)).all() if c.chapter_id == chapter.id]
            scores = [masteries[c.id].mastery for c in concepts if c.id in masteries]
            avg = round(sum(scores) / len(scores), 3) if scores else 0.0
            subj_scores.extend(scores)
            ch_out.append({
                "chapter_id": chapter.id,
                "name_fr": chapter.name_fr,
                "name_ar": chapter.name_ar,
                "mastery": avg,
                "concept_count": len(concepts),
            })
        subj_avg = round(sum(subj_scores) / len(subj_scores), 3) if subj_scores else 0.0
        out.append({
            "subject_id": subject.id,
            "name_fr": subject.name_fr,
            "name_ar": subject.name_ar,
            "color": subject.color,
            "mastery": subj_avg,
            "chapters": ch_out,
        })
    return out


def weak_points(session: Session, limit: int = 5) -> list[dict]:
    student = get_student(session)
    masteries = _masteries(session, student)
    ctx = _concept_context(session)
    scored = []
    for m in masteries:
        concept, chapter, subject = ctx[m.concept_id]
        error_rate = round(m.error_count / m.attempts, 3) if m.attempts else 0.0
        severity = 0.7 * (1.0 - m.mastery) + 0.3 * error_rate
        if m.mastery >= 0.6 and error_rate < 0.3:
            continue
        if error_rate >= 0.4:
            reason = "Erreurs fréquentes sur cette compétence"
        elif m.mastery < 0.35:
            reason = "Maîtrise encore faible"
        else:
            reason = "À consolider avant l'examen"
        scored.append((severity, {
            "concept_id": concept.id,
            "concept_name_fr": concept.name_fr,
            "concept_name_ar": concept.name_ar,
            "chapter_name_fr": chapter.name_fr,
            "subject_name_fr": subject.name_fr,
            "subject_color": subject.color,
            "mastery": m.mastery,
            "error_rate": error_rate,
            "reason_fr": reason,
        }))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:limit]]
