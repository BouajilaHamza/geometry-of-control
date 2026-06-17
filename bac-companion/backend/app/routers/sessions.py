from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from .. import scheduling
from ..database import get_session
from ..models import Mastery, StudySession
from ..schemas import SessionSubmitIn, SessionSubmitOut
from ..services import get_student

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionSubmitOut)
def submit(payload: SessionSubmitIn, session: Session = Depends(get_session)):
    """Record a completed session: update mastery + schedule, streak, activity."""
    try:
        student = get_student(session)
    except LookupError as e:
        raise HTTPException(404, str(e))

    now = datetime.utcnow()
    correct = 0
    mastery_before = 0.0
    mastery_after = 0.0
    counted = 0

    for result in payload.results:
        m = session.exec(
            select(Mastery).where(
                Mastery.student_id == student.id,
                Mastery.concept_id == result.concept_id,
            )
        ).first()
        if m is None:
            continue
        mastery_before += m.mastery
        if result.seconds:
            # Running average of time-per-item.
            n = max(1, m.attempts)
            m.avg_seconds = round((m.avg_seconds * n + result.seconds) / (n + 1), 1)
        scheduling.update_schedule(m, result.quality, now)
        mastery_after += m.mastery
        counted += 1
        if result.quality >= 3:
            correct += 1
        session.add(m)

    # Update streak: increment if the last active day was yesterday/today.
    if (now - student.last_active_at) > timedelta(days=2):
        student.streak = 1
    elif now.date() != student.last_active_at.date():
        student.streak += 1
    student.last_active_at = now
    session.add(student)

    study = StudySession(
        student_id=student.id,
        completed_at=now,
        kind=payload.kind,
        items_total=len(payload.results),
        items_correct=correct,
        duration_seconds=payload.duration_seconds,
    )
    session.add(study)
    session.commit()
    session.refresh(study)

    delta = round((mastery_after - mastery_before), 3) if counted else 0.0
    if correct == len(payload.results) and payload.results:
        msg = "Parfait ! Séance terminée 🎉"
    elif correct >= len(payload.results) / 2:
        msg = "Bien joué, ça progresse 👍"
    else:
        msg = "Séance terminée — on revoit les points fragiles bientôt 🌱"

    return {
        "session_id": study.id,
        "items_total": study.items_total,
        "items_correct": study.items_correct,
        "streak": student.streak,
        "mastery_delta": delta,
        "message_fr": msg,
    }
