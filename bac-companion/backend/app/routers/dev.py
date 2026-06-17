"""Demo / development helpers.

These endpoints make the retention behaviour easy to demo without waiting real
calendar days — e.g. simulate two weeks of inactivity to trigger recovery mode.
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..database import get_session
from ..seed import seed
from ..services import get_student

router = APIRouter(prefix="/api/dev", tags=["dev"])


@router.post("/reseed")
def reseed(session: Session = Depends(get_session)):
    seed(reset=True)
    return {"ok": True, "message": "Database reseeded."}


@router.post("/simulate-inactivity")
def simulate_inactivity(days: int = 10, session: Session = Depends(get_session)):
    """Backdate the student's last activity to trigger backlog/recovery states."""
    try:
        student = get_student(session)
    except LookupError as e:
        raise HTTPException(404, str(e))
    student.last_active_at = datetime.utcnow() - timedelta(days=days)
    if days >= 3:
        student.streak = 0
    session.add(student)
    session.commit()
    return {"ok": True, "days_inactive": days}
