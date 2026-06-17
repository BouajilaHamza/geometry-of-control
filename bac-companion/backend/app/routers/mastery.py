from fastapi import APIRouter, Depends
from sqlmodel import Session

from ..database import get_session
from ..schemas import SubjectMasteryOut, WeakPointOut
from ..services import mastery_overview, weak_points

router = APIRouter(prefix="/api/mastery", tags=["mastery"])


@router.get("", response_model=list[SubjectMasteryOut])
def overview(session: Session = Depends(get_session)):
    return mastery_overview(session)


@router.get("/weak-points", response_model=list[WeakPointOut])
def weak(limit: int = 5, session: Session = Depends(get_session)):
    return weak_points(session, limit=limit)
