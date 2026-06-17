from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..database import get_session
from ..schemas import StatusOut, StudentOut
from ..services import build_status, get_student

router = APIRouter(prefix="/api/student", tags=["student"])


@router.get("", response_model=StudentOut)
def me(session: Session = Depends(get_session)):
    try:
        return get_student(session)
    except LookupError as e:
        raise HTTPException(404, str(e))


@router.get("/status", response_model=StatusOut)
def status(session: Session = Depends(get_session)):
    try:
        return build_status(session)
    except LookupError as e:
        raise HTTPException(404, str(e))
