from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..database import get_session
from ..schemas import DailyPlanOut, RecoveryPlanOut, WeeklyPlanOut
from ..services import build_daily_plan, build_recovery_plan, build_weekly_plan

router = APIRouter(prefix="/api/plan", tags=["planning"])


@router.get("/today", response_model=DailyPlanOut)
def today(session: Session = Depends(get_session)):
    try:
        return build_daily_plan(session)
    except LookupError as e:
        raise HTTPException(404, str(e))


@router.get("/week", response_model=WeeklyPlanOut)
def week(session: Session = Depends(get_session)):
    try:
        return build_weekly_plan(session)
    except LookupError as e:
        raise HTTPException(404, str(e))


@router.get("/recovery", response_model=RecoveryPlanOut)
def recovery(session: Session = Depends(get_session)):
    try:
        return build_recovery_plan(session)
    except LookupError as e:
        raise HTTPException(404, str(e))
