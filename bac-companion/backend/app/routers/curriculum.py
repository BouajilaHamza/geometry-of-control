from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..models import Chapter, Concept, SubSkill, Subject
from ..schemas import SubjectOut, SubjectTreeOut

router = APIRouter(prefix="/api/curriculum", tags=["curriculum"])


@router.get("/subjects", response_model=list[SubjectOut])
def list_subjects(session: Session = Depends(get_session)):
    return sorted(session.exec(select(Subject)).all(), key=lambda s: s.order)


@router.get("/subjects/{subject_id}", response_model=SubjectTreeOut)
def subject_tree(subject_id: int, session: Session = Depends(get_session)):
    subject = session.get(Subject, subject_id)
    if subject is None:
        raise HTTPException(404, "Subject not found")

    chapters = sorted(
        [c for c in session.exec(select(Chapter)).all() if c.subject_id == subject_id],
        key=lambda c: c.order,
    )
    all_concepts = session.exec(select(Concept)).all()
    all_subskills = session.exec(select(SubSkill)).all()

    ch_out = []
    for chapter in chapters:
        concepts = sorted(
            [c for c in all_concepts if c.chapter_id == chapter.id], key=lambda c: c.order
        )
        co_out = []
        for concept in concepts:
            subskills = [ss for ss in all_subskills if ss.concept_id == concept.id]
            co_out.append({
                "id": concept.id,
                "name_fr": concept.name_fr,
                "name_ar": concept.name_ar,
                "kind": concept.kind,
                "subskills": [
                    {"id": ss.id, "name_fr": ss.name_fr, "name_ar": ss.name_ar}
                    for ss in subskills
                ],
            })
        ch_out.append({
            "id": chapter.id,
            "name_fr": chapter.name_fr,
            "name_ar": chapter.name_ar,
            "exam_weight": chapter.exam_weight,
            "concepts": co_out,
        })

    return {
        "id": subject.id,
        "section": subject.section,
        "name_fr": subject.name_fr,
        "name_ar": subject.name_ar,
        "color": subject.color,
        "chapters": ch_out,
    }
