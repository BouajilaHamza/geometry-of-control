"""Seed a realistic slice of the Tunisian bac (Mathématiques section).

Kept intentionally compact for the MVP: a handful of subjects, each with a few
chapters and concepts, plus a demo student whose mastery + scheduling state is
spread across past/future due dates so the planning and recovery views have
something interesting to show.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from sqlmodel import Session, select

from .database import engine
from .models import Chapter, Concept, Mastery, Student, SubSkill, Subject

# (section, name_fr, name_ar, color, chapters)
# Each chapter: (name_fr, name_ar, exam_weight, [concepts])
# Each concept: (name_fr, name_ar, kind, [subskills])
CURRICULUM = [
    (
        "math", "Mathématiques", "الرياضيات", "#2563eb",
        [
            ("Limites et continuité", "النهايات والاتصال", 0.9, [
                ("Limites de fonctions", "نهايات الدوال", "problem", ["Formes indéterminées", "Limites en l'infini"]),
                ("Continuité", "الاتصال", "memory", ["Théorème des valeurs intermédiaires"]),
            ]),
            ("Dérivabilité", "الاشتقاق", 0.95, [
                ("Nombre dérivé", "العدد المشتق", "memory", ["Tangente", "Approximation affine"]),
                ("Étude de fonctions", "دراسة الدوال", "problem", ["Variations", "Points d'inflexion"]),
            ]),
            ("Suites numériques", "المتتاليات العددية", 0.85, [
                ("Suites arithmétiques et géométriques", "المتتاليات الحسابية والهندسية", "memory", ["Terme général", "Somme"]),
                ("Convergence", "التقارب", "problem", ["Suites récurrentes", "Limites de suites"]),
            ]),
            ("Nombres complexes", "الأعداد المركبة", 0.8, [
                ("Forme algébrique", "الشكل الجبري", "memory", ["Conjugué", "Module"]),
                ("Forme trigonométrique", "الشكل المثلثي", "problem", ["Argument", "Formule de Moivre"]),
            ]),
            ("Intégrales", "التكامل", 0.9, [
                ("Primitives", "الدوال الأصلية", "memory", ["Tableau des primitives"]),
                ("Intégration par parties", "التكامل بالتجزئة", "problem", ["Calcul d'aires"]),
            ]),
        ],
    ),
    (
        "math", "Sciences Physiques", "العلوم الفيزيائية", "#7c3aed",
        [
            ("Dipôle RC", "ثنائي القطب RC", 0.85, [
                ("Charge du condensateur", "شحن المكثف", "problem", ["Constante de temps", "Équation différentielle"]),
                ("Énergie emmagasinée", "الطاقة المخزنة", "memory", []),
            ]),
            ("Oscillations électriques", "الذبذبات الكهربائية", 0.8, [
                ("Circuit RLC", "الدارة RLC", "problem", ["Régime pseudo-périodique"]),
                ("Résonance", "الرنين", "memory", ["Fréquence propre"]),
            ]),
            ("Mécanique — lois de Newton", "الميكانيك — قوانين نيوتن", 0.9, [
                ("Mouvement dans un champ", "الحركة في حقل", "problem", ["Champ de pesanteur", "Champ électrique"]),
                ("Quantité de mouvement", "كمية الحركة", "memory", []),
            ]),
            ("Chimie — cinétique", "الكيمياء — الحركية", 0.75, [
                ("Vitesse de réaction", "سرعة التفاعل", "problem", ["Temps de demi-réaction"]),
                ("Catalyse", "الحفز", "memory", []),
            ]),
        ],
    ),
    (
        "math", "Informatique", "الإعلامية", "#0d9488",
        [
            ("Algorithmique", "الخوارزميات", 0.8, [
                ("Structures itératives", "الهياكل التكرارية", "problem", ["Boucles imbriquées"]),
                ("Récursivité", "الاستدعاء الذاتي", "problem", ["Cas de base"]),
            ]),
            ("Tableaux et tris", "الجداول والترتيب", 0.75, [
                ("Tri par sélection", "الترتيب بالانتقاء", "memory", []),
                ("Recherche dichotomique", "البحث الثنائي", "problem", []),
            ]),
        ],
    ),
    (
        "math", "Anglais", "الإنڨليزية", "#ea580c",
        [
            ("Grammar", "القواعد", 0.6, [
                ("Tenses", "الأزمنة", "memory", ["Present perfect", "Past simple"]),
                ("Conditionals", "الشرطية", "memory", ["Type 2", "Type 3"]),
            ]),
            ("Vocabulary", "المفردات", 0.55, [
                ("Phrasal verbs", "الأفعال المركبة", "memory", []),
            ]),
        ],
    ),
    (
        "math", "Philosophie", "الفلسفة", "#db2777",
        [
            ("La conscience", "الوعي", 0.7, [
                ("Conscience et inconscient", "الوعي واللاوعي", "memory", ["Freud"]),
            ]),
            ("La vérité", "الحقيقة", 0.7, [
                ("Vérité et certitude", "الحقيقة واليقين", "memory", ["Descartes"]),
            ]),
        ],
    ),
]


def seed(reset: bool = False) -> None:
    with Session(engine) as session:
        existing = session.exec(select(Subject)).first()
        if existing and not reset:
            return
        if reset:
            for model in (Mastery, SubSkill, Concept, Chapter, Subject, Student):
                for row in session.exec(select(model)).all():
                    session.delete(row)
            session.commit()

        concept_rows: list[Concept] = []
        for s_order, (section, name_fr, name_ar, color, chapters) in enumerate(CURRICULUM):
            subject = Subject(section=section, name_fr=name_fr, name_ar=name_ar, color=color, order=s_order)
            session.add(subject)
            session.commit()
            session.refresh(subject)

            for c_order, (ch_fr, ch_ar, weight, concepts) in enumerate(chapters):
                chapter = Chapter(
                    subject_id=subject.id, name_fr=ch_fr, name_ar=ch_ar,
                    order=c_order, exam_weight=weight,
                )
                session.add(chapter)
                session.commit()
                session.refresh(chapter)

                for co_order, (co_fr, co_ar, kind, subskills) in enumerate(concepts):
                    concept = Concept(
                        chapter_id=chapter.id, name_fr=co_fr, name_ar=co_ar,
                        order=co_order, kind=kind,
                    )
                    session.add(concept)
                    session.commit()
                    session.refresh(concept)
                    concept_rows.append(concept)
                    for ss in subskills:
                        session.add(SubSkill(concept_id=concept.id, name_fr=ss, name_ar=ss))
                session.commit()

        # Demo student.
        student = Student(name="Yassine", section="math", language="fr", streak=4)
        session.add(student)
        session.commit()
        session.refresh(student)

        # Seed mastery with a deterministic spread of due dates so the demo is
        # stable across reseeds.
        rng = random.Random(42)
        now = datetime.utcnow()
        for concept in concept_rows:
            mastery = round(rng.uniform(0.1, 0.95), 3)
            # Some due in the past (overdue), some today, some future.
            offset = rng.choice([-9, -5, -3, -1, 0, 0, 1, 2, 4, 7])
            reps = rng.randint(0, 4)
            session.add(Mastery(
                student_id=student.id,
                concept_id=concept.id,
                mastery=mastery,
                confidence=round(mastery * rng.uniform(0.7, 1.0), 3),
                ease=round(rng.uniform(1.8, 2.6), 2),
                interval_days=float(rng.choice([1, 3, 6, 12])),
                repetitions=reps,
                last_reviewed_at=now - timedelta(days=abs(offset) + 1),
                due_at=now + timedelta(days=offset, hours=rng.randint(0, 12)),
                avg_seconds=round(rng.uniform(20, 90), 1),
                error_count=rng.randint(0, 5),
                attempts=rng.randint(1, 10),
            ))
        session.commit()


if __name__ == "__main__":
    from .database import create_db_and_tables

    create_db_and_tables()
    seed(reset=True)
    print("Seeded database.")
