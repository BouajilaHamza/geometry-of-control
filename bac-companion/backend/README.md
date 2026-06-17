# Bac Companion — Backend (FastAPI)

REST API for the Tunisian Bac Companion. Owns the curriculum competency graph,
the per-concept mastery profile, the planning engine, and the **backlog
forgiveness / recovery** system that is the product's defining feature.

## Stack
- **FastAPI** + **SQLModel** (SQLAlchemy + Pydantic) on **SQLite**.
- No auth in the MVP — a single seeded demo student.

## Run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload      # http://localhost:8000
```

The database is created and seeded automatically on first start.
Interactive docs: <http://localhost:8000/docs>.

## Tests

```bash
PYTHONPATH=. python tests/test_scheduling.py    # or: pytest
```

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | Liveness check |
| GET  | `/api/curriculum/subjects` | List subjects |
| GET  | `/api/curriculum/subjects/{id}` | Subject → chapter → concept → sub-skill tree |
| GET  | `/api/student` | The demo student profile |
| GET  | `/api/student/status` | Status, caps, streak, supportive headline |
| GET  | `/api/mastery` | Mastery by subject & chapter |
| GET  | `/api/mastery/weak-points` | Top weak points with reasons |
| GET  | `/api/plan/today` | Today's capped session (reviews + drills + maybe 1 new) |
| GET  | `/api/plan/week` | Weekly plan spread across days |
| GET  | `/api/plan/recovery` | Gentle, capped restart after inactivity |
| POST | `/api/sessions` | Submit a completed session → updates mastery + schedule |
| POST | `/api/dev/reseed` | Reset & reseed the demo data |
| POST | `/api/dev/simulate-inactivity?days=N` | Backdate activity to demo recovery mode |

## How the engine works (`app/scheduling.py`)

- **Spaced repetition** — an SM-2 variant (`update_schedule`) drives review
  intervals for memory-heavy concepts and folds results into a smooth 0–1
  mastery estimate.
- **Status detection** — inactivity is the dominant signal, overdue queue size
  the secondary one: `on_track → mild_backlog → major_backlog → re_entry`.
- **Workload caps** — each status caps the number of *visible* daily tasks, so
  the home screen never shows a backlog wall. New material is paused once the
  student falls into recovery.
- **Recovery ranking** — `recovery_score` blends exam importance, weakness
  severity, recency and forgetting risk, then reintroduces work gradually over
  a few days. Low-priority overdue items are kept off the visible queue
  (`hidden_count`) instead of being dumped on the student.

Tuning knobs (thresholds, caps) live in `app/config.py`.
