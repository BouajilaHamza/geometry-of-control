# Bac Companion 🇹🇳

A **retention-first bac companion** for Tunisian students. Instead of competing
with the existing archives and AI-correction tools, it acts as a year-long study
coach that:

- tells each student **what to revise this week**,
- tracks **mastery by concept** and surfaces **weak points early**, and
- makes **re-entry after inactivity painless** — never a wall of overdue work.

> Positioning: *A bac companion for Tunisian students that tells them what to
> revise every week, tracks weak points, and helps them recover without
> overwhelm.*

This repository is a self-contained monorepo (frontend + backend) ready to be
moved into its own repository.

```
bac-companion/
├── backend/    # FastAPI + SQLModel (SQLite) — curriculum, mastery, planning, recovery
└── frontend/   # React + TypeScript + Tailwind (Vite) — mobile-first, FR/AR + RTL
```

## What's implemented

| PRD area | Where |
|----------|-------|
| Curriculum competency graph (subject → chapter → concept → sub-skill) | `backend/app/models.py`, `seed.py` |
| Mastery profile + behavioural signals | `backend/app/models.py` (`Mastery`) |
| Planning engine (spaced repetition + adaptive practice) | `backend/app/scheduling.py`, `services.py` |
| **Backlog forgiveness / recovery mode** | `scheduling.py` (status, caps, recovery ranking) + `frontend .../Recovery.tsx` |
| Daily & weekly plans with workload caps | `services.build_daily_plan` / `build_weekly_plan` |
| Weak-point detection | `services.weak_points` |
| Home screen answering *today / weak / on-track* | `frontend/src/pages/Home.tsx` |
| Supportive bilingual tone (FR/AR, RTL) | `frontend/src/i18n.tsx` |

## Quick start

Two terminals.

### 1. Backend (port 8000)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The SQLite DB is auto-created and seeded with the Mathématiques section and a
demo student. Docs at <http://localhost:8000/docs>.

### 2. Frontend (port 5173)

```bash
cd frontend
npm install
cp .env.example .env     # points at http://localhost:8000
npm run dev
```

Open <http://localhost:5173>.

## Demoing the defining feature

The **backlog-forgiveness / recovery** system is the heart of the product. To
see it without waiting real days:

1. Open the app → **Profil** tab.
2. Under *Outils de démo*, tap **10** or **20** days away.
3. Go back to **Aujourd'hui** — the home screen switches to a supportive
   recovery banner ("Reprendre en douceur"), the daily queue is capped, new
   material is paused, and low-priority items are *set aside* rather than shown
   as a backlog wall.
4. Tap **Réinitialiser la démo** to restore the on-track state.

## Design principles honoured

1. **Retention before intelligence** — the core loop is a short, winnable daily
   session, not an AI chat surface.
2. **Never punish return** — recovery mode shows a small capped plan and hides
   low-value overdue items (`hidden_count`) instead of a count of 127.
3. **One clear action at a time** — the home screen always answers *what now?*.
4. **Mastery over volume** — progress is shown as competency strength.
5. **Local relevance** — Tunisian sections/subjects, FR + Arabic, dialectal
   nudges in the supportive copy.

## Tech notes

- No authentication in the MVP — a single seeded demo student.
- The scheduling engine (`backend/app/scheduling.py`) is framework-free and unit
  tested (`backend/tests/test_scheduling.py`).
- Tuning knobs (inactivity thresholds, daily caps) live in `backend/app/config.py`.
