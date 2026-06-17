from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import create_db_and_tables
from .routers import curriculum, dev, mastery, planning, sessions, student
from .seed import seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    seed()  # idempotent: only seeds when the DB is empty
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(curriculum.router)
app.include_router(student.router)
app.include_router(mastery.router)
app.include_router(planning.router)
app.include_router(sessions.router)
app.include_router(dev.router)


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok", "app": settings.app_name}
