from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings. Override via environment variables or a .env file."""

    app_name: str = "Tunisian Bac Companion API"
    database_url: str = "sqlite:///./bac_companion.db"

    # CORS origins allowed to call the API (the Vite dev server by default).
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # --- Backlog-forgiveness tuning knobs (see scheduling.py) ---------------
    # Days of inactivity before we switch the student into recovery mode.
    recovery_threshold_days: int = 7
    # Days of inactivity that triggers a full re-entry week.
    reentry_threshold_days: int = 14

    # Visible daily task caps by status. These bound the home screen so a
    # returning student never sees a "127 tasks overdue" wall.
    cap_on_track: int = 9
    cap_mild_backlog: int = 8
    cap_major_backlog: int = 6
    cap_reentry: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
