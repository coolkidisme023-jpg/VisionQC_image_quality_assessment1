"""
Centralized, environment-variable-driven configuration.

No secrets or API keys are used anywhere in this application - VisionQC
performs all inference locally with a bundled scikit-learn model, so there
is nothing to configure beyond file paths, CORS origins, and upload limits.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


class Settings:
    """Application settings, overridable via environment variables."""

    #: Where the trained model + evaluation artifacts live.
    MODELS_DIR: Path = _env_path("VISIONQC_MODELS_DIR", REPO_ROOT / "models")
    MODEL_PATH: Path = MODELS_DIR / "quality_model.joblib"
    MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

    #: SQLite database file. Overridable so Docker can bind-mount a volume.
    DATABASE_PATH: Path = _env_path(
        "VISIONQC_DATABASE_PATH", REPO_ROOT / "data" / "visionqc.db"
    )

    #: Comma-separated list of allowed CORS origins for the frontend.
    FRONTEND_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.environ.get(
            "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    ]

    MAX_CONTENT_LENGTH: int = int(os.environ.get("VISIONQC_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

    HOST: str = os.environ.get("VISIONQC_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("VISIONQC_PORT", "8000"))
    DEBUG: bool = os.environ.get("VISIONQC_DEBUG", "false").lower() == "true"

    APP_VERSION: str = "1.0.0"


settings = Settings()
