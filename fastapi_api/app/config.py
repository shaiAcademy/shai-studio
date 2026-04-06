"""
config.py — все конфигурационные переменные из окружения.
"""
import os
from dotenv import load_dotenv

# Загружаем .env из корня проекта (родительская папка fastapi_api)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


# ── RunPod ────────────────────────────────────────────────────────────────────
RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_BASE: str = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# ── RunPod S3 (Network Volume) ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
RUNPOD_S3_REGION: str = os.getenv("RUNPOD_S3_REGION", "EU-RO-1")
RUNPOD_S3_ENDPOINT_URL: str = os.getenv(
    "RUNPOD_S3_ENDPOINT_URL", "https://s3api-eu-ro-1.runpod.io/"
)
RUNPOD_VOLUME_ID: str = os.getenv("RUNPOD_VOLUME_ID", "")  # bucket = Volume ID

# ── Database ────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# ── JWT ─────────────────────────────────────────────────────────────────────
JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-please")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRES_MIN", os.getenv("JWT_EXPIRE_MINUTES", "60")))

# ── n8n Integration ──────────────────────────────────────────────────────────
N8N_PROXY_URL: str = os.getenv("N8N_PROXY_URL", "https://n8n-proxy.shai.academy")
N8N_TOKEN_TTL_SECONDS: int = int(os.getenv("N8N_TOKEN_TTL_SECONDS", "300"))
N8N_DB_PATH: str = os.getenv(
    "N8N_DB_PATH",
    "/var/lib/docker/volumes/n8n_n8n_data/_data/database.sqlite",
)


def require_runpod_env() -> None:
    """Бросает HTTPException 500 если нужные переменные не заданы."""
    from fastapi import HTTPException

    missing = [
        name
        for name, val in [
            ("RUNPOD_API_KEY", RUNPOD_API_KEY),
            ("RUNPOD_ENDPOINT_ID", RUNPOD_ENDPOINT_ID),
            ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
            ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
            ("RUNPOD_VOLUME_ID", RUNPOD_VOLUME_ID),
        ]
        if not val
    ]
    if missing:
        msg = "Missing env vars: " + ", ".join(missing)
        print(f"❌ {msg}")
        raise HTTPException(status_code=500, detail=msg)
