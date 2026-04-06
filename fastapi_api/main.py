"""
main.py — точка входа FastAPI приложения.

Структура модулей:
  app/config.py       — переменные окружения
  app/database.py     — SQLAlchemy engine / session
  app/models.py       — ORM модели (User, Task)
  app/schemas.py      — Pydantic схемы
  app/auth.py         — JWT + роутер /api/auth
  app/generate.py     — RunPod генерация + роутер /api/generate
  app/media.py        — S3 прокси + роутер /api/media
  app/n8n_routes.py   — n8n SSO + роутер /api/n8n
  n8n_user_service.py — утилита записи пользователей в SQLite n8n
"""
import os
import sys

# Гарантируем что /app (рабочая директория) есть в sys.path,
# чтобы модули внутри app/ могли импортировать n8n_user_service.py
sys.path.insert(0, os.path.dirname(__file__))

import firebase_admin
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as firebase_auth, credentials

from app.database import engine
from app.models import Base
# ── Firebase инициализируется один раз при старте сервера. ───────────────────
_FIREBASE_KEY_PATH = os.path.join(os.path.dirname(__file__), "firebase_key.json")

try:
    if os.path.exists(_FIREBASE_KEY_PATH):
        cred = credentials.Certificate(_FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
        print(f"✅ Firebase Admin initialized using {_FIREBASE_KEY_PATH}")
        try:
            firebase_auth.list_users(max_results=1)
            print("✅ Firebase Auth connection verified.")
        except Exception as e:
            print(f"⚠️ Firebase initialized but Auth check failed: {e}")
    else:
        print(f"❌ WARNING: Firebase key not found at {_FIREBASE_KEY_PATH}")
except ValueError:
    print("ℹ️ Firebase app already initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Firebase: {e}")

# ── Создаём таблицы ───────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="RunPod Gateway API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Роутеры ───────────────────────────────────────────────────────────────────
from app.auth import router as auth_router        # noqa: E402
from app.generate import router as generate_router  # noqa: E402
from app.media import router as media_router       # noqa: E402
from app.n8n_routes import router as n8n_router    # noqa: E402

app.include_router(auth_router)
app.include_router(generate_router)
app.include_router(media_router)
app.include_router(n8n_router)


@app.get("/api/health")
def health():
    return {"ok": True}
