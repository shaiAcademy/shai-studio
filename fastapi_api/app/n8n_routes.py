"""
n8n_routes.py — SSO интеграция с n8n.
Роутер: /api/n8n
"""
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user
from app.config import N8N_PROXY_URL, N8N_TOKEN_TTL_SECONDS
from app.models import User
from app.schemas import N8nRedirectResponse, N8nTokenValidateRequest, N8nTokenValidateResponse

from n8n_user_service import create_or_get_n8n_user

router = APIRouter(prefix="/api/n8n", tags=["n8n"])

# In-memory хранилище токенов (в продакшне лучше Redis)
# Формат: {token: {"email": str, "userId": str, "expiresAt": float}}
_token_store: dict = {}


def _cleanup_expired_tokens() -> None:
    now = time.time()
    expired = [t for t, d in _token_store.items() if d["expiresAt"] < now]
    for t in expired:
        del _token_store[t]


@router.post("/redirect", response_model=N8nRedirectResponse)
def redirect_to_n8n(current_user: User = Depends(get_current_user)):
    """
    Создаёт/находит пользователя в n8n и возвращает URL с одноразовым токеном.
    Требует аутентификации.
    """
    try:
        n8n_user = create_or_get_n8n_user(
            email=current_user.email,
            first_name=current_user.name,
        )

        token = secrets.token_hex(32)
        _token_store[token] = {
            "email": current_user.email,
            "userId": n8n_user.get("userId"),
            "expiresAt": time.time() + N8N_TOKEN_TTL_SECONDS,
        }

        _cleanup_expired_tokens()

        redirect_url = f"{N8N_PROXY_URL}/n8n-auth?token={token}"
        print(f"✅ Created n8n redirect for user: {current_user.email}")
        return N8nRedirectResponse(success=True, redirectUrl=redirect_url)

    except FileNotFoundError as e:
        print(f"❌ n8n database not found: {e}")
        raise HTTPException(status_code=503, detail="n8n service is not available. Database not found.")
    except Exception as e:
        print(f"❌ Error creating n8n redirect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-token", response_model=N8nTokenValidateResponse)
def validate_n8n_token(data: N8nTokenValidateRequest):
    """
    Проверяет одноразовый токен. Вызывается n8n-proxy сервисом.
    Публичный (без авторизации).
    """
    token_data = _token_store.get(data.token)

    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid token")

    if time.time() > token_data["expiresAt"]:
        del _token_store[data.token]
        raise HTTPException(status_code=401, detail="Token expired")

    # Одноразовый — удаляем после использования
    del _token_store[data.token]

    return N8nTokenValidateResponse(
        email=token_data["email"],
        userId=token_data.get("userId"),
    )
