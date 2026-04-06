"""
schemas.py — Pydantic схемы запросов и ответов.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ── Auth ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6, max_length=72)  # bcrypt limit


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    id: int
    email: EmailStr
    name: str
    created_at: datetime


# ── Tasks ────────────────────────────────────────────────────────────────────

class TaskResponse(BaseModel):
    id: int
    task_id: str
    prompt: str
    kind: str
    status: str
    media_url: Optional[str] = None
    created_at: datetime

    class Config:
        orm_mode = True


# ── n8n ──────────────────────────────────────────────────────────────────────

class N8nRedirectResponse(BaseModel):
    success: bool
    redirectUrl: str


class N8nTokenValidateRequest(BaseModel):
    token: str


class N8nTokenValidateResponse(BaseModel):
    email: str
    userId: Optional[str] = None
