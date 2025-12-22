import os
import mimetypes
from typing import Optional, Tuple
from datetime import datetime, timedelta

import requests
import boto3
from botocore.client import Config

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

# ----------------------------
# App + CORS
# ----------------------------
app = FastAPI(title="RunPod Gateway API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для прод: ограничь доменом фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# ENV (RunPod)
# ----------------------------
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")

# RunPod S3 API key (отдельный от обычного RunPod API key!)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

RUNPOD_S3_REGION = os.getenv("RUNPOD_S3_REGION", "EU-RO-1")
RUNPOD_S3_ENDPOINT_URL = os.getenv("RUNPOD_S3_ENDPOINT_URL", "https://s3api-eu-ro-1.runpod.io/")
RUNPOD_VOLUME_ID = os.getenv("RUNPOD_VOLUME_ID", "")  # bucket = Network Volume ID

API_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# ----------------------------
# ENV (Auth/JWT + DB)
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-please")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ----------------------------
# DB
# ----------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------------------
# Schemas
# ----------------------------
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


# ----------------------------
# Auth helpers
# ----------------------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(pw: str) -> str:
    # bcrypt limit: passlib/bcrypt используют только первые 72 bytes
    # мы ограничили пароль в pydantic до 72 символов
    return pwd_context.hash(pw)


def create_access_token(subject: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)) -> User:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        sub = payload.get("sub")
        if not sub:
            raise cred_exc
        user_id = int(sub)
    except (JWTError, ValueError):
        raise cred_exc

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise cred_exc
    return user


# ----------------------------
# RunPod/S3 helpers
# ----------------------------
def require_env():
    miss = []
    if not RUNPOD_API_KEY:
        miss.append("RUNPOD_API_KEY")
    if not RUNPOD_ENDPOINT_ID:
        miss.append("RUNPOD_ENDPOINT_ID")
    if not AWS_ACCESS_KEY_ID:
        miss.append("AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        miss.append("AWS_SECRET_ACCESS_KEY")
    if not RUNPOD_VOLUME_ID:
        miss.append("RUNPOD_VOLUME_ID")
    if miss:
        raise RuntimeError("Missing env vars: " + ", ".join(miss))


def s3_client():
    cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=RUNPOD_S3_REGION,
        endpoint_url=RUNPOD_S3_ENDPOINT_URL,
        config=cfg,
    )


def guess_mime(key: str) -> str:
    mt, _ = mimetypes.guess_type(key)
    return mt or "application/octet-stream"


def parse_range(range_header: str, size: int) -> Optional[Tuple[int, int]]:
    if not range_header or not range_header.startswith("bytes="):
        return None
    part = range_header.replace("bytes=", "").strip()
    if "-" not in part:
        return None
    start_s, end_s = part.split("-", 1)
    if start_s == "" and end_s == "":
        return None

    if start_s == "":
        length = int(end_s)
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else (size - 1)

    if start < 0 or end < start:
        return None
    end = min(end, size - 1)
    return start, end


# ----------------------------
# Public endpoints
# ----------------------------
@app.get("/api/health")
def health():
    return {"ok": True}


# ----------------------------
# Auth endpoints
# ----------------------------
@app.post("/api/auth/register", response_model=AuthResponse, status_code=201)
def register(payload: UserCreate, db=Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email,
        name=payload.name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id))
    return AuthResponse(access_token=token)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: UserLogin, db=Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(user.id))
    return AuthResponse(access_token=token)


@app.get("/api/auth/me", response_model=MeResponse)
def me(current_user: User = Depends(get_current_user)):
    return MeResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at,
    )


# ----------------------------
# Protected endpoints (RunPod)
# ----------------------------
@app.post("/api/generate/{kind}")
def generate(kind: str, payload: dict, current_user: User = Depends(get_current_user)):
    """
    kind: image|video
    payload: {prompt, steps, seed?}
    """
    require_env()

    kind = kind.lower()
    if kind not in ("image", "video"):
        raise HTTPException(400, "kind must be image or video")

    prompt = (payload or {}).get("prompt")
    steps = int((payload or {}).get("steps", 30))
    seed = (payload or {}).get("seed")

    if not prompt:
        raise HTTPException(422, "prompt is required")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    body = {
        "input": {
            "type": kind,
            "prompt": prompt,
            "steps": steps,
        }
    }
    if seed is not None:
        body["input"]["seed"] = int(seed)

    r = requests.post(f"{API_BASE}/run", json=body, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(502, f"RunPod /run failed: {r.status_code} {r.text}")

    data = r.json()
    data["user"] = {"id": current_user.id, "email": current_user.email}
    return data


@app.get("/api/generate/status/{task_id}")
def generate_status(task_id: str, current_user: User = Depends(get_current_user)):
    require_env()

    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    r = requests.get(f"{API_BASE}/status/{task_id}", headers=headers, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(502, f"RunPod /status failed: {r.status_code} {r.text}")

    data = r.json()

    # добавляем media_url для фронта
    if data.get("status") == "COMPLETED":
        out = data.get("output") or {}
        media_key = out.get("media_key")

        # fallback если у тебя возвращается filename
        if not media_key and out.get("filename"):
            media_key = f"ComfyUI/output/{out['filename']}"

        if media_key:
            data.setdefault("output", {})
            data["output"]["media_url"] = f"/api/media/{media_key}"

    return data


# ----------------------------
# PUBLIC media proxy (Solution A)
# ----------------------------
@app.get("/api/media/{key:path}")
def get_media(key: str, request: Request):
    """
    ПУБЛИЧНЫЙ endpoint для медиа.
    Важно: <img>/<video> не могут послать Authorization header, поэтому auth тут убран.
    """
    require_env()

    s3 = s3_client()
    bucket = RUNPOD_VOLUME_ID

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise HTTPException(404, f"Not found: {key}. {e}")

    size = int(head.get("ContentLength", 0))
    content_type = guess_mime(key)

    range_header = request.headers.get("range")
    byte_range = parse_range(range_header, size) if size else None

    if byte_range:
        start, end = byte_range
        rng = f"bytes={start}-{end}"
        obj = s3.get_object(Bucket=bucket, Key=key, Range=rng)

        def iter_body():
            yield from obj["Body"].iter_chunks(chunk_size=1024 * 1024)

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(iter_body(), status_code=206, media_type=content_type, headers=headers)

    obj = s3.get_object(Bucket=bucket, Key=key)

    def iter_body():
        yield from obj["Body"].iter_chunks(chunk_size=1024 * 1024)

    headers = {"Accept-Ranges": "bytes"}
    if size:
        headers["Content-Length"] = str(size)

    return StreamingResponse(iter_body(), media_type=content_type, headers=headers)
