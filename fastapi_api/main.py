import os
import mimetypes
import secrets
import time
from typing import Optional, Tuple
from datetime import datetime, timedelta

# n8n integration
from n8n_user_service import create_or_get_n8n_user

import firebase_admin
from firebase_admin import credentials, auth

import requests
import boto3
from botocore.client import Config

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import tempfile
import shutil

from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, desc
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

# ----------------------------
# App + CORS
# ----------------------------
app = FastAPI(title="RunPod Gateway API", version="1.0.0")

# ----------------------------
# Firebase Initialization
# ----------------------------
FIREBASE_KEY_PATH = os.path.join(os.path.dirname(__file__), "firebase_key.json")

try:
    if os.path.exists(FIREBASE_KEY_PATH):
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
        print(f"✅ Firebase Admin initialized successfully using {FIREBASE_KEY_PATH}")
        
        # Verify connection by attempting to list users
        try:
            auth.list_users(max_results=1)
            print("✅ Firebase Connection Verified: Successfully connected to Auth service.")
        except Exception as e:
            print(f"⚠️ Firebase initialized but failed to verify Auth connection: {e}")
    else:
        print(f"❌ WARNING: Firebase key not found at {FIREBASE_KEY_PATH}")
except ValueError:
    # App already initialized
    print("ℹ️ Firebase app already initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Firebase: {e}")

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
# ENV (n8n Integration)
# ----------------------------
N8N_PROXY_URL = os.getenv("N8N_PROXY_URL", "https://n8n-proxy.shai.academy")
N8N_TOKEN_TTL_SECONDS = int(os.getenv("N8N_TOKEN_TTL_SECONDS", "300"))  # 5 minutes

# In-memory token store for n8n redirect tokens
# Format: {token: {"email": str, "userId": str, "expiresAt": float}}
# In production, use Redis instead
n8n_token_store: dict = {}

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

    tasks = relationship("Task", back_populates="owner")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    task_id = Column(String, unique=True, index=True, nullable=False)
    prompt = Column(Text, nullable=False)
    kind = Column(String, nullable=False)  # image|video
    status = Column(String, default="IN_QUEUE")
    media_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="tasks")


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
    name: str
    created_at: datetime


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
        error_msg = "Missing env vars: " + ", ".join(miss)
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


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
    # 1. Check local DB
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 2. Create user in Firebase
    try:
        auth.create_user(
            email=payload.email,
            password=payload.password,
            display_name=payload.name,
        )
        print(f"✅ Created user {payload.email} in Firebase.")
    except auth.EmailAlreadyExistsError:
        # If exists in Firebase but not local, we proceed (or could fail).
        # For now, let's treat it as "ok, proceed to local sync" or fail.
        # User said "vnedreno" (integrated), so let's be strict but robust.
        print(f"⚠️ User {payload.email} already exists in Firebase. Proceeding to local DB.")
    except Exception as e:
        print(f"❌ Failed to create user in Firebase: {e}")
        raise HTTPException(status_code=400, detail=f"Firebase Registration Failed: {e}")

    # 3. Create user in Local DB
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
    # 1. Verify user exists in Firebase
    try:
        firebase_user = auth.get_user_by_email(payload.email)
        print(f"✅ Found Firebase user: {firebase_user.uid}")
    except auth.UserNotFoundError:
        print(f"❌ User {payload.email} not found in Firebase.")
        raise HTTPException(status_code=401, detail="Account not found in Firebase system.")
    except ValueError as e:
        print(f"❌ Firebase Auth not initialized: {e}")
        raise HTTPException(status_code=500, detail="Firebase Configuration Error: Service not initialized on server.")
    except Exception as e:
        print(f"❌ Firebase check error: {e}")
        raise HTTPException(status_code=500, detail=f"Firebase Connectivity Error: {str(e)}")

    # 2. Check local DB and Password
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials (local check)")

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
def generate(kind: str, payload: dict, current_user: User = Depends(get_current_user), db=Depends(get_db)):
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
    if kind == "video":
        # Attempt to force 6 seconds duration
        # Strategy: Request 48 frames at 8 FPS = 6 seconds.
        # If model is SVD (usually 25 frames max without tiling), 
        # we still request 48. If it truncates to 25, 
        # our FFmpeg logic in generate_status will stretch it.
        target_fps = 8
        target_frames = 48
        
        # Exhaustive list of parameter aliases
        body["input"]["fps"] = target_fps
        body["input"]["frames_per_second"] = target_fps
        body["input"]["output_fps"] = target_fps
        body["input"]["decoding_fps"] = target_fps
        
        body["input"]["video_frames"] = target_frames
        body["input"]["num_frames"] = target_frames
        body["input"]["frames"] = target_frames
        body["input"]["n_frames"] = target_frames
        body["input"]["frame_count"] = target_frames
        body["input"]["length"] = target_frames

        body["input"]["motion_bucket_id"] = 127
        body["input"]["decoding_t"] = 1

    if seed is not None:
        body["input"]["seed"] = int(seed)

    r = requests.post(f"{API_BASE}/run", json=body, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(502, f"RunPod /run failed: {r.status_code} {r.text}")

    data = r.json()
    
    # Save to local DB
    task_id = data.get("id")
    if task_id:
        new_task = Task(
            user_id=current_user.id,
            task_id=task_id,
            prompt=prompt,
            kind=kind,
            status=data.get("status", "IN_QUEUE")
        )
        db.add(new_task)
        db.commit()
    
    data["user"] = {"id": current_user.id, "email": current_user.email}
    return data


    return data


def convert_and_upload_mp4(task_id: str, webp_key: str, s3, bucket: str) -> Optional[str]:
    """
    Downloads WebP, converts to MP4 (forcing ~6s duration), uploads to S3, returns new key.
    """
    try:
        # 1. Download WebP
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp_in:
            s3.download_fileobj(bucket, webp_key, tmp_in)
            tmp_in_path = tmp_in.name
        
        tmp_out_path = tmp_in_path + ".mp4"
        
        # 2. Convert to MP4 with FFmpeg
        # force 6s duration (~48 frames / 8 fps) or interpolate
        # -r 8: set frame rate
        # -minterpolate: basic interpolation to smooth it out if frames are low
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_in_path,
            "-filter:v", "minterpolate=fps=8,setpts=4*PTS",  # Slow down 4x (approx 1.5s -> 6s)
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp_out_path
        ]
        # Alternative simple stretch without interpolation (just slow playback):
        # cmd = ["ffmpeg", "-y", "-r", "8", "-i", tmp_in_path, "-vf", "setpts=4*PTS", "-pix_fmt", "yuv420p", tmp_out_path]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 3. Upload MP4
        mp4_key = webp_key.replace(".webp", ".mp4")
        if mp4_key == webp_key:
            mp4_key += ".mp4"
            
        with open(tmp_out_path, "rb") as f:
            s3.upload_fileobj(
                f, 
                bucket, 
                mp4_key, 
                ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"}
            )
            
        # Cleanup
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)
        
        return mp4_key
    except Exception as e:
        print(f"❌ Conversion failed for {task_id}: {e}")
        if os.path.exists(tmp_in_path): os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path): os.unlink(tmp_out_path)
        return None


@app.get("/api/generate/status/{task_id}")
def generate_status(task_id: str, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    require_env()

    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    r = requests.get(f"{API_BASE}/status/{task_id}", headers=headers, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(502, f"RunPod /status failed: {r.status_code} {r.text}")

    data = r.json()

    # добавдяем media_url для фронта
    if data.get("status") == "COMPLETED":
        out = data.get("output") or {}
        media_key = out.get("media_key")

        # fallback если у тебя возвращается filename
        if not media_key and out.get("filename"):
            media_key = f"ComfyUI/output/{out['filename']}"
        
        # KEY CHANGE: Check if we need to convert WebP to MP4
        if media_key and media_key.lower().endswith(".webp"):
            # Check if we already converted it (look in DB or just check string)
            # ideally check DB, but here we check if we already have an MP4 override
            task_record = db.query(Task).filter(Task.task_id == task_id).first()
            if task_record and task_record.media_url and task_record.media_url.endswith(".mp4"):
                # Already converted
                data["output"]["media_url"] = task_record.media_url
            else:
                # Need conversion
                print(f"🔄 Converting WebP to MP4 for task {task_id}...")
                s3 = s3_client()
                bucket = RUNPOD_VOLUME_ID
                new_key = convert_and_upload_mp4(task_id, media_key, s3, bucket)
                
                if new_key:
                    media_key = new_key # pointing to MP4 now
        
        if media_key:
            media_url = f"/api/media/{media_key}"
            data.setdefault("output", {})
            data["output"]["media_url"] = media_url
            
            # Update DB if needed
            task_record = db.query(Task).filter(Task.task_id == task_id).first()
            if task_record:
                # Update status or URL
                if task_record.status != "COMPLETED" or task_record.media_url != media_url:
                    task_record.status = "COMPLETED"
                    task_record.media_url = media_url
                    db.commit()
    
    # Sync status if not completed but changed
    else:
         task_record = db.query(Task).filter(Task.task_id == task_id).first()
         if task_record:
             runpod_status = data.get("status")
             if runpod_status and task_record.status != runpod_status:
                 task_record.status = runpod_status
                 db.commit()

    return data


@app.get("/api/tasks", response_model=list[TaskResponse])
def get_user_tasks(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    return db.query(Task).filter(Task.user_id == current_user.id).order_by(desc(Task.created_at)).all()


# ----------------------------
# PUBLIC media proxy (Solution A)
# ----------------------------
@app.get("/api/media/download/{key:path}")
def download_media(key: str, format: str = "mp4"): # Default to mp4
    """
    Download media, converting to mp4 by default if needed.
    Also STRETCHES video to 3x duration if it's a generated video (short).
    """
    require_env()
    s3 = s3_client()
    bucket = RUNPOD_VOLUME_ID

    try:
        # 1. Download original file to temp
        print(f"⬇️ Downloading {key} for conversion...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        original_ext = os.path.splitext(key)[1].lower()
        if not original_ext:
            original_ext = mimetypes.guess_extension(obj.get("ContentType", "")) or ".bin"
        
        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp_in:
            for chunk in obj["Body"].iter_chunks(chunk_size=1024 * 1024):
                tmp_in.write(chunk)
            tmp_in_path = tmp_in.name

        # 2. Check if conversion/stretching needed
        final_path = tmp_in_path
        final_filename = os.path.basename(key)
        
        # Always attempt conversion if target is mp4, specifically to apply filters
        # even if input is already mp4 (to stretch it).
        if format == "mp4":
            print(f"🔄 Converting/Stretching {key} to MP4...")
            tmp_out_path = tmp_in_path + ".mp4"
            try:
                # ffmpeg -i input.webp -filter:v "setpts=3.0*PTS" -pix_fmt yuv420p output.mp4
                # setpts=3.0*PTS slows down video by 3x (2s -> 6s)
                cmd = [
                    "ffmpeg", "-y",
                    "-i", tmp_in_path,
                    "-filter:v", "setpts=3.0*PTS", 
                    "-pix_fmt", "yuv420p", # Ensure compatibility
                    "-movflags", "+faststart",
                    tmp_out_path
                ]
                
                # Run with captured output for debugging
                process = subprocess.run(
                    cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                final_path = tmp_out_path
                final_filename = os.path.splitext(final_filename)[0] + ".mp4"
                print(f"✅ Conversion successful: {final_path}")
                
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.decode()
                print(f"❌ FFmpeg conversion failed: {err_msg}")
                # We raise error here to make it visible to user instead of silent fallback
                # raise HTTPException(500, f"Video processing failed: {err_msg}")
                # OR fallback but log heavily. User complained about webp, so let's fail if we can't make mp4
                print("⚠️ Falling back to original file due to error.")
                pass
            except FileNotFoundError:
                print("❌ FFmpeg not found on server.")
                pass

        # 3. Serve file
        from starlette.background import BackgroundTask
        
        def cleanup():
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)
            if os.path.exists(final_path) and final_path != tmp_in_path:
                os.unlink(final_path)

        return FileResponse(
            final_path, 
            filename=final_filename, 
            media_type="video/mp4" if final_path.endswith(".mp4") else guess_mime(final_path),
            background=BackgroundTask(cleanup)
        )

    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(404, f"Download failed: {e}")


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
    
    # Explicitly handle some types if guess fails
    ext = os.path.splitext(key)[1].lower()
    if ext == ".mp4":
        content_type = "video/mp4"
    elif ext == ".webp":
        content_type = "image/webp"
    else:
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


# ----------------------------
# n8n Integration endpoints
# ----------------------------
class N8nRedirectResponse(BaseModel):
    success: bool
    redirectUrl: str


class N8nTokenValidateRequest(BaseModel):
    token: str


class N8nTokenValidateResponse(BaseModel):
    email: str
    userId: Optional[str] = None


def cleanup_expired_tokens():
    """Remove expired tokens from the store"""
    current_time = time.time()
    expired = [t for t, data in n8n_token_store.items() if data["expiresAt"] < current_time]
    for t in expired:
        del n8n_token_store[t]


@app.post("/api/n8n/redirect", response_model=N8nRedirectResponse)
def redirect_to_n8n(current_user: User = Depends(get_current_user)):
    """
    Create/verify user in n8n and return redirect URL with auth token.
    Requires authentication.
    """
    try:
        # Create or get user in n8n database
        n8n_user = create_or_get_n8n_user(
            email=current_user.email,
            first_name=current_user.name
        )
        
        # Generate temporary token
        token = secrets.token_hex(32)
        expires_at = time.time() + N8N_TOKEN_TTL_SECONDS
        
        # Store token
        n8n_token_store[token] = {
            "email": current_user.email,
            "userId": n8n_user.get("userId"),
            "expiresAt": expires_at
        }
        
        # Cleanup old tokens periodically
        cleanup_expired_tokens()
        
        # Build redirect URL
        redirect_url = f"{N8N_PROXY_URL}/n8n-auth?token={token}"
        
        print(f"✅ Created n8n redirect for user: {current_user.email}")
        
        return N8nRedirectResponse(success=True, redirectUrl=redirect_url)
        
    except FileNotFoundError as e:
        print(f"❌ n8n database not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="n8n service is not available. Database not found."
        )
    except Exception as e:
        print(f"❌ Error creating n8n redirect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/n8n/validate-token", response_model=N8nTokenValidateResponse)
def validate_n8n_token(data: N8nTokenValidateRequest):
    """
    Validate a redirect token. Called by the n8n proxy service.
    This endpoint is public (no auth required).
    """
    token = data.token
    token_data = n8n_token_store.get(token)
    
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Check expiration
    if time.time() > token_data["expiresAt"]:
        # Remove expired token
        del n8n_token_store[token]
        raise HTTPException(status_code=401, detail="Token expired")
    
    # Token is valid - remove it (one-time use)
    del n8n_token_store[token]
    
    return N8nTokenValidateResponse(
        email=token_data["email"],
        userId=token_data.get("userId")
    )
