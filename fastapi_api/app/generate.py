"""
generate.py — RunPod генерация (image/video) + конвертация WebP→MP4.
Роутер: /api/generate
"""
import glob
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.auth import get_current_user
from app.config import RUNPOD_API_BASE, RUNPOD_API_KEY, RUNPOD_VOLUME_ID, require_runpod_env
from app.database import get_db
from app.models import Task, User
from app.s3_client import make_s3_client
from app.schemas import TaskResponse

router = APIRouter(prefix="/api", tags=["generate"])

_RUNPOD_HEADERS = lambda: {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
}


# ── Video config helpers ───────────────────────────────────────────────────────

def _build_video_input(base_input: dict) -> dict:
    """
    Добавляет параметры для генерации 8-секундного видео (AnimateDiff).
    96 frames @ 12 FPS = 8 секунд.
    """
    target_fps = 12
    target_frames = 96

    extra = {
        "fps": target_fps,
        "frames_per_second": target_fps,
        "output_fps": target_fps,
        "decoding_fps": target_fps,
        "video_frames": target_frames,
        "num_frames": target_frames,
        "frames": target_frames,
        "n_frames": target_frames,
        "frame_count": target_frames,
        "length": target_frames,
        "total_frames": target_frames,
        # AnimateDiff Uniform Context Options (безопасная работа на 4090 24GB)
        "context_length": 16,
        "context_stride": 1,
        "context_overlap": 4,
        "context_schedule": "uniform",
        "uniform_context_options": {
            "context_length": 16,
            "context_stride": 1,
            "context_overlap": 4,
            "context_schedule": "uniform",
        },
        # Motion parameters
        "motion_bucket_id": 127,
        "decoding_t": 1,
        "motion_scale": 1.0,
    }
    return {**base_input, **extra}


# ── WebP → MP4 conversion ─────────────────────────────────────────────────────

def convert_and_upload_mp4(task_id: str, webp_key: str, s3, bucket: str) -> Optional[str]:
    """
    Скачивает animated WebP, конвертирует в MP4, загружает обратно в S3.
    Использует GIF как промежуточный формат (лучшая поддержка в FFmpeg).
    Возвращает ключ нового MP4 или None при ошибке.
    """
    if not shutil.which("ffmpeg"):
        print("❌ FFmpeg not installed. Cannot convert video.")
        return None

    tmp_in_path: Optional[str] = None
    tmp_gif_path: Optional[str] = None
    tmp_out_path: Optional[str] = None

    try:
        # 1. Скачиваем WebP
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp_in:
            s3.download_fileobj(bucket, webp_key, tmp_in)
            tmp_in_path = tmp_in.name

        tmp_gif_path = tmp_in_path + ".gif"
        tmp_out_path = tmp_in_path + ".mp4"

        # 2. WebP → GIF через ImageMagick (если доступен)
        use_gif = False
        if shutil.which("convert"):
            try:
                subprocess.run(
                    ["convert", tmp_in_path, tmp_gif_path],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
                )
                use_gif = True
                print(f"✅ Converted WebP → GIF for task {task_id}")
            except Exception as e:
                print(f"⚠️ GIF intermediate failed: {e}")

        # 3. GIF/WebP → MP4 через FFmpeg
        input_file = tmp_gif_path if use_gif and os.path.exists(tmp_gif_path) else tmp_in_path

        if use_gif:
            cmd = [
                "ffmpeg", "-y", "-ignore_loop", "0", "-i", input_file,
                "-t", "6",
                "-filter:v", "fps=24,setpts=3.0*PTS",
                "-pix_fmt", "yuv420p", "-c:v", "libx264",
                "-preset", "fast", "-movflags", "+faststart",
                tmp_out_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-framerate", "8", "-i", input_file,
                "-filter:v", "setpts=4*PTS",
                "-pix_fmt", "yuv420p", "-c:v", "libx264",
                "-preset", "fast", "-movflags", "+faststart",
                tmp_out_path,
            ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

        # 4. Загружаем MP4
        mp4_key = webp_key.replace(".webp", ".mp4")
        if mp4_key == webp_key:
            mp4_key += ".mp4"

        with open(tmp_out_path, "rb") as f:
            s3.upload_fileobj(
                f, bucket, mp4_key,
                ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"},
            )

        print(f"✅ Uploaded MP4: {mp4_key}")
        return mp4_key

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        print(f"❌ FFmpeg conversion failed for {task_id}: {err}")
        return None
    except Exception as e:
        print(f"❌ Conversion failed for {task_id}: {e}")
        return None
    finally:
        for path in [tmp_in_path, tmp_gif_path, tmp_out_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate/{kind}")
def generate(
    kind: str,
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    kind: image | video
    body: {prompt, steps, seed?}
    """
    require_runpod_env()
    kind = kind.lower()
    if kind not in ("image", "video"):
        raise HTTPException(400, "kind must be image or video")

    prompt = (payload or {}).get("prompt")
    steps = int((payload or {}).get("steps", 30))
    seed = (payload or {}).get("seed")

    if not prompt:
        raise HTTPException(422, "prompt is required")

    base_input: dict = {"type": kind, "prompt": prompt, "steps": steps}
    job_input = _build_video_input(base_input) if kind == "video" else base_input
    if seed is not None:
        job_input["seed"] = int(seed)

    r = requests.post(
        f"{RUNPOD_API_BASE}/run",
        json={"input": job_input},
        headers=_RUNPOD_HEADERS(),
        timeout=60,
    )
    if r.status_code >= 400:
        raise HTTPException(502, f"RunPod /run failed: {r.status_code} {r.text}")

    data = r.json()

    # Сохраняем задачу в БД
    task_id = data.get("id")
    if task_id:
        db.add(Task(
            user_id=current_user.id,
            task_id=task_id,
            prompt=prompt,
            kind=kind,
            status=data.get("status", "IN_QUEUE"),
        ))
        db.commit()

    data["user"] = {"id": current_user.id, "email": current_user.email}
    return data


@router.get("/generate/status/{task_id}")
def generate_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_runpod_env()

    r = requests.get(
        f"{RUNPOD_API_BASE}/status/{task_id}",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        timeout=60,
    )
    if r.status_code >= 400:
        raise HTTPException(502, f"RunPod /status failed: {r.status_code} {r.text}")

    data = r.json()

    if data.get("status") == "COMPLETED":
        out = data.get("output") or {}
        media_key = out.get("media_key")

        # Fallback: если вернулся filename
        if not media_key and out.get("filename"):
            media_key = f"ComfyUI/output/{out['filename']}"

        # Конвертация WebP → MP4 если нужно
        if media_key and media_key.lower().endswith(".webp"):
            task_record = db.query(Task).filter(Task.task_id == task_id).first()
            if task_record and task_record.media_url and task_record.media_url.endswith(".mp4"):
                # Уже сконвертировано
                data.setdefault("output", {})["media_url"] = task_record.media_url
            else:
                print(f"🔄 Converting WebP → MP4 for task {task_id}...")
                s3 = make_s3_client()
                new_key = convert_and_upload_mp4(task_id, media_key, s3, RUNPOD_VOLUME_ID)
                if new_key:
                    media_key = new_key

        if media_key:
            media_url = f"/api/media/{media_key}"
            data.setdefault("output", {})["media_url"] = media_url

            # Обновляем запись в БД
            task_record = db.query(Task).filter(Task.task_id == task_id).first()
            if task_record:
                if task_record.status != "COMPLETED" or task_record.media_url != media_url:
                    task_record.status = "COMPLETED"
                    task_record.media_url = media_url
                    db.commit()

    else:
        # Синхронизируем статус
        task_record = db.query(Task).filter(Task.task_id == task_id).first()
        if task_record:
            runpod_status = data.get("status")
            if runpod_status and task_record.status != runpod_status:
                task_record.status = runpod_status
                db.commit()

    return data


@router.get("/tasks", response_model=list[TaskResponse])
def get_user_tasks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(Task)
        .filter(Task.user_id == current_user.id)
        .order_by(desc(Task.created_at))
        .all()
    )
