"""
media.py — S3-прокси для медиафайлов.
Роутер: /api/media
"""
import glob
import mimetypes
import os
import shutil
import subprocess
import tempfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.config import RUNPOD_VOLUME_ID, require_runpod_env
from app.s3_client import guess_mime, make_s3_client, parse_range

router = APIRouter(prefix="/api/media", tags=["media"])


# ── Download (with optional MP4 conversion) ──────────────────────────────────

@router.get("/download/{key:path}")
def download_media(key: str, format: str = None):
    """
    Скачивает медиафайл из S3.
    - format=mp4 → конвертирует animated WebP в MP4 и отдаёт клиенту.
    """
    require_runpod_env()
    s3 = make_s3_client()
    bucket = RUNPOD_VOLUME_ID

    tmp_in_path: str | None = None
    tmp_frames_dir: str | None = None
    tmp_out_path: str | None = None

    try:
        print(f"⬇️ Downloading {key}...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        original_ext = os.path.splitext(key)[1].lower()
        if not original_ext:
            original_ext = mimetypes.guess_extension(obj.get("ContentType", "")) or ".bin"

        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp_in:
            for chunk in obj["Body"].iter_chunks(chunk_size=1024 * 1024):
                tmp_in.write(chunk)
            tmp_in_path = tmp_in.name

        final_path = tmp_in_path
        final_filename = os.path.basename(key)

        # Конвертация в MP4 только если явно запрошено
        if format == "mp4":
            print(f"🔄 Converting {key} to MP4...")
            tmp_out_path = tmp_in_path + ".mp4"
            conversion_success = False

            is_webp = original_ext == ".webp"

            if is_webp:
                tmp_frames_dir = tempfile.mkdtemp()

                # Метод 1: webpmux — извлечение кадров
                if shutil.which("webpmux") and not conversion_success:
                    try:
                        frame_files = []
                        for i in range(1, 100):
                            frame_path = os.path.join(tmp_frames_dir, f"frame_{i:04d}.webp")
                            result = subprocess.run(
                                ["webpmux", "-get", "frame", str(i), tmp_in_path, "-o", frame_path],
                                capture_output=True, timeout=10,
                            )
                            if result.returncode == 0 and os.path.exists(frame_path):
                                frame_files.append(frame_path)
                            else:
                                break

                        if frame_files:
                            frame_pattern = os.path.join(tmp_frames_dir, "frame_%04d.webp")
                            subprocess.run(
                                [
                                    "ffmpeg", "-y", "-framerate", "8", "-i", frame_pattern,
                                    "-filter:v", "setpts=3.0*PTS,fps=24",
                                    "-pix_fmt", "yuv420p", "-c:v", "libx264",
                                    "-preset", "fast", "-movflags", "+faststart",
                                    tmp_out_path,
                                ],
                                check=True, capture_output=True, timeout=120,
                            )
                            conversion_success = True
                            print(f"✅ Converted {len(frame_files)} WebP frames → MP4")
                    except Exception as e:
                        print(f"⚠️ webpmux method failed: {e}")

                # Метод 2: ImageMagick — конвертация через PNG кадры
                if shutil.which("convert") and not conversion_success:
                    try:
                        frame_pattern_png = os.path.join(tmp_frames_dir, "frame_%04d.png")
                        subprocess.run(
                            ["convert", "-coalesce", tmp_in_path, frame_pattern_png],
                            check=True, capture_output=True, timeout=60,
                        )
                        frame_files = sorted(glob.glob(os.path.join(tmp_frames_dir, "frame_*.png")))
                        if frame_files:
                            subprocess.run(
                                [
                                    "ffmpeg", "-y", "-framerate", "8",
                                    "-i", os.path.join(tmp_frames_dir, "frame_%04d.png"),
                                    "-filter:v", "setpts=3.0*PTS,fps=24",
                                    "-pix_fmt", "yuv420p", "-c:v", "libx264",
                                    "-preset", "fast", "-movflags", "+faststart",
                                    tmp_out_path,
                                ],
                                check=True, capture_output=True, timeout=120,
                            )
                            conversion_success = True
                            print(f"✅ Converted {len(frame_files)} PNG frames → MP4")
                    except Exception as e:
                        print(f"⚠️ ImageMagick method failed: {e}")

                if not conversion_success:
                    print("⚠️ All conversion methods failed, serving original WebP")
                    final_filename = os.path.splitext(final_filename)[0] + ".webp"

            else:
                # Не-WebP видео — прямая конвертация
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", tmp_in_path,
                            "-filter:v", "setpts=3.0*PTS",
                            "-pix_fmt", "yuv420p", "-c:v", "libx264",
                            "-preset", "fast", "-movflags", "+faststart",
                            tmp_out_path,
                        ],
                        check=True, capture_output=True, timeout=120,
                    )
                    conversion_success = True
                except subprocess.CalledProcessError as e:
                    err_msg = e.stderr.decode() if e.stderr else str(e)
                    print(f"❌ FFmpeg conversion failed: {err_msg}")
                except FileNotFoundError:
                    print("❌ FFmpeg not found, serving original file")

            if conversion_success and os.path.exists(tmp_out_path):
                final_path = tmp_out_path
                final_filename = os.path.splitext(final_filename)[0] + ".mp4"

        def cleanup():
            for path in [tmp_in_path, tmp_out_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            if tmp_frames_dir and os.path.exists(tmp_frames_dir):
                try:
                    shutil.rmtree(tmp_frames_dir)
                except OSError:
                    pass

        return FileResponse(
            final_path,
            filename=final_filename,
            media_type="video/mp4" if final_path.endswith(".mp4") else guess_mime(final_path),
            background=BackgroundTask(cleanup),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(404, f"Download failed: {e}")


# ── Streaming proxy (публичный) ───────────────────────────────────────────────

@router.get("/{key:path}")
def get_media(key: str, request: Request):
    """
    Публичный S3-прокси с поддержкой HTTP Range (для видео).
    <img> и <video> не могут отправить Authorization header,
    поэтому эндпоинт намеренно открытый.
    """
    require_runpod_env()
    s3 = make_s3_client()
    bucket = RUNPOD_VOLUME_ID

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise HTTPException(404, f"Not found: {key}. {e}")

    size = int(head.get("ContentLength", 0))

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
        obj = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")

        def iter_ranged():
            yield from obj["Body"].iter_chunks(chunk_size=1024 * 1024)

        return StreamingResponse(
            iter_ranged(),
            status_code=206,
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(end - start + 1),
            },
        )

    obj = s3.get_object(Bucket=bucket, Key=key)

    def iter_full():
        yield from obj["Body"].iter_chunks(chunk_size=1024 * 1024)

    headers = {"Accept-Ranges": "bytes"}
    if size:
        headers["Content-Length"] = str(size)

    return StreamingResponse(iter_full(), media_type=content_type, headers=headers)
