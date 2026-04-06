"""
s3_client.py — S3-совместимый клиент для RunPod Network Volume.
"""
import mimetypes
from typing import Optional, Tuple

import boto3
from botocore.client import Config

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    RUNPOD_S3_ENDPOINT_URL,
    RUNPOD_S3_REGION,
)


def make_s3_client():
    """Создаёт boto3 клиент с настройками RunPod S3."""
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
    """Разбирает заголовок Range: bytes=start-end."""
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
