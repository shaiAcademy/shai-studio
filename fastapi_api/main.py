import asyncio
import os
import time
import copy
import random
import json
import logging
from typing import Optional
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import requests

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

# Получаем URL ComfyUI из переменной окружения
COMFY_URL = os.getenv("RUNPOD_COMFY_URL")

app = FastAPI(title="shai.academy API", version="1.0.0")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://shai:shai@postgres:5432/shai_db"
)

# Создаем engine с пулом соединений и настройками для retry
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Проверяет соединение перед использованием
    pool_recycle=300,    # Переиспользует соединения
    connect_args={"connect_timeout": 10}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    prompt = Column(Text)
    user_id = Column(String, nullable=True)
    type = Column(String)  # 'image' or 'video'
    status = Column(String)
    result_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# Функция для инициализации БД с retry логикой
def init_db():
    """Инициализирует БД с повторными попытками подключения"""
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database initialized successfully")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to initialize database after {max_retries} attempts: {str(e)}")
                logger.warning("Application will continue, but database operations may fail")


# Инициализируем БД при старте (с retry)
init_db()


def get_comfyui_image_url(prompt_id: str, max_wait: int = 120) -> Optional[str]:
    """
    Опрашивает ComfyUI API для получения URL сгенерированного изображения.
    
    Args:
        prompt_id: ID задачи от ComfyUI
        max_wait: Максимальное время ожидания в секундах
    
    Returns:
        URL изображения или None, если не удалось получить
    """
    start_time = time.time()
    poll_interval = 2  # Опрашиваем каждые 2 секунды
    
    logger.info(f"Starting to poll for prompt_id: {prompt_id}")
    
    while time.time() - start_time < max_wait:
        try:
            # Проверяем историю выполнения
            history_url = f"{COMFY_URL.rstrip('/')}/history/{prompt_id}"
            response = requests.get(history_url, timeout=10)
            
            if response.status_code == 200:
                history_data = response.json()
                logger.debug(f"History response: {json.dumps(history_data, indent=2)}")
                
                # ComfyUI может возвращать историю в разных форматах
                # Формат 1: {prompt_id: {status: [...], outputs: {...}}}
                # Формат 2: {prompt_id: [{status: {...}, outputs: {...}}]}
                
                task_data = None
                if prompt_id in history_data:
                    task_data = history_data[prompt_id]
                elif isinstance(history_data, dict):
                    # Может быть вложенная структура
                    for key, value in history_data.items():
                        if key == prompt_id or (isinstance(value, dict) and prompt_id in str(value)):
                            task_data = value
                            break
                
                if task_data:
                    # Если task_data - список, берем последний элемент
                    if isinstance(task_data, list) and len(task_data) > 0:
                        task_data = task_data[-1]
                    
                    # Проверяем, есть ли выходные данные (outputs)
                    if isinstance(task_data, dict) and "outputs" in task_data:
                        outputs = task_data["outputs"]
                        
                        # Ищем узел SaveImage (обычно это узел "9")
                        for node_id, node_output in outputs.items():
                            if isinstance(node_output, dict) and "images" in node_output:
                                images = node_output["images"]
                                if images and len(images) > 0:
                                    # Берем первое изображение
                                    image_info = images[0]
                                    filename = image_info.get("filename", "")
                                    subfolder = image_info.get("subfolder", "")
                                    image_type = image_info.get("type", "output")
                                    
                                    if filename:
                                        # Формируем URL для получения изображения
                                        # ComfyUI обычно хранит изображения в /view endpoint
                                        view_url = f"{COMFY_URL.rstrip('/')}/view"
                                        params = {}
                                        if filename:
                                            params["filename"] = filename
                                        if subfolder:
                                            params["subfolder"] = subfolder
                                        params["type"] = image_type
                                        
                                        # Строим полный URL
                                        image_url = f"{view_url}?{urlencode(params)}"
                                        
                                        logger.info(f"Found image URL: {image_url}")
                                        return image_url
                    
                    # Если outputs нет, но есть status, проверяем статус
                    if isinstance(task_data, dict) and "status" in task_data:
                        status = task_data["status"]
                        if isinstance(status, list) and len(status) > 0:
                            # Берем последний статус
                            last_status = status[-1]
                            if isinstance(last_status, dict) and last_status.get("completed", False):
                                # Задача завершена, но outputs может быть в другом месте
                                logger.debug("Task completed, but outputs not found yet")
            
            # Если задача еще не завершена, ждем и пробуем снова
            logger.debug(f"Task {prompt_id} still processing, waiting {poll_interval}s...")
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error polling ComfyUI: {str(e)}, retrying...")
            time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Unexpected error while polling: {str(e)}", exc_info=True)
            time.sleep(poll_interval)
    
    logger.warning(f"Timeout waiting for prompt_id {prompt_id} after {max_wait}s")
    return None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt for generation")
    steps: int = Field(20, ge=1, le=100, description="Number of sampling steps")
    user_id: Optional[str] = Field(
        None, description="Optional user identifier for tracking"
    )


class GenerateResponse(BaseModel):
    task_id: str
    status: str
    image_url: str


class GenerateVideoResponse(BaseModel):
    task_id: str
    status: str
    message: str
    video_url: str


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TaskResponse(BaseModel):
    id: int
    task_id: str
    prompt: str
    user_id: Optional[str]
    type: str
    status: str
    result_url: str
    created_at: datetime


# Шаблон ComfyUI workflow (обновленный по новому JSON)
COMFY_WORKFLOW_TEMPLATE = {
    "3": {
        "inputs": {
            "seed": 891982008105110,  # будет переопределен на случайный
            "steps": 30,  # заменяется на steps пользователя
            "cfg": 7,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1,
            "model": ["10", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
    },
    "4": {
        "inputs": {
            "ckpt_name": "dreamshaper_8.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "5": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
    },
    "6": {
        "inputs": {
            "text": "masterpiece, best quality, ultra-detailed, 8K, RAW photo, intricate details, stunning visuals,upper-body, highly detailed face, smooth skin, white colthes, realistic lighting, beautiful Chinese girl, solo, traditional Chinese dress, golden embroidery, elegant, black hair, delicate hair ornament, cinematic lighting, soft focus,(white background:1.05)",  # заменяется на prompt пользователя
            "clip": ["10", 1]
        },
        "class_type": "CLIPTextEncode"
    },
    "7": {
        "inputs": {
            "text": "(low quality, worst quality:1.4), (blurry:1.2), (bad anatomy:1.3), extra limbs, deformed, watermark, text, signature, bareness",
            "clip": ["10", 1]
        },
        "class_type": "CLIPTextEncode"
    },
    "8": {
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        },
        "class_type": "VAEDecode"
    },
    "9": {
        "inputs": {
            "filename_prefix": "2loras_test_",
            "images": ["8", 0]
        },
        "class_type": "SaveImage"
    },
    "10": {
        "inputs": {
            "model": ["11", 0],
            "clip": ["11", 1]
        },
        "class_type": "LoraLoader",
        "properties": {
            "models": [
                {
                    "name": "MoXinV1.safetensors",
                    "url": "https://civitai.com/api/download/models/14856?type=Model&format=SafeTensor&size=full&fp=fp16",
                    "directory": "loras"
                }
            ]
        },
        "widgets_values": [
            "add-detail-xl.safetensors",
            1,
            1
        ]
    },
    "11": {
        "inputs": {
            "model": ["4", 0],
            "clip": ["4", 1]
        },
        "class_type": "LoraLoader",
        "properties": {
            "models": [
                {
                    "name": "blindbox_v1_mix.safetensors",
                    "url": "https://civitai.com/api/download/models/32988?type=Model&format=SafeTensor&size=full&fp=fp16",
                    "directory": "loras"
                }
            ]
        },
        "widgets_values": [
            "blindbox_v1_mix.safetensors",
            0.9,
            1
        ]
    }
}


@app.post("/api/generate/image", response_model=GenerateResponse)
async def generate_image(payload: GenerateRequest):
    if not COMFY_URL:
        raise HTTPException(
            status_code=500,
            detail="COMFY_URL not configured. Please set RUNPOD_COMFY_URL environment variable."
        )
    
    # Создаем копию шаблона workflow
    workflow = copy.deepcopy(COMFY_WORKFLOW_TEMPLATE)
    
    # Вставляем данные пользователя в workflow
    workflow["3"]["inputs"]["steps"] = payload.steps
    workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)  # Случайный seed для разнообразия
    workflow["6"]["inputs"]["text"] = payload.prompt
    
    # Подготавливаем данные для отправки в ComfyUI
    # ComfyUI API требует client_id для отслеживания сессий
    client_id = f"shai_academy_{int(time.time())}"
    prompt_data = {
        "prompt": workflow,
        "client_id": client_id
    }
    
    # Логируем данные для отладки
    logger.info(f"Sending request to ComfyUI: {COMFY_URL}")
    logger.info(f"Client ID: {client_id}")
    logger.debug(f"Workflow data: {json.dumps(workflow, indent=2)}")
    
    # Отправляем POST-запрос на ComfyUI API
    try:
        # Пробуем разные варианты URL
        comfy_urls = [
            f"{COMFY_URL.rstrip('/')}/prompt",
            f"{COMFY_URL.rstrip('/')}/api/v1/prompt",
        ]
        
        last_error = None
        for comfy_url in comfy_urls:
            try:
                logger.info(f"Trying URL: {comfy_url}")
                response = requests.post(
                    comfy_url,
                    json=prompt_data,
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Увеличиваем таймаут для генерации изображений
                )
                
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                # Если получили успешный ответ, выходим из цикла
                if response.status_code == 200:
                    break
                    
                # Если это не последний URL, пробуем следующий
                if comfy_url != comfy_urls[-1]:
                    logger.warning(f"URL {comfy_url} returned status {response.status_code}, trying next...")
                    continue
                
                # Если это последний URL и статус не 200, поднимаем ошибку
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Error with URL {comfy_url}: {str(e)}")
                if comfy_url == comfy_urls[-1]:
                    raise
                continue
        
        # Если все URL не сработали
        if last_error:
            raise last_error
        
        # Получаем ответ от ComfyUI
        try:
            comfy_response = response.json()
            logger.info(f"ComfyUI response: {json.dumps(comfy_response, indent=2)}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response: {response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"ComfyUI returned invalid JSON: {response.text[:500]}"
            )
        
        # Генерируем task_id (можно использовать из ответа ComfyUI, если есть)
        task_id = comfy_response.get("prompt_id", f"comfy_task_{int(time.time())}")
        
        logger.info(f"ComfyUI accepted prompt with ID: {task_id}")
        
        # Опрашиваем ComfyUI для получения результата
        logger.info("Polling ComfyUI for result...")
        image_url = get_comfyui_image_url(task_id, max_wait=120)
        
        # Сохраняем задачу в БД
        db = SessionLocal()
        try:
            if image_url:
                status = "completed"
                result_url = image_url
                logger.info(f"Image generated successfully: {image_url}")
            else:
                status = "processing"
                result_url = ""
                logger.warning(f"Image not ready yet for task_id: {task_id}")
            
            task = Task(
                task_id=task_id,
                prompt=payload.prompt,
                user_id=payload.user_id,
                type="image",
                status=status,
                result_url=result_url,
            )
            db.add(task)
            db.commit()
        finally:
            db.close()
        
        # Возвращаем ответ
        if image_url:
            # Если получили изображение, возвращаем его URL
            return GenerateResponse(
                task_id=task_id,
                status="completed",
                image_url=image_url
            )
        else:
            # Если изображение еще не готово, возвращаем placeholder
            # В реальном сценарии можно добавить отдельный endpoint для опроса статуса
            placeholder_url = f"https://placehold.co/512x512/3b82f6/ffffff/png?text=Processing+{task_id[:8]}"
            return GenerateResponse(
                task_id=task_id,
                status="processing",
                image_url=placeholder_url  # Placeholder пока изображение генерируется
            )
        
    except requests.exceptions.Timeout:
        logger.error("Timeout connecting to ComfyUI")
        raise HTTPException(
            status_code=502,
            detail="ComfyUI request timed out. Please check if the service is running."
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to ComfyUI at {COMFY_URL}. Please check the URL and ensure the service is running."
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        error_detail = f"ComfyUI returned status {e.response.status_code}"
        try:
            error_json = e.response.json()
            if "error" in error_json:
                error_detail += f": {error_json['error']}"
            elif "message" in error_json:
                error_detail += f": {error_json['message']}"
        except:
            error_detail += f": {e.response.text[:500]}"
        raise HTTPException(
            status_code=502,
            detail=error_detail
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to ComfyUI: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/generate/video", response_model=GenerateVideoResponse)
async def generate_video(payload: GenerateRequest):
    await asyncio.sleep(10)
    task_id = f"mock_video_{int(time.time())}"
    video_url = "https://placehold.co/600x400/cc0000/ffffff/png?text=Video+MOCK"

    db = SessionLocal()
    try:
        task = Task(
            task_id=task_id,
            prompt=payload.prompt,
            user_id=payload.user_id,
            type="video",
            status="completed",
            result_url=video_url,
        )
        db.add(task)
        db.commit()
    finally:
        db.close()

    return GenerateVideoResponse(
        task_id=task_id,
        status="completed",
        message="Video generation mock completed.",
        video_url=video_url,
    )


@app.get("/api/tasks", response_model=list[TaskResponse])
async def get_tasks():
    db = SessionLocal()
    try:
        tasks = db.query(Task).order_by(Task.id.desc()).limit(50).all()
        return tasks
    finally:
        db.close()


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(auth: AuthRequest):
    return AuthResponse(access_token="mock_jwt_token")


@app.get("/api/tasks/{task_id}/status", response_model=GenerateResponse)
async def get_task_status(task_id: str):
    """
    Опрашивает статус задачи и возвращает URL изображения, если оно готово.
    """
    if not COMFY_URL:
        raise HTTPException(
            status_code=500,
            detail="COMFY_URL not configured. Please set RUNPOD_COMFY_URL environment variable."
        )
    
    # Проверяем БД
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Если задача уже завершена и есть URL, возвращаем его
        if task.status == "completed" and task.result_url:
            return GenerateResponse(
                task_id=task.task_id,
                status="completed",
                image_url=task.result_url
            )
        
        # Если задача еще обрабатывается, опрашиваем ComfyUI
        if task.status == "processing":
            image_url = get_comfyui_image_url(task_id, max_wait=5)  # Короткий опрос
            
            if image_url:
                # Обновляем задачу в БД
                task.status = "completed"
                task.result_url = image_url
                db.commit()
                
                return GenerateResponse(
                    task_id=task.task_id,
                    status="completed",
                    image_url=image_url
                )
            else:
                # Все еще обрабатывается
                placeholder_url = f"https://placehold.co/512x512/3b82f6/ffffff/png?text=Processing+{task_id[:8]}"
                return GenerateResponse(
                    task_id=task.task_id,
                    status="processing",
                    image_url=placeholder_url
                )
        
        # Если задача завершена, но нет URL (не должно быть, но на всякий случай)
        placeholder_url = f"https://placehold.co/512x512/3b82f6/ffffff/png?text=Processing+{task_id[:8]}"
        return GenerateResponse(
            task_id=task.task_id,
            status=task.status,
            image_url=placeholder_url
        )
        
    finally:
        db.close()


