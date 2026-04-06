"""
n8n_user_service.py — создание/поиск пользователей в SQLite базе n8n.

Обеспечивает SSO-подобное поведение: пользователи studio.shai.academy
автоматически создаются в n8n.shai.academy.
"""
import os
import secrets
import sqlite3
from datetime import datetime
from typing import Optional

import bcrypt

# Путь к SQLite базе n8n (монтируется через docker volume)
N8N_DB_PATH: str = os.getenv(
    "N8N_DB_PATH",
    "/var/lib/docker/volumes/n8n_n8n_data/_data/database.sqlite",
)


def create_or_get_n8n_user(email: str, first_name: str = "") -> dict:
    """
    Создаёт пользователя в БД n8n или возвращает существующего.

    Returns:
        dict: {userId, email, firstName, globalRole, exists}

    Raises:
        FileNotFoundError: если база n8n недоступна
        Exception: при ошибке БД
    """
    if not os.path.exists(N8N_DB_PATH):
        raise FileNotFoundError(f"n8n database not found at {N8N_DB_PATH}")

    with sqlite3.connect(N8N_DB_PATH) as conn:
        cursor = conn.cursor()

        # Проверяем существование пользователя
        cursor.execute(
            'SELECT id, email, firstName FROM "user" WHERE email = ?',
            (email,),
        )
        row = cursor.fetchone()

        if row:
            return {
                "userId": row[0],
                "email": row[1],
                "firstName": row[2] or "",
                "globalRole": "global:member",
                "exists": True,
            }

        # Создаём нового пользователя
        user_id = secrets.token_hex(16)
        api_key = secrets.token_hex(20)

        # Генерируем случайный пароль (пользователь входит через SSO, не через пароль)
        random_password = secrets.token_hex(32)
        hashed_password = bcrypt.hashpw(
            random_password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

        user_first_name = first_name or email.split("@")[0]
        now = datetime.utcnow().isoformat()

        try:
            cursor.execute(
                """
                INSERT INTO "user" (
                    id, email, firstName, lastName, password,
                    apiKey,
                    personalizationAnswers, settings,
                    createdAt, updatedAt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, email, user_first_name, "",
                    hashed_password, api_key,
                    "{}", None,
                    now, now,
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            print(f"❌ SQLite error creating n8n user: {e}")
            raise Exception(f"Database error: {e}") from e

    print(f"✅ Created n8n user: {email} (id: {user_id})")
    return {
        "userId": user_id,
        "email": email,
        "firstName": user_first_name,
        "globalRole": "global:member",
        "exists": False,
    }


def get_n8n_user_by_email(email: str) -> Optional[dict]:
    """
    Ищет пользователя n8n по email.

    Returns:
        dict с данными пользователя или None если не найден.
    """
    if not os.path.exists(N8N_DB_PATH):
        return None

    with sqlite3.connect(N8N_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, email, firstName FROM "user" WHERE email = ?',
            (email,),
        )
        row = cursor.fetchone()

    if row:
        return {
            "userId": row[0],
            "email": row[1],
            "firstName": row[2] or "",
            "globalRole": "global:member",
        }
    return None
