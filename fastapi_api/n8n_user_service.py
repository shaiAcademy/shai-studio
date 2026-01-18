"""
n8n User Service Module

Provides functionality to create and manage users in n8n's SQLite database.
This enables SSO-like functionality from studio.shai.academy to n8n.shai.academy.
"""

import os
import sqlite3
import secrets
from datetime import datetime
from typing import Optional

import bcrypt

# Path to n8n SQLite database
N8N_DB_PATH = os.getenv("N8N_DB_PATH", "/var/lib/docker/volumes/n8n_n8n_data/_data/database.sqlite")


def create_or_get_n8n_user(email: str, first_name: str = "") -> dict:
    """
    Create a new user in n8n database or return existing user info.
    
    Args:
        email: User's email address
        first_name: User's first name (optional, defaults to email prefix)
    
    Returns:
        dict with keys: userId, email, firstName, globalRole, exists
    
    Raises:
        Exception if database is not accessible
    """
    
    # Check if the database file exists
    if not os.path.exists(N8N_DB_PATH):
        raise FileNotFoundError(f"n8n database not found at {N8N_DB_PATH}")
    
    conn = sqlite3.connect(N8N_DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if user already exists
        cursor.execute(
            'SELECT id, email, firstName, globalRole FROM "user" WHERE email = ?',
            (email,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                "userId": row[0],
                "email": row[1],
                "firstName": row[2] or "",
                "globalRole": row[3],
                "exists": True
            }
        
        # Create new user
        user_id = secrets.token_hex(16)
        api_key = secrets.token_hex(20)
        
        # Generate random password and hash it with bcrypt
        random_password = secrets.token_hex(32)
        hashed_password = bcrypt.hashpw(
            random_password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
        
        # Use email prefix as first name if not provided
        user_first_name = first_name or email.split('@')[0]
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute('''
            INSERT INTO "user" (
                id, email, firstName, lastName, password, 
                role, globalRole, apiKey, 
                personalizationAnswers, settings,
                createdAt, updatedAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            email,
            user_first_name,
            '',  # lastName
            hashed_password,
            'global:member',  # role
            'global:member',  # globalRole
            api_key,
            '{}',  # personalizationAnswers
            None,  # settings
            now,
            now
        ))
        
        conn.commit()
        
        print(f"✅ Created n8n user: {email} (id: {user_id})")
        
        return {
            "userId": user_id,
            "email": email,
            "firstName": user_first_name,
            "globalRole": "global:member",
            "exists": False
        }
        
    except sqlite3.Error as e:
        conn.rollback()
        print(f"❌ SQLite error creating n8n user: {e}")
        raise Exception(f"Database error: {e}")
        
    finally:
        conn.close()


def get_n8n_user_by_email(email: str) -> Optional[dict]:
    """
    Get n8n user by email address.
    
    Args:
        email: User's email address
    
    Returns:
        dict with user info or None if not found
    """
    
    if not os.path.exists(N8N_DB_PATH):
        return None
    
    conn = sqlite3.connect(N8N_DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'SELECT id, email, firstName, globalRole FROM "user" WHERE email = ?',
            (email,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                "userId": row[0],
                "email": row[1],
                "firstName": row[2] or "",
                "globalRole": row[3]
            }
        
        return None
        
    finally:
        conn.close()
