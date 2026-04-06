"""
auth.py — JWT / password хелперы + роутер /api/auth.
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET
from app.database import get_db
from app.models import User
from app.schemas import AuthResponse, MeResponse, UserCreate, UserLogin

import firebase_admin
from firebase_admin import auth as firebase_auth

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── Password / JWT ────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def create_access_token(subject: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    return jwt.encode({"sub": subject, "exp": exp}, JWT_SECRET, algorithm=JWT_ALGORITHM)


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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(payload: UserCreate, db=Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user in Firebase
    try:
        firebase_auth.create_user(
            email=payload.email,
            password=payload.password,
            display_name=payload.name,
        )
        print(f"✅ Created user {payload.email} in Firebase.")
    except firebase_auth.EmailAlreadyExistsError:
        print(f"⚠️ User {payload.email} already exists in Firebase. Proceeding to local DB.")
    except Exception as e:
        print(f"❌ Failed to create user in Firebase: {e}")
        raise HTTPException(status_code=400, detail=f"Firebase Registration Failed: {e}")

    # Create user in local DB
    user = User(
        email=payload.email,
        name=payload.name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return AuthResponse(access_token=create_access_token(str(user.id)))


@router.post("/login", response_model=AuthResponse)
def login(payload: UserLogin, db=Depends(get_db)):
    # Verify in Firebase
    try:
        firebase_user = firebase_auth.get_user_by_email(payload.email)
        print(f"✅ Found Firebase user: {firebase_user.uid}")
    except firebase_auth.UserNotFoundError:
        print(f"❌ User {payload.email} not found in Firebase.")
        raise HTTPException(status_code=401, detail="Account not found in Firebase system.")
    except ValueError as e:
        print(f"❌ Firebase Auth not initialized: {e}")
        raise HTTPException(status_code=500, detail="Firebase Configuration Error: Service not initialized on server.")
    except Exception as e:
        print(f"❌ Firebase check error: {e}")
        raise HTTPException(status_code=500, detail=f"Firebase Connectivity Error: {str(e)}")

    # Verify in local DB
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials (local check)")

    return AuthResponse(access_token=create_access_token(str(user.id)))


@router.get("/me", response_model=MeResponse)
def me(current_user: User = Depends(get_current_user)):
    return MeResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at,
    )
