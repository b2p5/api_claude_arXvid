"""
Authentication system for RAG API using JWT tokens.
Provides user registration, login, and token-based authentication.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from logger import get_logger, log_info, log_error

# Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
AUTH_DB_PATH = "users.db"

logger = get_logger()
security = HTTPBearer()


# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class User(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool
    created_at: datetime


class AuthManager:
    """Manages user authentication and database operations."""
    
    def __init__(self, db_path: str = AUTH_DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the users database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    )
                """)
                conn.commit()
                log_info("Users database initialized", db_path=self.db_path)
        except Exception as e:
            log_error("Failed to initialize users database", error=str(e))
            raise
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return hashlib.sha256(password.encode()).hexdigest() == hashed
    
    def create_user(self, user_data: UserCreate) -> Dict[str, Any]:
        """Create a new user."""
        try:
            password_hash = self.hash_password(user_data.password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (email, password_hash, full_name)
                    VALUES (?, ?, ?)
                """, (user_data.email, password_hash, user_data.full_name))
                
                user_id = cursor.lastrowid
                conn.commit()
                
                log_info("User created successfully", user_id=user_id, email=user_data.email)
                
                return {
                    "id": user_id,
                    "email": user_data.email,
                    "full_name": user_data.full_name,
                    "is_active": True
                }
                
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        except Exception as e:
            log_error("User creation failed", error=str(e), email=user_data.email)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User creation failed"
            )
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, email, password_hash, full_name, is_active
                    FROM users WHERE email = ? AND is_active = TRUE
                """, (email,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                user_id, user_email, password_hash, full_name, is_active = result
                
                if not self.verify_password(password, password_hash):
                    return None
                
                # Update last login
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (user_id,))
                conn.commit()
                
                log_info("User authenticated successfully", user_id=user_id, email=email)
                
                return {
                    "id": user_id,
                    "email": user_email,
                    "full_name": full_name,
                    "is_active": is_active
                }
                
        except Exception as e:
            log_error("Authentication failed", error=str(e), email=email)
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, email, full_name, is_active, created_at
                    FROM users WHERE id = ? AND is_active = TRUE
                """, (user_id,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                user_id, email, full_name, is_active, created_at = result
                
                return {
                    "id": user_id,
                    "email": email,
                    "full_name": full_name,
                    "is_active": is_active,
                    "created_at": created_at
                }
                
        except Exception as e:
            log_error("Failed to get user by ID", error=str(e), user_id=user_id)
            return None


class JWTManager:
    """Manages JWT token creation and validation."""
    
    @staticmethod
    def create_access_token(user_data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        payload = {
            "user_id": user_data["id"],
            "email": user_data["email"],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


# Global instances
auth_manager = AuthManager()
jwt_manager = JWTManager()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from JWT token.
    """
    token = credentials.credentials
    
    # Verify token and get payload
    payload = jwt_manager.verify_token(token)
    
    # Get user from database
    user = auth_manager.get_user_by_id(payload.get("user_id"))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user


def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency to get current active user.
    """
    if not current_user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user


# Authentication endpoints functions
async def register_user(user_data: UserCreate) -> Dict[str, Any]:
    """Register a new user."""
    user = auth_manager.create_user(user_data)
    token = jwt_manager.create_access_token(user)
    
    return {
        "user": user,
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600  # seconds
    }


async def login_user(login_data: UserLogin) -> Dict[str, Any]:
    """Login user and return JWT token."""
    user = auth_manager.authenticate_user(login_data.email, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    token = jwt_manager.create_access_token(user)
    
    return {
        "user": user,
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600  # seconds
    }


def get_username_from_user(user: Dict[str, Any]) -> str:
    """
    Extract normalized username from user data.
    Uses email as username base.
    """
    email = user.get("email", "")
    # Use the part before @ as username
    username = email.split("@")[0] if "@" in email else email
    return username.lower().replace(".", "_").replace("-", "_")