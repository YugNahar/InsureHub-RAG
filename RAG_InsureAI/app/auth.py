"""
auth.py — Drop this file into your /app folder.
Then add 3 lines to api.py (shown at bottom of this file).

Requirements (add to requirements.txt):
    python-jose[cryptography]
    passlib[bcrypt]
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config from .env ──────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("AUTH_TOKEN_EXPIRE_MINUTES") or "480") # 8 hours

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")  # bcrypt hash
ADMIN_PASSWORD_PLAIN = os.getenv("ADMIN_PASSWORD", "")       # fallback plain (dev only)
AGENT_PASSWORD_PLAIN = os.getenv("AGENT_PASSWORD", "") or ADMIN_PASSWORD_PLAIN

# Temporary escape hatch: set DISABLE_AUTH=true to skip login entirely.
# Unset this (or set to false) before exposing the app beyond trusted testers.
DISABLE_AUTH = os.getenv("DISABLE_AUTH", "false").strip().lower() in ("1", "true", "yes")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _verify_password(plain: str) -> bool:
    """Accept bcrypt hash from env OR plain password (dev fallback)."""
    if ADMIN_PASSWORD_HASH:
        return pwd_context.verify(plain, ADMIN_PASSWORD_HASH)
    if ADMIN_PASSWORD_PLAIN:
        return plain == ADMIN_PASSWORD_PLAIN
    return False


def _create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ── Dependency: require valid JWT ─────────────────────────────────────────────
def require_auth(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """FastAPI dependency — raise 401 if token is missing or invalid."""
    if DISABLE_AUTH:
        return ADMIN_USERNAME
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in at /auth",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise JWTError()
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again at /auth",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


# ── Login request/response models ────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


# ── Login endpoint (attach to app in api.py) ──────────────────────────────────
def create_login_endpoint(app):
    """Call this in api.py: create_login_endpoint(app)"""

    @app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
    async def login(req: LoginRequest):
        """Exchange username + password for a JWT token."""
        if req.username != ADMIN_USERNAME or not _verify_password(req.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password.",
            )
        token = _create_access_token({"sub": req.username})
        return TokenResponse(
            access_token=token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    @app.post("/auth/agent-login", tags=["auth"])
    async def agent_login(req: LoginRequest):
        """Validate an agent password. Uses AGENT_PASSWORD (falls back to ADMIN_PASSWORD)."""
        if req.password != AGENT_PASSWORD_PLAIN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect agent password.",
            )
        return {"ok": True}

    @app.get("/auth/verify", tags=["auth"])
    async def verify_token(username: str = Depends(require_auth)):
        """Frontend can call this to check if stored token is still valid."""
        return {"valid": True, "username": username}

    @app.get("/auth/status", tags=["auth"])
    async def auth_status():
        """Public: lets the frontend know whether login is currently required."""
        return {"disabled": DISABLE_AUTH}


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO WIRE THIS INTO api.py  (add these 3 things)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. At the top of api.py, after other imports:
#       from auth import create_login_endpoint, require_auth
#
# 2. After `app = FastAPI(...)`:
#       create_login_endpoint(app)
#
# 3. Add `require_auth` as a dependency to protected endpoints:
#
#    @app.post("/upload")
#    async def upload(file: UploadFile = File(...), _=Depends(require_auth)):
#
#    @app.post("/upload-video")
#    async def upload_video(req: URLRequest, _=Depends(require_auth)):
#
#    @app.post("/upload-webpage")
#    async def upload_webpage(req: URLRequest, _=Depends(require_auth)):
#
#    @app.delete("/docs")
#    async def clear_docs(_=Depends(require_auth)):
#
#    @app.delete("/docs/{name:path}")
#    async def remove_doc(name: str, _=Depends(require_auth)):
#
#    @app.delete("/videos/{url:path}")
#    async def delete_video(url: str, _=Depends(require_auth)):
#
#    @app.delete("/webpages/{url:path}")
#    async def delete_webpage(url: str, _=Depends(require_auth)):
# ─────────────────────────────────────────────────────────────────────────────
