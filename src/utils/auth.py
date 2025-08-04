"""Enhanced authentication and authorization system for production deployment.

This module provides comprehensive security features including:
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- API key management with scopes
- Session management with Redis
- Multi-factor authentication support
- Audit logging for security events
"""

import hashlib
import json
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import jwt
import redis.asyncio as redis
from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from src.utils.settings import settings


class AuthScope(str, Enum):
    """Authentication scopes for API access control."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    EMBED = "embed"
    CHAT = "chat"
    DELETE = "delete"
    METRICS = "metrics"


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"


class TokenType(str, Enum):
    """Types of authentication tokens."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SESSION = "session"


class AuthConfig(BaseModel):
    """Authentication configuration."""

    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_length: int = 32
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    enable_mfa: bool = False
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_number: bool = True
    password_require_uppercase: bool = True
    audit_log_retention_days: int = 90


class TokenData(BaseModel):
    """Token payload data."""

    sub: str  # Subject (user ID or API key ID)
    type: TokenType
    scopes: list[AuthScope] = []
    role: UserRole | None = None
    exp: int | None = None
    iat: int | None = None
    jti: str | None = None  # JWT ID for revocation
    session_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


class User(BaseModel):
    """User model for authentication."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str | None = None
    role: UserRole = UserRole.USER
    scopes: list[AuthScope] = []
    is_active: bool = True
    is_locked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_login: datetime | None = None
    failed_attempts: int = 0
    mfa_enabled: bool = False
    mfa_secret: str | None = None


class APIKey(BaseModel):
    """API key model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key_hash: str
    name: str
    user_id: str
    scopes: list[AuthScope] = []
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime | None = None
    expires_at: datetime | None = None
    usage_count: int = 0
    rate_limit: int | None = None


class AuditLog(BaseModel):
    """Security audit log entry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    user_id: str | None = None
    api_key_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    resource: str | None = None
    action: str
    result: str
    details: dict[str, Any] = {}


class AuthenticationError(HTTPException):
    """Custom authentication exception."""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization exception."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


class AuthManager:
    """Comprehensive authentication and authorization manager.
    
    Handles JWT tokens, API keys, sessions, and security policies.
    """

    def __init__(self, config: AuthConfig | None = None):
        self.config = config or AuthConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.redis_client: redis.Redis | None = None
        self.security = HTTPBearer()

    async def initialize(self):
        """Initialize authentication system with Redis connection."""
        if not self.redis_client:
            self.redis_client = await redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        self._validate_password_strength(password)
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def _validate_password_strength(self, password: str):
        """Validate password meets security requirements."""
        if len(password) < self.config.password_min_length:
            raise ValueError(f"Password must be at least {self.config.password_min_length} characters")

        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            raise ValueError("Password must contain special characters")

        if self.config.password_require_number and not any(c.isdigit() for c in password):
            raise ValueError("Password must contain numbers")

        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            raise ValueError("Password must contain uppercase letters")

    def create_access_token(
        self,
        data: TokenData,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create JWT access token."""
        to_encode = data.dict()

        # Set expiration
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=self.config.access_token_expire_minutes)

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(UTC),
            "jti": str(uuid.uuid4()),
        })

        # Create token
        encoded_jwt = jwt.encode(
            to_encode,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm,
        )

        return encoded_jwt

    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token for token renewal."""
        data = TokenData(
            sub=user_id,
            type=TokenType.REFRESH,
            jti=str(uuid.uuid4()),
        )

        expires_delta = timedelta(days=self.config.refresh_token_expire_days)
        return self.create_access_token(data, expires_delta)

    async def decode_token(self, token: str) -> TokenData:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
            )

            # Check if token is revoked
            jti = payload.get("jti")
            if jti and await self._is_token_revoked(jti):
                raise AuthenticationError("Token has been revoked")

            return TokenData(**payload)

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")

    async def revoke_token(self, jti: str):
        """Revoke a token by its JWT ID."""
        if self.redis_client:
            # Store revoked token ID with expiration
            await self.redis_client.setex(
                f"revoked_token:{jti}",
                86400 * 7,  # Keep for 7 days
                "1",
            )

    async def _is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        if self.redis_client:
            return bool(await self.redis_client.get(f"revoked_token:{jti}"))
        return False

    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and its hash."""
        key = secrets.token_urlsafe(self.config.api_key_length)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash

    async def validate_api_key(self, api_key: str) -> APIKey | None:
        """Validate API key and return key data."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Look up key in Redis cache first
        if self.redis_client:
            cached = await self.redis_client.get(f"api_key:{key_hash}")
            if cached:
                return APIKey(**json.loads(cached))

        # Would normally look up in database here
        # For now, return None (key not found)
        return None

    async def create_session(
        self,
        user_id: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> str:
        """Create user session."""
        session_id = str(uuid.uuid4())

        session_data = {
            "user_id": user_id,
            "created_at": datetime.now(UTC).isoformat(),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "last_activity": datetime.now(UTC).isoformat(),
        }

        if self.redis_client:
            await self.redis_client.setex(
                f"session:{session_id}",
                self.config.session_timeout_minutes * 60,
                json.dumps(session_data),
            )

        return session_id

    async def validate_session(self, session_id: str) -> dict[str, Any] | None:
        """Validate and refresh session."""
        if self.redis_client:
            session_data = await self.redis_client.get(f"session:{session_id}")
            if session_data:
                data = json.loads(session_data)

                # Update last activity
                data["last_activity"] = datetime.now(UTC).isoformat()

                # Refresh session TTL
                await self.redis_client.setex(
                    f"session:{session_id}",
                    self.config.session_timeout_minutes * 60,
                    json.dumps(data),
                )

                return data

        return None

    async def destroy_session(self, session_id: str):
        """Destroy user session."""
        if self.redis_client:
            await self.redis_client.delete(f"session:{session_id}")

    async def log_security_event(
        self,
        event_type: str,
        action: str,
        result: str,
        user_id: str | None = None,
        api_key_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Log security audit event."""
        audit_log = AuditLog(
            event_type=event_type,
            action=action,
            result=result,
            user_id=user_id,
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            details=details or {},
        )

        # Store in Redis with expiration
        if self.redis_client:
            await self.redis_client.setex(
                f"audit_log:{audit_log.id}",
                self.config.audit_log_retention_days * 86400,
                audit_log.json(),
            )

            # Add to audit log list
            await self.redis_client.lpush(
                "audit_logs",
                audit_log.id,
            )
            await self.redis_client.ltrim("audit_logs", 0, 10000)  # Keep last 10k entries

    def check_scopes(self, required_scopes: list[AuthScope], user_scopes: list[AuthScope]) -> bool:
        """Check if user has required scopes."""
        return all(scope in user_scopes for scope in required_scopes)

    def check_role_permissions(self, role: UserRole, required_scopes: list[AuthScope]) -> bool:
        """Check if role has permissions for required scopes."""
        role_permissions = {
            UserRole.ADMIN: [AuthScope.READ, AuthScope.WRITE, AuthScope.ADMIN, AuthScope.EMBED,
                           AuthScope.CHAT, AuthScope.DELETE, AuthScope.METRICS],
            UserRole.USER: [AuthScope.READ, AuthScope.WRITE, AuthScope.EMBED, AuthScope.CHAT],
            UserRole.SERVICE: [AuthScope.READ, AuthScope.WRITE, AuthScope.EMBED, AuthScope.CHAT,
                             AuthScope.METRICS],
            UserRole.READONLY: [AuthScope.READ],
        }

        user_scopes = role_permissions.get(role, [])
        return self.check_scopes(required_scopes, user_scopes)

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
        request: Request = None,
    ) -> TokenData:
        """Get current user from bearer token."""
        if not credentials:
            raise AuthenticationError("Missing authentication credentials")

        token_data = await self.decode_token(credentials.credentials)

        # Log access
        await self.log_security_event(
            event_type="auth",
            action="token_validation",
            result="success",
            user_id=token_data.sub,
            ip_address=request.client.host if request else None,
            user_agent=request.headers.get("user-agent") if request else None,
        )

        return token_data

    def require_scopes(self, scopes: list[AuthScope]):
        """Dependency to require specific scopes."""
        async def scope_checker(
            token_data: TokenData = Security(self.get_current_user),
        ):
            if not self.check_scopes(scopes, token_data.scopes):
                await self.log_security_event(
                    event_type="authz",
                    action="scope_check",
                    result="denied",
                    user_id=token_data.sub,
                    details={"required_scopes": scopes, "user_scopes": token_data.scopes},
                )
                raise AuthorizationError(f"Required scopes: {', '.join(scopes)}")
            return token_data

        return scope_checker

    def require_role(self, role: UserRole):
        """Dependency to require specific role."""
        async def role_checker(
            token_data: TokenData = Security(self.get_current_user),
        ):
            if token_data.role != role:
                await self.log_security_event(
                    event_type="authz",
                    action="role_check",
                    result="denied",
                    user_id=token_data.sub,
                    details={"required_role": role, "user_role": token_data.role},
                )
                raise AuthorizationError(f"Required role: {role}")
            return token_data

        return role_checker


# Global auth manager instance
auth_manager = AuthManager()
