"""密码哈希与 JWT 工具。

- 密码：bcrypt 哈希存储，verify 校验
- Token：PyJWT HS256，payload 内嵌用户身份 + 角色 + 权限（无状态，校验不依赖 DB）
"""
import time
import uuid
from typing import Any, Dict, List, Optional

import bcrypt
import jwt

from app.core.logger import get_logger

logger = get_logger(__name__)

# bcrypt 5.x 对超过 72 字节的密码会抛异常，这里统一截断避免异常
_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """bcrypt 哈希密码，返回可入库的字符串。"""
    pw = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """校验密码，失败（含格式异常）返回 False。"""
    try:
        pw = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
        return bcrypt.checkpw(pw, hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(
    user_id: str,
    username: str,
    display_name: str,
    tenant_id: str,
    roles: List[str],
    permissions: List[str],
    secret: str,
    expires_hours: int,
    algorithm: str = "HS256",
) -> str:
    """签发 JWT。权限/角色内嵌 token，校验阶段无需查库。"""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "username": username,
        "display_name": display_name,
        "tenant_id": tenant_id,
        "roles": list(roles),
        "permissions": sorted(set(permissions)),
        "iat": now,
        "exp": now + int(expires_hours * 3600),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(
    token: str, secret: str, algorithm: str = "HS256"
) -> Optional[Dict[str, Any]]:
    """解析 JWT；无效/过期/签名错误返回 None。"""
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.PyJWTError as e:
        logger.debug("token 解析失败: %s", e)
        return None
