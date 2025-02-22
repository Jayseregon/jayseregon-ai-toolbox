# ----------------------------------------------------------------------
# Portions of this code are derived from fastapi-limiter
# (https://github.com/long2ice/fastapi-limiter.git) which is licensed
# under the Apache License, Version 2.0.
#
# Modifications Copyright (C) Jayseregon, 2025.
#
# This file is part of fastapi-ai-tools which is licensed under LGPL-3.0-or-later.
#
# Licensed under the Apache License, Version 2.0 (for fastapi-limiter derived parts)
# and under LGPL-3.0-or-later for the overall project.
# You may obtain a copy of the Apache License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------

__all__ = [
    "FastAPILimiter",
    "default_identifier",
    "http_default_callback",
    "ws_default_callback",
]

import asyncio
import inspect
import logging
from math import ceil
from typing import Any, Callable, Optional, Union

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from starlette.websockets import WebSocket

from .backends import RateLimiterBackend

logger = logging.getLogger(__name__)


async def default_identifier(request: Union[Request, WebSocket]):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0]
    else:
        ip = request.client.host if request.client else "unknown"
    return ip + ":" + request.scope["path"]


async def http_default_callback(request: Request, response: Response, pexpire: int):
    expire = ceil(pexpire / 1000)
    raise HTTPException(
        HTTP_429_TOO_MANY_REQUESTS,
        "Too Many Requests",
        headers={"Retry-After": str(expire)},
    )


async def ws_default_callback(ws: WebSocket, pexpire: int):
    expire = ceil(pexpire / 1000)
    logger.warning(
        "WebSocket rate limit exceeded, closing connection. Retry-After: %s seconds",
        expire,
    )
    # close with a status indicating service unavailability
    await ws.close(code=1013)


class FastAPILimiter:
    backend: Optional[RateLimiterBackend] = None
    redis: Any = None
    lua_sha: Optional[str] = None
    lua_script: str = "/* Lua rate limiting script */"
    identifier: Optional[Callable] = None
    http_callback: Optional[Callable] = None
    ws_callback: Optional[Callable] = None
    prefix: str = "rate-limit"

    lua_script = """local key = KEYS[1]
local limit = tonumber(ARGV[1])
local expire_time = ARGV[2]

local current = tonumber(redis.call('get', key) or "0")
if current > 0 then
 if current + 1 > limit then
 return redis.call("PTTL",key)
 else
        redis.call("INCR", key)
 return 0
 end
else
    redis.call("SET", key, 1,"px",expire_time)
 return 0
end"""

    @classmethod
    async def init(
        cls,
        backend: RateLimiterBackend,
        redis_instance: Any = None,
        prefix: str = "fastapi-limiter",
        identifier: Callable = default_identifier,
        http_callback: Callable = http_default_callback,
        ws_callback: Callable = ws_default_callback,
    ) -> None:
        cls.backend = backend
        cls.redis = redis_instance or getattr(backend, "redis", None)
        cls.prefix = prefix
        cls.identifier = identifier
        cls.http_callback = http_callback
        cls.ws_callback = ws_callback
        try:
            cls.lua_sha = await cls.backend.load_script(cls.lua_script)
        except Exception as e:
            logger.error("Error loading Lua script: %s", e)
            raise

    @classmethod
    async def close(cls) -> None:
        # If the backend has a close method, call it.
        if cls.backend and hasattr(cls.backend, "close"):
            close_method = getattr(cls.backend, "close")
            if callable(close_method):
                if inspect.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    await asyncio.to_thread(close_method)
