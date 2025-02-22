from abc import ABC, abstractmethod

import redis as pyredis

# ...existing code or imports...


class RateLimiterBackend(ABC):
    @abstractmethod
    async def eval_limiter(
        self, key: str, limit: int, expire: int, lua_sha: str, lua_script: str
    ) -> int:
        pass


class RedisRateLimiterBackend(RateLimiterBackend):
    def __init__(self, redis_instance):
        self.redis = redis_instance

    async def eval_limiter(
        self, key: str, limit: int, expire: int, lua_sha: str, lua_script: str
    ) -> int:
        try:
            pexpire = await self.redis.evalsha(lua_sha, 1, key, str(limit), str(expire))
        except pyredis.exceptions.NoScriptError:
            lua_sha = await self.redis.script_load(lua_script)
            # Update lua_sha globally so that future calls use the loaded script
            lua_sha = lua_sha
            pexpire = await self.redis.evalsha(lua_sha, 1, key, str(limit), str(expire))
        return pexpire
