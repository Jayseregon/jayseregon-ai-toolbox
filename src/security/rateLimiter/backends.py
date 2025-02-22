import asyncio
from abc import ABC, abstractmethod

import redis as pyredis
import valkey
import valkey.exceptions


class RateLimiterBackend(ABC):
    @abstractmethod
    async def eval_limiter(
        self, key: str, limit: int, expire: int, lua_sha: str, lua_script: str
    ) -> int:
        pass

    @abstractmethod
    async def load_script(self, lua_script: str) -> str:
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
            pexpire = await self.redis.evalsha(lua_sha, 1, key, str(limit), str(expire))
        return pexpire

    async def load_script(self, lua_script: str) -> str:
        if getattr(self.redis, "script_load", None):
            return await self.redis.script_load(lua_script)
        else:
            return await asyncio.to_thread(self.redis.script_load, lua_script)


class ValkeyRateLimiterBackend(RateLimiterBackend):
    def __init__(self, valkey_client):
        self.client = valkey_client

    async def eval_limiter(
        self, key: str, limit: int, expire: int, lua_sha: str, lua_script: str
    ) -> int:
        try:
            pexpire = await asyncio.to_thread(
                self.client.evalsha, lua_sha, 1, key, str(limit), str(expire)
            )
        except valkey.exceptions.NoScriptError:
            lua_sha = await asyncio.to_thread(self.client.script_load, lua_script)
            pexpire = await asyncio.to_thread(
                self.client.evalsha, lua_sha, 1, key, str(limit), str(expire)
            )
        return pexpire

    async def load_script(self, lua_script: str) -> str:
        return await asyncio.to_thread(self.client.script_load, lua_script)
