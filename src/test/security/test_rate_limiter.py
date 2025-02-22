from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket

from src.security.rateLimiter import FastAPILimiter
from src.security.rateLimiter.backends import (
    RedisRateLimiterBackend,
    ValkeyRateLimiterBackend,
)
from src.security.rateLimiter.depends import RateLimiter, WebSocketRateLimiter


# Test Backend Implementations
@pytest.mark.asyncio
async def test_redis_backend_eval_limiter(mocker):
    redis_mock = mocker.AsyncMock()
    redis_mock.evalsha.return_value = 0
    backend = RedisRateLimiterBackend(redis_mock)

    result = await backend.eval_limiter("test_key", 5, 1000, "sha", "script")
    assert result == 0
    redis_mock.evalsha.assert_called_once()


@pytest.mark.asyncio
async def test_valkey_backend_eval_limiter(mocker):
    valkey_mock = mocker.Mock()
    valkey_mock.evalsha.return_value = 0
    backend = ValkeyRateLimiterBackend(valkey_mock)

    result = await backend.eval_limiter("test_key", 5, 1000, "sha", "script")
    assert result == 0
    valkey_mock.evalsha.assert_called_once()


# Test Rate Limiter Initialization
@pytest.mark.asyncio
async def test_fastapi_limiter_init(mocker):
    backend_mock = mocker.AsyncMock()
    backend_mock.load_script.return_value = "test_sha"

    await FastAPILimiter.init(backend_mock)
    assert FastAPILimiter.backend == backend_mock
    assert FastAPILimiter.lua_sha == "test_sha"
    backend_mock.load_script.assert_called_once_with(FastAPILimiter.lua_script)


# Test Rate Limiter Dependency
@pytest.mark.asyncio
async def test_rate_limiter_under_limit():
    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route(rate_limit: None = Depends(RateLimiter(times=2, seconds=5))):
        return {"status": "ok"}

    # Initialize rate limiter with dummy backend
    backend_mock = AsyncMock()
    backend_mock.eval_limiter.return_value = 0
    await FastAPILimiter.init(backend_mock)

    with TestClient(test_app) as client:
        response = client.get("/test")
        assert response.status_code == 200

        # Second request within limit
        response = client.get("/test")
        assert response.status_code == 200

    await FastAPILimiter.close()


@pytest.mark.asyncio
async def test_rate_limiter_exceeds_limit():
    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route(rate_limit: None = Depends(RateLimiter(times=1, seconds=5))):
        return {"status": "ok"}

    # Initialize rate limiter with dummy backend
    backend_mock = AsyncMock()
    backend_mock.eval_limiter.side_effect = [
        0,
        1000,
    ]  # First request ok, second limited
    await FastAPILimiter.init(backend_mock)

    with TestClient(test_app) as client:
        response = client.get("/test")
        assert response.status_code == 200

        # Second request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    await FastAPILimiter.close()


# Test WebSocket Rate Limiter
@pytest.mark.asyncio
async def test_websocket_rate_limiter():
    # Create a more complete WebSocket mock
    ws_mock = AsyncMock(spec=WebSocket)
    ws_mock.scope = {"path": "/test"}
    ws_mock.client = AsyncMock()
    ws_mock.client.host = "127.0.0.1"
    ws_mock.headers = {}

    limiter = WebSocketRateLimiter(times=1, seconds=5)

    # Initialize rate limiter with dummy backend
    backend_mock = AsyncMock()
    backend_mock.eval_limiter.side_effect = [0, 1000]  # First ok, second limited
    await FastAPILimiter.init(backend_mock)

    # First connection should succeed
    await limiter(ws_mock)
    assert not ws_mock.close.called

    # Second connection should be rate limited
    await limiter(ws_mock)
    ws_mock.close.assert_called_once_with(code=1013)

    await FastAPILimiter.close()


# Test Error Cases
@pytest.mark.asyncio
async def test_rate_limiter_without_init():
    FastAPILimiter.backend = None
    limiter = RateLimiter(times=1, seconds=5)

    request_mock = AsyncMock()
    response_mock = AsyncMock()

    with pytest.raises(Exception, match="You must call FastAPILimiter.init"):
        await limiter(request_mock, response_mock)


@pytest.mark.asyncio
async def test_rate_limiter_with_invalid_identifier():
    async def invalid_identifier(request):
        raise ValueError("Invalid request")

    # Initialize rate limiter with dummy backend
    backend_mock = AsyncMock()
    await FastAPILimiter.init(backend_mock)

    limiter = RateLimiter(times=1, seconds=5, identifier=invalid_identifier)
    request_mock = AsyncMock()
    response_mock = AsyncMock()

    with pytest.raises(Exception, match="Error computing rate key"):
        await limiter(request_mock, response_mock)

    await FastAPILimiter.close()


# Test Custom Callbacks
@pytest.mark.asyncio
async def test_rate_limiter_custom_callback():
    async def custom_callback(request, response, pexpire):
        response.status_code = 418  # I'm a teapot
        response.headers["X-RateLimit-Retry-After"] = str(pexpire)
        return response  # Return the response directly

    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route(
        rate_limit: None = Depends(
            RateLimiter(times=1, seconds=5, callback=custom_callback)
        )
    ):
        return {"status": "ok"}

    # Initialize rate limiter with dummy backend
    backend_mock = AsyncMock()
    backend_mock.eval_limiter.side_effect = [0, 1000]  # First ok, second limited
    await FastAPILimiter.init(backend_mock)

    with TestClient(test_app) as client:
        response = client.get("/test")
        assert response.status_code == 200

        # Second request should use custom callback
        response = client.get("/test")
        assert response.status_code == 418
        assert "X-RateLimit-Retry-After" in response.headers

    await FastAPILimiter.close()
