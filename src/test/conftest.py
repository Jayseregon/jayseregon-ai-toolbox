import os
from typing import Any, AsyncGenerator, Generator, MutableMapping
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.types import ASGIApp, Receive, Scope, Send

from src.security.rateLimiter import FastAPILimiter

os.environ["ENV_STATE"] = "test"

from src.main import app  # noqa: E402


class DummyBackend:
    async def init(self):
        pass

    async def close(self):
        pass

    async def eval_limiter(self, key, times, milliseconds, lua_sha, lua_script):
        return 0

    async def load_script(self, lua_script):
        return "dummy_sha"


async def dummy_identifier(request):
    return "test_identifier"


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def client() -> Generator:
    yield TestClient(app)


@pytest_asyncio.fixture
async def async_client(client) -> AsyncGenerator[AsyncClient, None]:
    """Create async client for testing"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=client.base_url) as ac:
        yield ac


class SecurityHeadersMiddleware:
    """Custom middleware to add security headers for HTTPS requests"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        is_secure = scope.get("scheme", "http") == "https" or any(
            h
            for h in scope.get("headers", [])
            if h[0] == b"x-forwarded-proto" and h[1] == b"https"
        )

        async def wrapped_send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start" and is_secure:
                headers = list(message.get("headers", []))
                headers.append(
                    (
                        b"strict-transport-security",
                        b"max-age=31536000; includeSubDomains",
                    )
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, wrapped_send)


@pytest_asyncio.fixture
async def https_app() -> FastAPI:
    """Creates a FastAPI app instance configured for HTTPS testing"""
    app = FastAPI()

    # Add security headers middleware first
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(HTTPSRedirectMiddleware)

    @app.get("/")
    async def test_root():
        return {"message": "test"}

    return app


@pytest_asyncio.fixture(autouse=True)
async def mock_rate_limiter():
    """Initialize rate limiter with a dummy backend for all tests."""
    backend = DummyBackend()
    await FastAPILimiter.init(backend, identifier=dummy_identifier)
    yield
    if FastAPILimiter.backend:
        await FastAPILimiter.close()


@pytest.fixture
def mock_env_state(monkeypatch):
    """Fixture to control environment state and variables for tests"""

    def _set_env(env_state=None, env_vars=None):
        # Clear all relevant environment variables
        for key in list(os.environ.keys()):
            if any(prefix in key for prefix in ["ENV_STATE", "DEV_", "PROD_", "TEST_"]):
                monkeypatch.delenv(key, raising=False)

        # Set new environment state if provided
        if env_state is not None:
            monkeypatch.setenv("ENV_STATE", env_state)

        # Set additional environment variables if provided
        if env_vars:
            for key, value in env_vars.items():
                monkeypatch.setenv(key, value)

    return _set_env


@pytest.fixture
def mock_embeddings():
    return np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]])


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """Mock SentenceTransformer for testing"""
    mock = MagicMock()
    mock.encode.return_value = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]]
    )

    def mock_init(self, *args, **kwargs):
        self.model = mock
        return None

    monkeypatch.setattr("sentence_transformers.SentenceTransformer.__init__", mock_init)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock.encode)

    return mock
