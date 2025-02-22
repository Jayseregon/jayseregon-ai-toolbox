import os
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

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


@pytest.fixture()
async def async_client(client) -> AsyncGenerator:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=client.base_url) as ac:
        yield ac


@pytest.fixture(autouse=True)
async def mock_rate_limiter():
    """Initialize rate limiter with a dummy backend for all tests."""
    backend = DummyBackend()
    await FastAPILimiter.init(backend, identifier=dummy_identifier)
    yield
    if FastAPILimiter.backend:
        await FastAPILimiter.close()
