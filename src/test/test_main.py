import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from src.configs.env_config import config


@pytest.mark.asyncio
async def test_read_root(async_client):
    """Test the root endpoint returns the expected greeting"""
    response = await async_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"greetings": "Welcome to Jayseregon AI toolbox API."}


@pytest.mark.asyncio
async def test_cors_middleware(async_client):
    """Test CORS middleware configuration"""
    response = await async_client.options(
        "/",
        headers={
            "Origin": config.get_allowed_hosts[0],
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == status.HTTP_200_OK
    assert "access-control-allow-origin" in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheme,expected_status,expected_headers",
    [
        (
            "http",
            status.HTTP_307_TEMPORARY_REDIRECT,
            {"location": "https://testserver/"},
        ),
        (
            "https",
            status.HTTP_200_OK,
            {"strict-transport-security": "max-age=31536000; includeSubDomains"},
        ),
    ],
)
async def test_https_redirect(https_app, scheme, expected_status, expected_headers):
    """Test HTTPS redirect middleware behavior"""
    base_url = f"{scheme}://testserver"
    transport = ASGITransport(app=https_app)

    headers = {
        "Host": "testserver",
    }
    if scheme == "https":
        headers["X-Forwarded-Proto"] = "https"

    async with AsyncClient(
        transport=transport,
        base_url=base_url,
        verify=False,
        follow_redirects=False,
        headers=headers,
    ) as client:
        response = await client.get("/", headers=headers)
        assert response.status_code == expected_status

        for header, value in expected_headers.items():
            assert header.lower() in {k.lower(): v for k, v in response.headers.items()}
            assert response.headers[header] == value


@pytest.mark.asyncio
async def test_correlation_id_middleware(async_client):
    """Test correlation ID middleware adds header to response"""
    response = await async_client.get("/")
    assert (
        "x-request-id" in response.headers
    )  # Changed from X-Correlation-ID to x-request-id


@pytest.mark.asyncio
async def test_error_handling(async_client):
    """Test custom error handling for non-existent endpoint"""
    response = await async_client.get("/non-existent-endpoint")
    assert response.status_code == status.HTTP_404_NOT_FOUND
