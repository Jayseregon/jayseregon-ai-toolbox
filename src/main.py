import logging
from contextlib import asynccontextmanager

import redis.asyncio as redis
import valkey
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from src.configs.env_config import config
from src.configs.log_config import configure_logging
from src.routes.embedding import router as embedding_router
from src.security.rateLimiter import FastAPILimiter
from src.security.rateLimiter.backends import (
    RedisRateLimiterBackend,
    ValkeyRateLimiterBackend,
)
from src.security.rateLimiter.depends import RateLimiter

# Initialize logging
logger = logging.getLogger(__name__)


async def get_backend_instance():
    if config.ENV_STATE == "prod":
        # Use Redis as the rate limiter backend for production
        logger.info("Using Redis as the rate limiter backend")
        redis_client = redis.from_url(config.REDIS_URL)

        if not redis_client:
            logger.error("Please configure Redis client for rate limiting")
            raise Exception("Please configure Redis client for rate limiting")

        return RedisRateLimiterBackend(redis_client)

    # Use Valkey as the rate limiter backend for development
    logger.info("Using Valkey as the rate limiter backend")
    valkey_client = valkey.from_url(config.VALKEY_URL)

    if not valkey_client:
        logger.error("Please configure Valkey client for rate limiting")
        raise Exception("Please configure Valkey client for rate limiting")

    return ValkeyRateLimiterBackend(valkey_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging
    configure_logging()

    # Get rate limiter backend instance
    backend_instance = await get_backend_instance()

    # Initialize rate limiter
    await FastAPILimiter.init(backend=backend_instance)
    yield
    await FastAPILimiter.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.get_allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_allowed_hosts,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config.ENV_STATE == "prod":
    app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(CorrelationIdMiddleware)


@app.get("/")
async def read_root(rate: None = Depends(RateLimiter(times=3, seconds=10))):
    return {"greetings": "Welcome to Jayseregon AI toolbox API."}


app.include_router(embedding_router)


@app.exception_handler(HTTPException)
async def http_exception_handle_logging(request, exc):
    logger.error(f"HTTPException: {exc.status_code} {exc.detail}")
    return await http_exception_handler(request, exc)
