import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, status

from src.models.embedding import Keywords, Sentence
from src.security.rateLimiter.depends import RateLimiter
from src.services.embedding import Embeddings, EmbeddingService

# Initialize logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/embedding", tags=["embedding"])


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Create an instance of the EmbeddingService class."""
    logger.debug("Creating an instance of EmbeddingService")
    return EmbeddingService()


@router.post(
    "/keywords", response_model=Embeddings, status_code=status.HTTP_201_CREATED
)
async def create_embeddings(
    keywords: Keywords,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    rate: None = Depends(RateLimiter(times=3, seconds=10)),
):
    """Create embeddings from a list of keywords."""
    logger.debug(f"Processing keywords embedding for {keywords.keywords}")
    return embedding_service.process_keywords(keywords.keywords)


@router.post(
    "/sentence", response_model=Embeddings, status_code=status.HTTP_201_CREATED
)
async def process_demo_text(
    sentence: Sentence,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    rate: None = Depends(RateLimiter(times=3, seconds=10)),
):
    """Process a demo text by splitting it into words and creating embeddings."""
    logger.debug(f"Processing sentence embedding for {sentence.text}")
    keywords = sentence.text.split()
    return embedding_service.process_keywords(keywords)
