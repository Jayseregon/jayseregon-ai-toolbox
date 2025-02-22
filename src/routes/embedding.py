from functools import lru_cache

from fastapi import APIRouter, Depends

from src.models.embedding import Keywords, Sentence
from src.services.embedding import Embeddings, EmbeddingService

router = APIRouter(prefix="/v1/embedding", tags=["embedding"])


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Create an instance of the EmbeddingService class."""
    return EmbeddingService()


@router.post("/keywords", response_model=Embeddings)
async def create_embeddings(
    keywords: Keywords,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Create embeddings from a list of keywords."""
    return embedding_service.process_keywords(keywords.keywords)


@router.post("/sentence", response_model=Embeddings)
async def process_demo_text(
    sentence: Sentence,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Process a demo text by splitting it into words and creating embeddings."""
    keywords = sentence.text.split()
    return embedding_service.process_keywords(keywords)
