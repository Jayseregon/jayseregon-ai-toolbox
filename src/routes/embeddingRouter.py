from functools import lru_cache

from fastapi import APIRouter, Depends

from src.models.embeddingModel import KeywordsRequest, SentenceRequest
from src.services.embeddingService import Embeddings, EmbeddingService

router = APIRouter(prefix="/v1/embedding", tags=["embedding"])


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Create an instance of the EmbeddingService class."""
    return EmbeddingService()


@router.post("/keywords", response_model=Embeddings)
async def create_embeddings(
    keywordsRequest: KeywordsRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Create embeddings from a list of keywords."""
    return embedding_service.process_keywords(
        keywordsRequest.keywords, n_components=keywordsRequest.n_components
    )


@router.post("/sentence", response_model=Embeddings)
async def process_demo_text(
    sentenceRequest: SentenceRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Process a demo text by splitting it into words and creating embeddings."""
    keywords = sentenceRequest.text.split()
    return embedding_service.process_keywords(
        keywords, n_components=sentenceRequest.n_components
    )
