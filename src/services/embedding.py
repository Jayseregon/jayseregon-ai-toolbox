import asyncio
import logging
from typing import List, Literal, Tuple

import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.models.embedding import EmbeddedKeyword, Embeddings, ModelName

# Initialize logging
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: ModelName) -> None:
        try:
            logger.debug(f"Loading model {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to initialize embedding model"
            )

    def create_embeddings(self, keywords: List[str]) -> np.ndarray:
        try:
            logger.debug(f"Encoding keywords: {keywords}")
            return self.model.encode(keywords)
        except Exception as e:
            logger.error(f"Embedding creation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create embeddings")

    def reduce_dimensions(
        self, embeddings: np.ndarray, n_components: Literal[2, 3] = 2
    ) -> np.ndarray:
        logger.debug(f"Reducing dimensions with {n_components} components")
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)

    def get_normalized_list(
        self, embeddings: np.ndarray, value_range: Tuple[float, float] = (0, 1)
    ) -> List[float]:
        logger.debug("Normalizing embeddings.")
        scaler = MinMaxScaler(feature_range=value_range)
        normalized = scaler.fit_transform(embeddings)
        return normalized.tolist()

    def get_embeddings(
        self, normalized_embeddings: List, keywords: List[str]
    ) -> Embeddings:
        logger.debug("Mapping normalized embeddings.")
        return Embeddings(
            keywords=[
                EmbeddedKeyword(word=word, x=x, y=y)
                for word, (x, y) in zip(keywords, normalized_embeddings)
            ]
        )

    async def process_keywords(
        self,
        keywords: List[str],
    ) -> Embeddings:
        try:
            embeddings = await asyncio.to_thread(self.create_embeddings, keywords)
            reduced = await asyncio.to_thread(self.reduce_dimensions, embeddings)
            normalized = await asyncio.to_thread(self.get_normalized_list, reduced)
            return self.get_embeddings(normalized, keywords)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Keyword processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process keywords")
