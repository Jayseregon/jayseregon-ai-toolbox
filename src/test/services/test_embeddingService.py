import numpy as np
import pytest
import pytest_asyncio
from fastapi import HTTPException

from src.models.embedding import ModelName
from src.services.embedding import EmbeddingService


@pytest_asyncio.fixture
async def embedding_service(mock_sentence_transformer):
    return EmbeddingService(ModelName.MPNET)


class TestEmbeddingService:
    def test_initialization(self, mock_sentence_transformer):
        service = EmbeddingService(ModelName.MPNET)
        assert service.model is not None

    def test_initialization_failure(self, monkeypatch):
        def raise_error(*args, **kwargs):
            raise Exception("Model loading failed")

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer.__init__", raise_error
        )

        with pytest.raises(HTTPException) as exc_info:
            EmbeddingService(ModelName.MPNET)
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to initialize embedding model"

    def test_create_embeddings(self, embedding_service, mock_sentence_transformer):
        keywords = ["test", "keywords"]
        _ = embedding_service.create_embeddings(keywords)
        mock_sentence_transformer.encode.assert_called_once_with(keywords)

    def test_create_embeddings_failure(
        self, embedding_service, mock_sentence_transformer
    ):
        mock_sentence_transformer.encode.side_effect = Exception("Encoding failed")
        embedding_service.model = mock_sentence_transformer

        with pytest.raises(HTTPException) as exc_info:
            embedding_service.create_embeddings(["test"])
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to create embeddings"

    def test_reduce_dimensions(self, embedding_service, mock_embeddings):
        reduced = embedding_service.reduce_dimensions(mock_embeddings)
        assert reduced.shape[1] == 2
        assert reduced.shape[0] == mock_embeddings.shape[0]

    def test_get_normalized_list(self, embedding_service):
        test_data = np.array([[1, 2], [3, 4], [5, 6]])
        normalized = embedding_service.get_normalized_list(test_data)

        assert all(0 <= x <= 1 for row in normalized for x in row)
        assert len(normalized) == len(test_data)

    def test_get_embeddings(self, embedding_service):
        normalized = [[0.1, 0.2], [0.3, 0.4]]
        keywords = ["test1", "test2"]

        result = embedding_service.get_embeddings(normalized, keywords)

        assert len(result.keywords) == 2
        assert result.keywords[0].word == "test1"
        assert result.keywords[0].x == 0.1
        assert result.keywords[0].y == 0.2

    @pytest_asyncio.fixture(autouse=True)
    async def _setup(self, embedding_service):
        self.service = embedding_service

    @pytest.mark.asyncio
    async def test_process_keywords(self):
        keywords = ["test1", "test2", "test3"]
        result = await self.service.process_keywords(keywords)

        assert len(result.keywords) == 3
        assert all(0 <= k.x <= 1 and 0 <= k.y <= 1 for k in result.keywords)
        assert [k.word for k in result.keywords] == keywords

    @pytest.mark.asyncio
    async def test_process_keywords_failure(self, mock_sentence_transformer):
        mock_sentence_transformer.encode.side_effect = Exception("Processing failed")
        self.service.model = mock_sentence_transformer

        with pytest.raises(HTTPException) as exc_info:
            await self.service.process_keywords(["test"])
        assert exc_info.value.status_code == 500
        # The error should now propagate from create_embeddings
        assert exc_info.value.detail == "Failed to create embeddings"
