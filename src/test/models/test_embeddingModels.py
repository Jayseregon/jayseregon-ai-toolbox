import pytest
from pydantic import ValidationError

from src.models.embedding import (
    EmbeddedKeyword,
    Embeddings,
    Keywords,
    ModelName,
    Sentence,
)


@pytest.fixture
def valid_keywords():
    return ["python", "fastapi", "testing"]


@pytest.fixture
def valid_embedded_keywords():
    return [
        EmbeddedKeyword(word="python", x=0.1, y=0.2),
        EmbeddedKeyword(word="fastapi", x=0.3, y=0.4),
        EmbeddedKeyword(word="testing", x=0.5, y=0.6),
    ]


class TestModelName:
    def test_valid_model_names(self):
        assert ModelName.MPNET == "all-mpnet-base-v2"
        assert ModelName.MINI_L6 == "all-MiniLM-L6-v2"
        assert ModelName.MINI_L12 == "all-MiniLM-L12-v2"

    def test_invalid_model_name(self):
        with pytest.raises(ValueError):
            ModelName("invalid-model")


class TestSentence:
    def test_valid_sentence(self):
        sentence = Sentence(text="This is a test sentence")
        assert sentence.text == "This is a test sentence"

    def test_empty_sentence(self):
        sentence = Sentence(text="")
        assert sentence.text == ""


class TestKeywords:
    def test_valid_keywords(self, valid_keywords):
        keywords = Keywords(keywords=valid_keywords)
        assert keywords.keywords == valid_keywords

    def test_invalid_keywords_length(self):
        # Test with only one keyword
        with pytest.raises(ValidationError):
            Keywords(keywords=["python"])

        # Test with too many keywords
        with pytest.raises(ValidationError):
            Keywords(keywords=["word"] * 101)


class TestEmbeddedKeyword:
    def test_valid_embedded_keyword(self):
        keyword = EmbeddedKeyword(word="python", x=0.5, y=0.5)
        assert keyword.word == "python"
        assert keyword.x == 0.5
        assert keyword.y == 0.5

    def test_embedded_keyword_with_integer_coordinates(self):
        keyword = EmbeddedKeyword(word="python", x=1, y=0)
        assert isinstance(keyword.x, float)
        assert isinstance(keyword.y, float)


class TestEmbeddings:
    def test_valid_embeddings(self, valid_embedded_keywords):
        embeddings = Embeddings(keywords=valid_embedded_keywords)
        assert len(embeddings.keywords) == 3
        assert all(isinstance(k, EmbeddedKeyword) for k in embeddings.keywords)

    def test_empty_embeddings(self):
        embeddings = Embeddings(keywords=[])
        assert embeddings.keywords == []
