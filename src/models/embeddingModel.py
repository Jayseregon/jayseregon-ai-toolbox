from typing import List, Literal

from pydantic import BaseModel


class SentenceRequest(BaseModel):
    text: str
    n_components: Literal[2, 3] = 2


class KeywordsRequest(BaseModel):
    keywords: List[str]
    n_components: Literal[2, 3] = 2


class EmbeddedKeyword(BaseModel):
    word: str
    x: float
    y: float


class Embeddings(BaseModel):
    keywords: List[EmbeddedKeyword]
