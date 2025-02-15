from typing import List

from pydantic import BaseModel


class Sentence(BaseModel):
    text: str


class Keywords(BaseModel):
    keywords: List[str]


class EmbeddedKeyword(BaseModel):
    word: str
    x: float
    y: float


class Embeddings(BaseModel):
    keywords: List[EmbeddedKeyword]
