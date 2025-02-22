from typing import Annotated

from pydantic import BaseModel, Field


class Sentence(BaseModel):
    text: str


class Keywords(BaseModel):
    keywords: Annotated[list[str], Field(min_items=2)]


class EmbeddedKeyword(BaseModel):
    word: str
    x: float
    y: float


class Embeddings(BaseModel):
    keywords: list[EmbeddedKeyword]
