from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    MPNET = "all-mpnet-base-v2"
    MINI_L6 = "all-MiniLM-L6-v2"
    MINI_L12 = "all-MiniLM-L12-v2"


class Sentence(BaseModel):
    text: str


class Keywords(BaseModel):
    keywords: Annotated[list[str], Field(min_length=2, max_length=100)]


class EmbeddedKeyword(BaseModel):
    word: str
    x: float
    y: float


class Embeddings(BaseModel):
    keywords: list[EmbeddedKeyword]
