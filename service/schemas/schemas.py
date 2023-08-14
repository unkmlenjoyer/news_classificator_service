from typing import List

from pydantic import BaseModel


class NewsHeadline(BaseModel):
    data: str


class NewsScores(BaseModel):
    scores: List[float]
