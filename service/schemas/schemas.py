from typing import Dict

from pydantic import BaseModel


class NewsHeadline(BaseModel):
    data: str


class NewsScores(BaseModel):
    scores: Dict[str, float]
