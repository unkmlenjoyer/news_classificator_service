"""Custom validaton schemas"""

from typing import Dict

from pydantic import BaseModel, constr


class ErrorResponse(BaseModel):
    """Customized Error Response"""

    status_code: int
    detail: str


class NewsInputData(BaseModel):
    """Customized News input"""

    data: constr(min_length=30)

    class Config:
        json_schema_extra = {
            "example": {
                "data": "Kaliningrad Region will attract up to 3 million tourists a year",
            }
        }


class NewsScores(BaseModel):
    """Customized validation model for new's scores"""

    scores: Dict[str, float]

    class Config:
        json_schema_extra = {
            "example": {
                "scores": {
                    "PARENTING": 0.040988382420555775,
                    "ENTERTAINMENT": 0.13196931409221313,
                    "POLITICS": 0.15514073112869747,
                    "WELLNESS": 0.05261686760366114,
                    "BUSINESS": 0.07684630909296118,
                    "STYLE & BEAUTY": 0.02075620773424583,
                    "FOOD & DRINK": 0.012706247512765783,
                    "QUEER VOICES": 0.024479378670144415,
                    "TRAVEL": 0.4395957214093014,
                    "HEALTHY LIVING": 0.04490084033545394,
                }
            }
        }


class NewsOutputData(BaseModel):
    """Customized validation model as new's output"""

    text_id: str
    insert_time: str
    text: str
    prediction: Dict[str, float]


class NewsShortsData(BaseModel):
    """Customized validation model as new's short data"""

    text_id: str
    insert_time: str
