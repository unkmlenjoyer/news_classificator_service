"""News classifier API. You can predict category on news"""


# import libraries
import datetime
import hashlib
import os
from typing import List

import numpy as np
import uvicorn
from database.data_extractor import NewsClassifierDB
from dotenv import load_dotenv
from exception.exceptions import (
    EmptyModelInputException,
    NewsNotFound,
    NewsNotInsertedException,
)
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from schemas.schemas import (
    ErrorResponse,
    NewsInputData,
    NewsOutputData,
    NewsScores,
    NewsShortsData,
)
from src.classifier import ArtifactLoader
from src.text_utils import TextPreprocess

# load environment variables from .env file
load_dotenv()

# env for DB (mongo)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# env for FastAPI
APP_HOST = os.getenv("APP_HOST")
APP_PORT = int(os.getenv("APP_PORT"))

# DB Mongo connector
db = NewsClassifierDB(DB_HOST, DB_PORT)

# Preprocessor, classifier, category mapper
formatter = TextPreprocess()
model = ArtifactLoader.load("storage/tfidf_logreg.pkl")
idx2category = ArtifactLoader.load("storage/idx2category.pkl")

app = FastAPI()


# Exception handler for validation
@app.exception_handler(RequestValidationError)
def handle_input(request: Request, exc: ErrorResponse):
    return JSONResponse(
        content={"message": "Lenght of text must be more than 30 chars"},
        status_code=exc.status_code,
    )


# Exception handler for empty model input
@app.exception_handler(EmptyModelInputException)
def handle_empty_pruned_form(request: Request, exc: ErrorResponse):
    return JSONResponse(
        content={"message": exc.detail},
        status_code=exc.status_code,
    )


# Exception handler for non inserted news
@app.exception_handler(NewsNotInsertedException)
def handle_not_inserted(request: Request, exc: ErrorResponse):
    return JSONResponse(
        content={"message": exc.detail},
        status_code=exc.status_code,
    )


@app.get("/")
def root():
    return {"message": "service is alive"}


@app.post("/predict", response_model=NewsScores)
def predict_category(headline: NewsInputData) -> NewsScores:
    # prepare text
    preprocessed = formatter.process_text(headline.data)

    if formatter.is_empty(preprocessed):
        raise EmptyModelInputException(
            400, "Pruned (cleaned) data is empty. Can't use it to model's input"
        )

    # get scores and map them by category
    probas = model.predict_proba(np.array([preprocessed])).ravel()
    mapped_score = {idx2category[i]: score for i, score in enumerate(probas)}

    # generate unique id for given text
    insert_datetime = str(datetime.datetime.now())
    text_id = hashlib.sha1(insert_datetime.encode("UTF-8")).hexdigest()[:16]

    # TODO logging inserting
    # send data to db
    result = db.insert_prediction(
        {
            "text_id": text_id,
            "insert_time": insert_datetime,
            "text": headline.data,
            "prediction": mapped_score,
        }
    )

    if not result.inserted_id:
        raise NewsNotInsertedException(
            500, f"News data with ID = {text_id} didn't inserted"
        )
    return {"scores": mapped_score}


@app.get("/get_news_data", response_model=NewsOutputData)
def retrieve_news(news_id: str) -> NewsOutputData:
    result = db.get_one_news(news_id)
    if result is None:
        raise NewsNotFound(400, f"News with ID = {news_id} not found")
    return result


@app.post("/get_news_batch", response_model=List[NewsShortsData])
def retrieve_news(news_id: List[str]) -> List[NewsShortsData]:
    return list(db.select_news_short(news_id))


# TODO create logging
@app.delete("/delete_news")
def delete_news(news_id: str):
    result = db.delete_one_news(news_id)
    if not result.deleted_count:
        raise NewsNotFound(400, f"News with ID = {news_id} not found and can't delete")
    return {"message": f"News with ID = {news_id} deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
