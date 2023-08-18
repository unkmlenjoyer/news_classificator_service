import datetime
import hashlib
import os
from typing import List

import numpy as np
import uvicorn
from database.data_extractor import NewsClassifierDB
from dotenv import load_dotenv
from fastapi import FastAPI
from schemas.schemas import NewsInputData, NewsOutputData, NewsScores, NewsShortsData
from src.classifier import ArtifactLoader
from src.text_utils import TextPreprocess

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
APP_HOST = os.getenv("APP_HOST")
APP_PORT = int(os.getenv("APP_PORT"))

db = NewsClassifierDB(DB_HOST, DB_PORT)

app = FastAPI()
formatter = TextPreprocess()
model = ArtifactLoader.load("storage/tfidf_logreg.pkl")
idx2category = ArtifactLoader.load("storage/idx2category.pkl")


@app.get("/")
def root():
    return {"message": "service is alive"}


@app.post("/predict", response_model=NewsScores)
def predict_category(headline: NewsInputData) -> NewsScores:
    # TODO handle if text is empty before or after preprocessing

    # prepare text
    preprocessed = formatter.process_text(headline.data)

    # get scores and map them by category
    probas = model.predict_proba(np.array([preprocessed])).ravel()
    mapped_score = {idx2category[i]: score for i, score in enumerate(probas)}

    # generate unique id for given text
    insert_datetime = str(datetime.datetime.now())
    text_id = hashlib.sha1(insert_datetime.encode("UTF-8")).hexdigest()[:16]

    # TODO logging inserting
    # send data to db
    db.insert_prediction(
        {
            "text_id": text_id,
            "insert_time": insert_datetime,
            "text": headline.data,
            "prediction": mapped_score,
        }
    )

    return {"scores": mapped_score}


# TODO handle if there is no such ID
@app.get("/get_news_data", response_model=NewsOutputData)
def retrieve_news(news_id: str) -> NewsOutputData:
    result = db.get_one_news(news_id)
    return result


# TODO handle if there is no result
@app.post("/get_all_ids", response_model=List[NewsShortsData])
def retrieve_news(news_id: List[str]) -> List[NewsShortsData]:
    result = db.select_news_short(news_id)
    return list(result)


# TODO create logging
@app.delete("/delete_news")
def delete_news(news_id: str):
    db.delete_one_news(news_id)
    return {"message": f"News with ID = {news_id} deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
