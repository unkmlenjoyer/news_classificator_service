import os

import numpy as np
import uvicorn
from database.data_extractor import NewsClassifierDB
from dotenv import load_dotenv
from fastapi import FastAPI
from schemas.schemas import NewsHeadline, NewsScores
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
def predict_category(headline: NewsHeadline) -> NewsScores:
    # prepare text
    preprocessed = formatter.process_text(headline.data)

    # get scores and map them by category
    probas = model.predict_proba(np.array([preprocessed])).ravel()
    mapped_score = {idx2category[i]: score for i, score in enumerate(probas)}

    # send data to db
    db.insert_prediction({"text": headline.data, "prediction": mapped_score})

    return {"scores": mapped_score}


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
