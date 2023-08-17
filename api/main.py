import os

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from schemas.schemas import NewsHeadline, NewsScores
from src.classifier import ArtifactLoader
from src.text_utils import TextPreprocess

load_dotenv()

APP_HOST = os.getenv("APP_HOST")
APP_PORT = int(os.getenv("APP_PORT"))

app = FastAPI()
formatter = TextPreprocess()
model = ArtifactLoader.load("storage/tfidf_logreg.pkl")
idx2category = ArtifactLoader.load("storage/idx2category.pkl")


@app.get("/")
def root():
    return {"message": "service alive"}


@app.post("/predict", response_model=NewsScores)
def predict_category(headline: NewsHeadline) -> NewsScores:
    # prepare text
    preprocessed = formatter.process_text(headline.data)

    # get scores and map them by category
    probas = model.predict_proba(np.array([preprocessed])).ravel()
    mapped_score = {idx2category[i]: score for i, score in enumerate(probas)}

    return {"scores": mapped_score}


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
