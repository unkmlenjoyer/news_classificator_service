import uvicorn
from fastapi import FastAPI
from schemas.schemas import NewsHeadline, NewsScores

app = FastAPI()


@app.get("/")
def root():
    return {"message": "service alive"}


@app.post("/predict", response_model=NewsScores)
def predict_category(data: NewsHeadline) -> NewsScores:
    return {
        "scores": [
            0,
            0,
            0,
            0,
            0,
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
