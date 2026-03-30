import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# allow local imports from src package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model_utils import load_model, predict_text

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./model/fake_news_detector.pkl")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

app = FastAPI(title="Fake News Detector API")

model = load_model(MODEL_PATH)

class NewsInput(BaseModel):
    text: str
    title: str = ""

class PredictionOutput(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionOutput)
def predict(input: NewsInput):
    if not input.text:
        raise HTTPException(status_code=422, detail="`text` must be provided")

    label, confidence = predict_text(model, input.text, input.title, threshold=CONFIDENCE_THRESHOLD)

    return PredictionOutput(label=label, confidence=round(confidence, 4))

@app.get("/health")
def health():
    return {"status": "ok"}
