# app/main.py
import os
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

from app.ensemble import Ensemble
from app.llm import explain_negative, rephrase_brand_friendly

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "..", "model")
ens = Ensemble(model_root=MODEL_DIR)

app = FastAPI(title="Product Review Sentiment API", version="1.1.0")

# Allow your future frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    final_sentiment: str
    confidence: float
    per_model: Dict[str, Dict[str, Any]]

@app.get("/")
def home():
    return {"message": "Sentiment API live. See /docs for Swagger UI."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(inp: ReviewIn):
    return ens.predict_one(inp.text)

class ExplainOut(BaseModel):
    sentiment: str
    explanation: str

@app.post("/explain", response_model=ExplainOut)
def explain(inp: ReviewIn):
    result = ens.predict_one(inp.text)
    if result["final_sentiment"] != "negative":
        return {
            "sentiment": result["final_sentiment"],
            "explanation": "Review is not negative; no escalation needed."
        }
    return {
        "sentiment": result["final_sentiment"],
        "explanation": explain_negative(inp.text)
    }

class RephraseOut(BaseModel):
    rephrased: str

@app.post("/rephrase", response_model=RephraseOut)
def rephrase(inp: ReviewIn):
    return {"rephrased": rephrase_brand_friendly(inp.text)}
