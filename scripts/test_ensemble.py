import os
import sys

# Add the project root (one level up from scripts/) to sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from app.ensemble import Ensemble

MODEL_DIR = os.path.join(ROOT, "model")

ens = Ensemble(MODEL_DIR)

samples = [
    "delivery was late and the box was damaged, very disappointed",
    "works perfectly, battery life is amazing!",
    "it's okay, nothing special but not bad either"
]

for s in samples:
    out = ens.predict_one(s)
    print("\nText:", s)
    print("Final:", out["final_sentiment"], "| confidence:", round(out["confidence"], 3))
    print("Per model:", {k: v["pred"] for k, v in out["per_model"].items()})
