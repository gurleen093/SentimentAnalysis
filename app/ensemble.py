# app/ensemble.py
import os
import re
import joblib
import numpy as np
from typing import Dict, Any, List
from nltk.sentiment import SentimentIntensityAnalyzer

LABELS = ["negative", "neutral", "positive"]

NEG_CUES = {
    "broken","damaged","defective","cracked","faulty","leaking","scratched",
    "late","delay","delayed","missing","lost","slow","never arrived",
    "refund","return","replacement","cancelled","scam","fraud","fake",
    "disappointed","worst","terrible","awful","poor","bad","rude","horrible",
    "hate","useless","waste","money","trash","garbage","pathetic","disgusted",
    "doesn't work","didn't work","not working","stopped working","failure",
    "avoid","warning","regret","mistake","problem","issue","complain","angry"
}

def _soft_scores(model, X):
    return model.predict_proba(X)  # calibrated SVC

def _contains_cues(text: str) -> bool:
    t = text.lower()
    if any(w in t for w in NEG_CUES):
        return True
    # enhanced negation patterns
    if re.search(r"\bnot (good|great|working|as described|worth|satisfied|happy|pleased|recommend)\b", t):
        return True
    # poor quality indicators
    if re.search(r"\b(1|one) star|poor quality|low quality|waste of money|do not buy|don't buy\b", t):
        return True
    # disappointment patterns
    if re.search(r"\bexpected (better|more)|fell apart|broke (down|after)|within (days|weeks) of\b", t):
        return True
    return False

class Ensemble:
    """
    Loads 3 (vectorizer, model) members, adds VADER as a 4th signal,
    then predicts by weighted averaging + simple keyword cue nudges.
    """
    def __init__(self, model_root: str):
        self.members: List = []
        for sub in ["cleaned_reviews", "flipkart", "dataset_sa"]:
            vec_path = os.path.join(model_root, sub, "vectorizer.pkl")
            mdl_path = os.path.join(model_root, sub, "model.pkl")
            if not (os.path.exists(vec_path) and os.path.exists(mdl_path)):
                raise FileNotFoundError(f"Missing artifacts in {os.path.join(model_root, sub)}")
            vectorizer = joblib.load(vec_path)
            model = joblib.load(mdl_path)
            self.members.append((sub, vectorizer, model))

        # VADER
        self.vader = SentimentIntensityAnalyzer()

    def _vader_probs(self, text: str) -> np.ndarray:
        """
        Convert VADER compound score into a pseudo-prob distribution over [neg,neu,pos].
        """
        c = self.vader.polarity_scores(text)["compound"]  # -1 .. +1
        pos = max(0.0, c)
        neg = max(0.0, -c)
        neu = max(0.0, 1.0 - (pos + neg))
        vec = np.array([neg, neu, pos], dtype=float)
        s = vec.sum()
        return vec / s if s > 0 else np.array([1/3,1/3,1/3])

    def predict_one(self, text: str) -> Dict[str, Any]:
        votes = []
        probs = []
        details = {}

        # 1) model members
        for name, vec, mdl in self.members:
            X = vec.transform([text])
            p = _soft_scores(mdl, X)[0]
            idx = int(np.argmax(p))
            label = LABELS[idx]
            votes.append(label)
            probs.append(p)
            details[name] = {
                "pred": label,
                "proba": {LABELS[i]: float(p[i]) for i in range(len(LABELS))}
            }

        # 2) VADER member (weight it a bit lighter)
        vader_p = self._vader_probs(text)
        details["vader"] = {
            "pred": LABELS[int(np.argmax(vader_p))],
            "proba": {LABELS[i]: float(vader_p[i]) for i in range(3)}
        }

        # 3) Weighted average of probabilities: 3 ML models (weight 1.0 each) + VADER (weight 0.6)
        ml_avg = np.mean(np.vstack(probs), axis=0)            # (3,)
        combined = (ml_avg * 1.0 + vader_p * 0.6) / (1.0 + 0.6)

        # 4) Enhanced keyword cue nudge toward negative when obvious red flags appear
        if _contains_cues(text):
            combined[0] += 0.25  # stronger boost to negative
            combined[2] -= 0.15  # stronger dampening of positive 
            combined[1] -= 0.10  # stronger dampening of neutral
            # re-normalize
            combined = np.clip(combined, 0, None)
            combined = combined / combined.sum()

        # 5) Additional anti-neutral bias for potential negatives
        # If negative probability is reasonably high but neutral is winning, bias toward negative
        if combined[1] > combined[0] and combined[0] > 0.25:  # neutral winning but negative substantial
            # Transfer some probability from neutral to negative
            transfer = min(0.1, combined[1] * 0.3)
            combined[0] += transfer
            combined[1] -= transfer
            combined = combined / combined.sum()

        final_idx = int(np.argmax(combined))
        final = LABELS[final_idx]
        confidence = float(np.max(combined))

        return {
            "final_sentiment": final,
            "confidence": confidence,
            "per_model": details,
            "combined_probs": {LABELS[i]: float(combined[i]) for i in range(3)}
        }
