import re
import pandas as pd

# Small helper to clean review text
def _basic_clean(s: str) -> str:
    s = str(s).lower().strip()          # Lowercase and trim
    s = re.sub(r"\s+", " ", s)          # Replace multiple spaces with single space
    return s

# Helper to make sentiment labels consistent
def _normalize_label(lbl: str) -> str:
    lbl = str(lbl).lower().strip()
    # Map common label formats to 3 standard values
    mapping = {
        "pos": "positive", "positive": "positive", "1": "positive", "2": "positive",
        "neu": "neutral", "neutral": "neutral", "0": "neutral",
        "neg": "negative", "negative": "negative", "-1": "negative"
    }
    return mapping.get(lbl, lbl)  # If unknown, keep as is

# Final step for all datasets after rename
def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    # Drop missing values
    df = df.dropna(subset=["text", "label"]).copy()
    # Clean text
    df["text"] = df["text"].map(_basic_clean)
    # Normalize sentiment
    df["label"] = df["label"].map(_normalize_label)
    # Keep only rows with valid labels
    df = df[df["label"].isin(["negative", "neutral", "positive"])]
    # Remove very short reviews (less than 3 chars)
    df = df[df["text"].str.len() > 3]
    return df[["text", "label"]]

# Loader for cleaned_reviews.csv
def load_cleaned_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"cleaned_review": "text", "sentiments": "label"})
    return _finalize(df)

# Loader for flipkart_rating_review.csv
def load_flipkart(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Review": "text", "Sentiment": "label"})
    return _finalize(df)

# Loader for Dataset-SA.csv
def load_dataset_sa(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Review": "text", "Sentiment": "label"})
    return _finalize(df)
