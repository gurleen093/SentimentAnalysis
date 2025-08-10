# scripts/train_single.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

def train_and_save(df: pd.DataFrame, out_dir: str):
    """
    Trains a TF-IDF + LinearSVC (calibrated) classifier on the given df,
    then saves vectorizer and model into out_dir.
    df must have columns: text, label (values in {negative, neutral, positive}).
    """
    os.makedirs(out_dir, exist_ok=True)

    # basic sanity
    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["label"].isin(["negative", "neutral", "positive"])]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # vectorize text
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=3)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # model: LinearSVC is strong; wrap with calibration to get predict_proba
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(Xtr, y_train)

    # evaluate
    y_pred = clf.predict(Xte)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, digits=4))

    # save artifacts
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.pkl"))
    joblib.dump(clf,        os.path.join(out_dir, "model.pkl"))
    print(f"Saved to: {out_dir}/vectorizer.pkl and {out_dir}/model.pkl")
