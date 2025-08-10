# scripts/train_all.py
import os
from datasets import load_cleaned_reviews, load_flipkart, load_dataset_sa
from train_single import train_and_save

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")

def main():
    # 1) cleaned_reviews.csv
    df1 = load_cleaned_reviews(os.path.join(DATA_DIR, "cleaned_reviews.csv"))
    out1 = os.path.join(MODEL_DIR, "cleaned_reviews")
    print("\n=== Training on cleaned_reviews.csv ===")
    train_and_save(df1, out1)

    # 2) flipkart_rating_review.csv
    df2 = load_flipkart(os.path.join(DATA_DIR, "flipkart_rating_review.csv"))
    out2 = os.path.join(MODEL_DIR, "flipkart")
    print("\n=== Training on flipkart_rating_review.csv ===")
    train_and_save(df2, out2)

    # 3) Dataset-SA.csv
    df3 = load_dataset_sa(os.path.join(DATA_DIR, "Dataset-SA.csv"))
    out3 = os.path.join(MODEL_DIR, "dataset_sa")
    print("\n=== Training on Dataset-SA.csv ===")
    train_and_save(df3, out3)

if __name__ == "__main__":
    main()
