import os
from datasets import load_cleaned_reviews, load_flipkart, load_dataset_sa

# Base project path
ROOT = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(ROOT, "data")

# Load each dataset
df1 = load_cleaned_reviews(os.path.join(data_dir, "cleaned_reviews.csv"))
df2 = load_flipkart(os.path.join(data_dir, "flipkart_rating_review.csv"))
df3 = load_dataset_sa(os.path.join(data_dir, "Dataset-SA.csv"))

# Print summary
print("cleaned_reviews.csv ->", df1.shape, df1["label"].value_counts().to_dict())
print("flipkart_rating_review.csv ->", df2.shape, df2["label"].value_counts().to_dict())
print("Dataset-SA.csv ->", df3.shape, df3["label"].value_counts().to_dict())

# Show a few samples
print("\nSamples:")
print("1:", df1["text"].iloc[0][:120])
print("2:", df2["text"].iloc[0][:120])
print("3:", df3["text"].iloc[0][:120])
