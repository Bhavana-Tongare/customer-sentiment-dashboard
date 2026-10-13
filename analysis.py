"""
STEP 1 of the project: run the full analysis and save results.

Run in Colab or locally:
    pip install -r requirements-analysis.txt
    python analysis.py

Input : dataset.csv  (your Flipkart file)
Output: reviews_scored.csv (used by the dashboard) + charts/*.png (for README)
"""
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

INPUT_CSV = "dataset.csv"
OUTPUT_CSV = "reviews_scored.csv"
MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LABELS = ["negative", "neutral", "positive"]
os.makedirs("charts", exist_ok=True)


# ---------------------------------------------------------------- 1. LOAD + CLEAN
def load_and_clean(path):
    df = pd.read_csv(path, nrows=10000)
    df = df.rename(columns={"Rate": "rating", "Review": "review",
                            "Summary": "summary", "Sentiment": "dataset_sentiment"})
    n_before = len(df)

    df = df.drop_duplicates()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)]
    for col in ["review", "summary"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # The original product names contain junk like "??????" -> remove it
    df["product_name"] = (df["product_name"].astype(str)
                          .str.replace(r"\?+", " ", regex=True)
                          .str.replace(r"\s+", " ", regex=True).str.strip())

    # Model reads review title + longer summary together = more context
    df["text"] = (df["review"] + ". " + df["summary"]).str.strip(". ")
    df = df[df["text"] != ""].reset_index(drop=True)

    print(f"Cleaning: {n_before} rows -> {len(df)} rows")
    return df


# ---------------------------------------------------------------- 2. SENTIMENT MODEL
def run_model(texts):
    from transformers import pipeline
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1

    clf = pipeline("text-classification", model=MODEL, device=device,
                   truncation=True, max_length=128)

    # Many reviews are identical ("super!", "nice product").
    # Score each UNIQUE text once = much faster, same result.
    unique = pd.Series(texts.unique())
    out = clf(unique.tolist(), batch_size=32)
    lookup = {t: (o["label"].lower(), o["score"]) for t, o in zip(unique, out)}

    labels = texts.map(lambda t: lookup[t][0])
    scores = texts.map(lambda t: lookup[t][1])
    return labels, scores


# ---------------------------------------------------------------- 3. VALIDATION
def rating_to_sentiment(r):
    # Our "ground truth" proxy: 1-2 stars negative, 3 neutral, 4-5 positive
    return "negative" if r <= 2 else ("neutral" if r == 3 else "positive")


def validate(df):
    acc = accuracy_score(df["rating_sentiment"], df["model_sentiment"])
    print(f"\nModel agrees with star rating on {acc:.1%} of reviews\n")
    print(classification_report(df["rating_sentiment"], df["model_sentiment"],
                                labels=LABELS, zero_division=0))

    cm = pd.crosstab(df["rating_sentiment"], df["model_sentiment"]) \
           .reindex(index=LABELS, columns=LABELS, fill_value=0)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Model predicted"); plt.ylabel("Star rating says")
    plt.title("Model vs Ratings"); plt.tight_layout()
    plt.savefig("charts/confusion_matrix.png", dpi=150); plt.close()


# ---------------------------------------------------------------- 4. COMPLAINTS
def top_phrases(text_series, n=15):
    """Most common 2-word phrases, e.g. 'not working', 'poor quality'."""
    if len(text_series) < 5:
        return pd.DataFrame(columns=["phrase", "count"])
    vec = CountVectorizer(stop_words="english", ngram_range=(2, 2), min_df=2)
    try:
        X = vec.fit_transform(text_series)
    except ValueError:
        return pd.DataFrame(columns=["phrase", "count"])
    counts = X.sum(axis=0).A1
    return (pd.DataFrame({"phrase": vec.get_feature_names_out(), "count": counts})
            .sort_values("count", ascending=False).head(n))


# ---------------------------------------------------------------- 5. CHARTS
def make_charts(df):
    sns.set_theme(style="whitegrid")

    order = df["model_sentiment"].value_counts().reindex(LABELS)
    ax = order.plot(kind="bar", color=["#E4572E", "#F3A712", "#29BF12"], rot=0)
    ax.bar_label(ax.containers[0]); ax.set_title("Sentiment distribution (model)")
    plt.tight_layout(); plt.savefig("charts/sentiment_distribution.png", dpi=150); plt.close()

    g = (df.groupby("product_name")
           .agg(reviews=("rating", "size"),
                negative_pct=("model_sentiment", lambda s: (s == "negative").mean() * 100))
           .query("reviews >= 30").sort_values("negative_pct").tail(10))
    if len(g):
        g.index = g.index.str[:40]
        g["negative_pct"].plot(kind="barh", color="#E4572E", figsize=(8, 5))
        plt.title("Products with highest % negative reviews (min 30 reviews)")
        plt.xlabel("% negative"); plt.tight_layout()
        plt.savefig("charts/negative_by_product.png", dpi=150); plt.close()

    neg_text = df.loc[df["rating"] <= 2, "summary"]
    tp = top_phrases(neg_text).iloc[::-1]
    if len(tp):
        plt.barh(tp["phrase"], tp["count"], color="#E4572E")
        plt.title("Top complaint phrases (1-2 star reviews)")
        plt.tight_layout(); plt.savefig("charts/complaints.png", dpi=150); plt.close()


# ---------------------------------------------------------------- MAIN
if __name__ == "__main__":
    df = load_and_clean(INPUT_CSV)
    df["model_sentiment"], df["model_confidence"] = run_model(df["text"])
    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    validate(df)
    make_charts(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} scored reviews to {OUTPUT_CSV}")
