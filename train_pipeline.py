"""
Bangla News Topic Engine — Training Script (GMM Unigram Edition)
============================================================
Reproduces the cleaning -> normalization -> stopword removal -> unigram
features -> TF-IDF -> PCA (k=5) -> Gaussian Mixture Model (k=3) pipeline
in plain scikit-learn so it can run anywhere and be saved as a
reusable, deployable model.

OPTIMAL CONFIGURATION:
  features          : unigrams
  vocab_size        : 5000
  pca_k             : 5
  clustering        : Gaussian Mixture Model (GMM, k=3)
  Hungarian accuracy: ~0.8945
"""

import os
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

from bangla_stopwords import BANGLA_STOPWORDS

# ------------------------------------------------------------------
# CONFIG — edit these for your environment
# ------------------------------------------------------------------
DATA_FILES = [
    # (path_to_csv, category_label)
    ("/kaggle/input/datasets/kamrun71/bangla-news/National_40k.csv", "National"),
    ("/kaggle/input/datasets/kamrun71/bangla-news/ScienceTechnology_40k.csv", "ScienceTechnology"),
    ("/kaggle/input/datasets/kamrun71/bangla-news/Sports_40k.csv", "Sports"),
]
TEXT_COLUMN = "article"           # column holding the raw article text
SAMPLE_PER_FILE = 8000
SAMPLE_SEED = 42

NGRAM_N = 1                      # Unigrams
VOCAB_SIZE = 5000
PCA_K = 5
N_CLUSTERS = 3
MIN_DF = 5
MAX_DF = 0.4

OUTPUT_DIR = "model"

# Bangla digit -> English digit map
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
EN_DIGITS = "0123456789"
BN2EN = str.maketrans({b: e for b, e in zip(BN_DIGITS, EN_DIGITS)})


# ------------------------------------------------------------------
# TEXT CLEANING & PREPROCESSING
# ------------------------------------------------------------------
def clean_article(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # Pass 1 — structural repair
    t = re.sub(r"[\n\r]+", " ", text)
    t = re.sub(r"[।,!?()\[\]{}:;\"'’“”\-–—/\\]+", "", t)
    t = re.sub(r"[\u00A0\u200C\u200B]+", " ", t)

    # Pass 2 — residual noise removal
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[A-Za-z]+", "", t)

    # Unicode normalization + digit unification + keep Bangla-only
    t = unicodedata.normalize("NFC", t)
    t = t.translate(BN2EN)
    t = re.sub(r"[^\u0980-\u09FF0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_and_filter(text: str) -> list:
    """Whitespace tokenize, drop tokens <=2 chars, remove stopwords."""
    tokens = text.split()
    tokens = [tok for tok in tokens if len(tok) > 2]
    tokens = [tok for tok in tokens if tok not in BANGLA_STOPWORDS]
    return tokens


def make_unigram_doc(tokens: list) -> str:
    """Join tokens back into space-separated string for CountVectorizer."""
    return " ".join(tokens)


def hungarian_cluster_to_label(clusters: np.ndarray, true_labels: pd.Series):
    """Maps cluster ids -> ground-truth class names via the Hungarian algorithm."""
    contingency = pd.crosstab(true_labels, clusters)
    row_ind, col_ind = linear_sum_assignment(-contingency.values)
    mapping = {
        contingency.columns[c]: contingency.index[r]
        for r, c in zip(row_ind, col_ind)
    }
    return mapping, contingency


def load_and_sample_data():
    frames = []
    for path, label in DATA_FILES:
        df = pd.read_csv(path)
        if TEXT_COLUMN not in df.columns:
            raise ValueError(f"'{TEXT_COLUMN}' column not found in {path}")
        df = df[[TEXT_COLUMN]].copy()
        df["class"] = label
        df = df.dropna(subset=[TEXT_COLUMN])
        df = df[df[TEXT_COLUMN].str.strip() != ""]
        if len(df) > SAMPLE_PER_FILE:
            df = df.sample(n=SAMPLE_PER_FILE, random_state=SAMPLE_SEED)
        frames.append(df)
    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=[TEXT_COLUMN])
    return full_df


def main():
    print("Loading and sampling data...")
    df = load_and_sample_data()
    print(f"Loaded {len(df)} articles across {df['class'].nunique()} classes.")

    print("Cleaning + normalizing text...")
    df["clean_article"] = df[TEXT_COLUMN].apply(clean_article)
    df["clean_tokens"] = df["clean_article"].apply(tokenize_and_filter)
    df["unigram_doc"] = df["clean_tokens"].apply(make_unigram_doc)
    df = df[df["unigram_doc"].str.strip() != ""].reset_index(drop=True)

    print("Fitting CountVectorizer + TF-IDF on unigrams...")
    vectorizer = CountVectorizer(
        max_features=VOCAB_SIZE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    raw_counts = vectorizer.fit_transform(df["unigram_doc"])

    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(raw_counts)

    normalizer = Normalizer(norm="l2")
    norm_features = normalizer.fit_transform(tfidf_features)

    print(f"Reducing to {PCA_K} components with PCA...")
    pca = PCA(n_components=PCA_K, random_state=1)
    pca_features = pca.fit_transform(norm_features.toarray()).astype(np.float32)

    print(f"Fitting Gaussian Mixture Model (k={N_CLUSTERS})...")
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=1, covariance_type='full')
    gmm.fit(pca_features)
    clusters = gmm.predict(pca_features)

    # Compute Total Log-Likelihood
    log_likelihood = gmm.score(pca_features) * len(pca_features)
    print(f"Log-Likelihood  : {log_likelihood:.4f}")

    score = silhouette_score(
        pca_features, clusters, metric="euclidean",
        sample_size=min(3000, len(pca_features)), random_state=42,
    )
    print(f"Silhouette Score: {score:.4f}")

    mapping, contingency = hungarian_cluster_to_label(clusters, df["class"])
    print("\nCluster -> class mapping:")
    for cluster_id, class_name in mapping.items():
        print(f"  Cluster {cluster_id} -> {class_name}")
    print("\nHungarian Crosstab:")
    print(contingency)

    total = len(df)
    correct = sum(
        1 for c, actual in zip(clusters, df["class"])
        if mapping.get(c) == actual
    )
    print(f"\nHungarian accuracy: {correct / total:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bundle = {
        "vectorizer": vectorizer,
        "tfidf": tfidf,
        "normalizer": normalizer,
        "pca": pca,
        "gmm": gmm,
        "cluster_to_label": mapping,
        "config": {
            "ngram_n": NGRAM_N,
            "vocab_size": VOCAB_SIZE,
            "pca_k": PCA_K,
            "n_clusters": N_CLUSTERS,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
        },
        "log_likelihood": log_likelihood,
        "silhouette_score": score,
        "hungarian_accuracy": correct / total,
    }
    out_path = os.path.join(OUTPUT_DIR, "bangla_topic_model.joblib")
    joblib.dump(bundle, out_path)
    print(f"\nSaved trained GMM pipeline to {out_path}")


if __name__ == "__main__":
    main()
