"""
Bangla News Topic Engine — Inference (GMM Unigram Edition)
===========================================================
Loads the trained GMM pipeline (from train_pipeline.py) and classifies a
brand-new Bangla article in milliseconds — no Spark or FAISS required.

Uses the optimal GMM Unigram configuration:
unigrams -> TF-IDF -> PCA(5) -> GMM(3)
Hungarian accuracy ~0.8945.

USAGE
-----
    python predict.py "আপনার বাংলা নিউজ আর্টিকেল এখানে..."

or import and call directly:

    from predict import classify_article
    result = classify_article("...")
    print(result)
"""

import sys
import joblib
import numpy as np

from train_pipeline import clean_article, tokenize_and_filter, make_unigram_doc

MODEL_PATH = "model/bangla_topic_model.joblib"

_bundle = None  # lazy-loaded singleton so repeated calls are fast


def _load_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def classify_article(raw_text: str) -> dict:
    bundle = _load_bundle()

    # Preprocessing pipeline
    cleaned = clean_article(raw_text)
    tokens = tokenize_and_filter(cleaned)
    unigram_doc = make_unigram_doc(tokens)

    if not unigram_doc.strip():
        return {
            "predicted_class": None,
            "cluster_id": None,
            "confidence": 0.0,
            "confidence_note": "Article too short after cleaning to form features "
                               "(needs at least 1 usable Bangla content word).",
        }

    # Feature extraction & reduction
    raw_counts = bundle["vectorizer"].transform([unigram_doc])
    tfidf_features = bundle["tfidf"].transform(raw_counts)
    norm_features = bundle["normalizer"].transform(tfidf_features)
    pca_features = bundle["pca"].transform(norm_features.toarray()).astype(np.float32)

    # GMM Prediction & Class Probabilities
    probabilities = bundle["gmm"].predict_proba(pca_features)[0]
    cluster_id = int(np.argmax(probabilities))
    predicted_class = bundle["cluster_to_label"].get(cluster_id, "Unknown")

    # Map probability scores directly to topic labels
    label_probabilities = {
        bundle["cluster_to_label"].get(cid, f"Cluster_{cid}"): round(float(prob), 4)
        for cid, prob in enumerate(probabilities)
    }

    return {
        "predicted_class": predicted_class,
        "cluster_id": cluster_id,
        "confidence": round(float(probabilities[cluster_id]), 4),
        "probabilities": label_probabilities,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<bangla article text>\"")
        sys.exit(1)

    article_text = sys.argv[1]
    result = classify_article(article_text)
    print(result)
