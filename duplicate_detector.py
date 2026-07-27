"""
Bangla News Duplicate Detector
================================
Reproduces the notebook's Section 7 (HashingTF -> MinHashLSH ->
approxSimilarityJoin) in plain Python using the `datasketch` library,
so duplicate/near-duplicate detection works without a Spark session.

Matches your notebook's validated settings:
  features        : trigrams (same trigram_doc used by the classifier)
  jaccard distance threshold : 0.30  ->  similarity >= 0.70
  num_hash_tables  : 8  (the value you used for validation)

USAGE
-----
    from duplicate_detector import ArticleStore

    store = ArticleStore()
    store.load("model/article_store.pkl")   # or store.add(...) fresh

    # Add an article (also stores it for future duplicate checks)
    result = store.add_article(article_id="a1", text="...", topic="Sports")

    # result["duplicates"] -> list of {article_id, similarity, topic}
    # for anything already in the store above the similarity threshold

    store.save("model/article_store.pkl")
"""

import os
import pickle
from datasketch import MinHash, MinHashLSH

from train_pipeline import clean_article, tokenize_and_filter

NUM_PERM = 128              # MinHash permutations (standard datasketch default)
JACCARD_SIMILARITY_THRESHOLD = 0.70   # matches notebook's distance=0.30 -> sim>=0.70
NGRAM_N = 3                 # trigrams, same as the classifier


def _trigram_shingles(text: str) -> set:
    """Build the same trigram feature set the classifier trains on,
    but as a set of distinct shingles for Jaccard/MinHash purposes."""
    cleaned = clean_article(text)
    tokens = tokenize_and_filter(cleaned)
    if len(tokens) < NGRAM_N:
        return set()
    return {
        "_".join(tokens[i:i + NGRAM_N])
        for i in range(len(tokens) - NGRAM_N + 1)
    }


def _make_minhash(shingles: set) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for shingle in shingles:
        mh.update(shingle.encode("utf8"))
    return mh


class ArticleStore:
    """Keeps a MinHashLSH index plus lightweight metadata (topic, id) for
    every article seen so far, enabling both:
      - duplicate/near-duplicate detection on ingest
      - similarity lookups for a personalized "similar articles" feed
    """

    def __init__(self):
        self.lsh = MinHashLSH(threshold=JACCARD_SIMILARITY_THRESHOLD, num_perm=NUM_PERM)
        self.metadata = {}   # article_id -> {"topic": ..., "text_preview": ...}
        self.minhashes = {}  # article_id -> MinHash (needed to recompute similarity scores)

    def add_article(self, article_id: str, text: str, topic: str = None) -> dict:
        """Adds an article to the store. Returns any near-duplicates found
        BEFORE this article is inserted, so you can decide whether to skip
        storing it."""
        shingles = _trigram_shingles(text)
        if not shingles:
            return {"added": False, "reason": "Too short to fingerprint.", "duplicates": []}

        mh = _make_minhash(shingles)

        # Query existing index for anything similar before inserting
        candidate_ids = self.lsh.query(mh)
        duplicates = []
        for cand_id in candidate_ids:
            cand_mh = self.minhashes[cand_id]
            similarity = mh.jaccard(cand_mh)
            duplicates.append({
                "article_id": cand_id,
                "similarity": round(float(similarity), 4),
                "topic": self.metadata[cand_id].get("topic"),
            })
        duplicates.sort(key=lambda d: -d["similarity"])

        if article_id in self.metadata:
            # Re-adding same id: remove old entry first (datasketch LSH
            # doesn't support update-in-place)
            self.lsh.remove(article_id)

        self.lsh.insert(article_id, mh)
        self.minhashes[article_id] = mh
        self.metadata[article_id] = {
            "topic": topic,
            "text_preview": text[:200],
        }

        return {"added": True, "duplicates": duplicates}

    def find_similar(self, text: str, top_k: int = 5) -> list:
        """Returns the top_k most similar stored articles to the given
        text, without inserting it into the store."""
        shingles = _trigram_shingles(text)
        if not shingles:
            return []
        mh = _make_minhash(shingles)
        candidate_ids = self.lsh.query(mh)
        scored = [
            {
                "article_id": cid,
                "similarity": round(float(mh.jaccard(self.minhashes[cid])), 4),
                "topic": self.metadata[cid].get("topic"),
            }
            for cid in candidate_ids
        ]
        scored.sort(key=lambda d: -d["similarity"])
        return scored[:top_k]

    def feed_by_topic(self, topic: str, exclude_duplicates: bool = True) -> list:
        """Returns stored article ids/metadata matching a topic. When
        exclude_duplicates is True, only the first-seen article in each
        near-duplicate cluster is returned."""
        matches = [
            (aid, meta) for aid, meta in self.metadata.items()
            if meta.get("topic") == topic
        ]

        if not exclude_duplicates:
            return [{"article_id": aid, **meta} for aid, meta in matches]

        seen_ids = set()
        result = []
        for aid, meta in matches:
            if aid in seen_ids:
                continue
            result.append({"article_id": aid, **meta})
            similar = self.lsh.query(self.minhashes[aid])
            seen_ids.update(similar)
        return result

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "minhashes": self.minhashes,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.metadata = data["metadata"]
        self.minhashes = data["minhashes"]
        self.lsh = MinHashLSH(threshold=JACCARD_SIMILARITY_THRESHOLD, num_perm=NUM_PERM)
        for aid, mh in self.minhashes.items():
            self.lsh.insert(aid, mh)
