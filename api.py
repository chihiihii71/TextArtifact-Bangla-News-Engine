"""
Bangla News Topic Engine — API
================================
Wraps the trained pipeline (predict.py / classify_article) in a small
FastAPI service so it's callable over HTTP from a frontend, another app,
or curl/Postman — the actual "product" interface.

RUN
---
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Then POST to http://localhost:8000/classify

TEST
----
    curl -X POST http://localhost:8000/classify \
      -H "Content-Type: application/json" \
      -d '{"text": "আপনার বাংলা আর্টিকেলের টেক্সট এখানে..."}'

Interactive docs (auto-generated) are at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
import os

from predict import classify_article
from duplicate_detector import ArticleStore

app = FastAPI(
    title="Bangla News Intelligence Engine",
    description="Classifies Bangla news articles into topics, detects near-duplicate/syndicated content, and serves a de-duplicated topic feed.",
    version="2.0.0",
)

STORE_PATH = "model/article_store.pkl"
store = ArticleStore()
if os.path.exists(STORE_PATH):
    store.load(STORE_PATH)


@app.get("/")
def demo_page():
    """Serves the built-in browser demo page."""
    return FileResponse("static/index.html")


class ArticleRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw Bangla article text to classify")


class ArticleResponse(BaseModel):
    predicted_class: str | None
    cluster_id: int | None
    distance_margin: float | None = None
    confidence_note: str | None = None


class IngestRequest(BaseModel):
    article_id: str = Field(..., description="Unique id for this article")
    text: str = Field(..., min_length=1)
    topic: Optional[str] = Field(None, description="If omitted, auto-classified before storing")


class DuplicateMatch(BaseModel):
    article_id: str
    similarity: float
    topic: Optional[str] = None


class IngestResponse(BaseModel):
    added: bool
    topic: Optional[str] = None
    duplicates: List[DuplicateMatch] = []
    reason: Optional[str] = None


class FeedItem(BaseModel):
    article_id: str
    topic: Optional[str] = None
    text_preview: Optional[str] = None


@app.get("/health")
def health_check():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/classify", response_model=ArticleResponse)
def classify(request: ArticleRequest):
    """Classify a single Bangla news article into a topic (no storage)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")

    try:
        result = classify_article(request.text)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run train_pipeline.py first to generate model/bangla_topic_model.joblib.",
        )

    return result


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """Auto-tag classification tool" for newsrooms: classify (if topic not
    given), check for near-duplicates against everything stored so far,
    and store the article for future duplicate checks / feeds."""
    topic = request.topic
    if topic is None:
        try:
            classification = classify_article(request.text)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run train_pipeline.py first.",
            )
        topic = classification.get("predicted_class")

    result = store.add_article(request.article_id, request.text, topic=topic)
    result["topic"] = topic
    store.save(STORE_PATH)
    return result


@app.get("/feed", response_model=List[FeedItem])
def feed(topic: str, exclude_duplicates: bool = True):
    """Personal news filter: return stored articles matching a topic,
    with near-duplicates collapsed by default."""
    return store.feed_by_topic(topic, exclude_duplicates=exclude_duplicates)
