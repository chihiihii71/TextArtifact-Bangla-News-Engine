"""
Global & National Crisis Monitor — API
======================================
Tracks and serves the single most concerning and important news article 
across 6 major languages: Bangla, English, Spanish, German, French, and Arabic.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os

from multilingual_focus import global_focus_engine

app = FastAPI(
    title="Global & National Crisis Monitor",
    description="Monitors and isolates the highest-priority national and global crisis articles across 6 languages.",
    version="3.0.0",
)

# Mount the static frontend directory if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def demo_page():
    """Serves the real-time crisis monitor dashboard UI."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"error": "Dashboard UI not found. Please ensure static/index.html is created."}


class ArticleInput(BaseModel):
    id: str = Field(..., description="Unique ID for the article")
    title: str = Field(..., min_length=1, description="Article headline")
    text: str = Field(..., min_length=1, description="Article body content")
    language: str = Field(..., description="Language code: bn, en, es, de, fr, or ar")


@app.post("/ingest-global")
def ingest_global_article(article: ArticleInput):
    """
    Ingests an article in any of the 6 supported languages, evaluates its 
    threat/urgency score, and updates the top concerning crisis feed for that region.
    """
    supported_langs = ["bn", "en", "es", "de", "fr", "ar"]
    if article.language not in supported_langs:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Choose from: {supported_langs}"
        )

    global_focus_engine.evaluate_article(
        article_id=article.id,
        title=article.title,
        text=article.text,
        language=article.language
    )
    
    return {
        "status": "success",
        "message": f"Article evaluated and processed for language: {article.language}"
    }


@app.get("/top-focus-all")
def get_all_top_focus():
    """
    Returns the single most concerning and critical article 
    for all 6 languages concurrently.
    """
    return {
        "status": "success",
        "languages_tracked": ["bn", "en", "es", "de", "fr", "ar"],
        "top_articles": global_focus_engine.get_all_top_focus()
    }


@app.get("/health")
def health_check():
    """Simple liveness check for Render deployment."""
    return {"status": "ok"}