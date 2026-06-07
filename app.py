from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import custom modules
from dataset import load_and_preprocess
from search_engine import MovieSearchEngine
from evaluation import evaluate_system

from contextlib import asynccontextmanager
from fastapi.responses import PlainTextResponse
import traceback

# Global Search Engine Instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading dataset and initializing BM25 Engine...")
    print("WARNING: First boot may take 10-30 seconds to run NLTK NLP preprocessing...")
    # Load from cache to make startup extremely fast, sample to 1000 to prevent timeout
    df = load_and_preprocess("corpus.jsonl", sample_size=1000)
    engine = MovieSearchEngine(df)
    print("BM25 Engine is ready!")
    yield
    # Cleanup resources if needed on shutdown
    print("Shutting down engine...")

app = FastAPI(title="Research Paper Search Engine", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return PlainTextResponse(f"Internal Server Error: {str(exc)}\n\n{traceback.format_exc()}", status_code=500)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup statics and templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Menampilkan halaman utama Dashboard (Web UI).
    """
    return templates.TemplateResponse(request=request, name="index.html")

import time

@app.get("/api/search")
async def api_search(q: str = Query("", description="Kueri pencarian")):
    """
    Endpoint untuk mencari film menggunakan algoritma BM25.
    """
    if not q.strip():
        return {"query": q, "query_tokens": [], "execution_time_ms": 0, "results": []}
    
    start_time = time.time()
    # Ambil top 15 hasil
    search_data = engine.search(q, top_k=15)
    execution_time_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "query": q,
        "query_tokens": search_data.get('query_tokens', []),
        "execution_time_ms": execution_time_ms,
        "results": search_data.get('results', [])
    }

@app.get("/api/evaluate")
async def api_evaluate():
    """
    Endpoint untuk mengembalikan hasil evaluasi performa sistem.
    """
    # Memanggil script evaluate_system yang menggunakan Ground Truth 10 Kueri
    summary, details = evaluate_system(engine)
    return {
        "summary": summary,
        "details": details
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
