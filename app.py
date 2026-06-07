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

# Global Search Engine Instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading dataset and initializing BM25 Engine...")
    # Load from cache to make startup extremely fast
    df = load_and_preprocess("indonesian_movies.csv")
    engine = MovieSearchEngine(df)
    print("BM25 Engine is ready!")
    yield
    # Cleanup resources if needed on shutdown
    print("Shutting down engine...")

app = FastAPI(title="Indonesian Movie Search Engine", lifespan=lifespan)

# Setup statics and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Menampilkan halaman utama Dashboard (Web UI).
    """
    return templates.TemplateResponse("index.html", {"request": request})

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
