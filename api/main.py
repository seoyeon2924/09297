"""
방송 심의 AI Agent — FastAPI 백엔드 서버.

실행:
    uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from storage.database import init_db
from api.routes.review import router as review_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 DB 초기화."""
    init_db()
    yield


app = FastAPI(
    title="방송 심의 AI Agent API",
    description="LangGraph 기반 멀티 에이전트 방송 심의 시스템 REST API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — Streamlit(localhost:8501)에서 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(review_router, prefix="/api", tags=["review"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "방송 심의 AI Agent API"}
