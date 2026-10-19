"""FastAPI application entry point.

    uvicorn app.main:app --reload

The spaCy pipeline is built once during the lifespan startup (it takes a few
seconds) and shared for the process's lifetime.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from nlp.pipeline import get_pipeline

from . import db
from .routers import analyze as analyze_router
from .routers import records as records_router
from .schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    pipeline = get_pipeline()
    app.state.pipeline = pipeline
    yield


app = FastAPI(
    title="Clinical Note Analyzer",
    version="1.0.0",
    description=(
        "Extracts structured physiotherapy fields from free-text notes. "
        "Assists documentation only -- it does not diagnose."
    ),
    lifespan=lifespan,
)

app.include_router(analyze_router.router, prefix="/api")
app.include_router(records_router.router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    pipeline = get_pipeline()
    return HealthResponse(
        model_version=pipeline.model_version,
        labels=pipeline.labels,
        record_count=db.count_records(),
    )
