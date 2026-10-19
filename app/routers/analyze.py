"""``POST /api/analyze`` -- run the clinical pipeline over a note."""
from __future__ import annotations

from fastapi import APIRouter

from nlp.pipeline import get_pipeline

from ..schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze one note.

    ``def`` (not ``async def``) is deliberate: FastAPI runs it in the worker
    threadpool, so the CPU-bound spaCy pass never blocks the event loop.  The
    pipeline's own lock serialises concurrent inference.
    """
    result = get_pipeline().analyze(request.text)
    return AnalyzeResponse.model_validate(result)
