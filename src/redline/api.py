"""FastAPI service: POST /review returns playbook-cited redlines as JSON."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .backends import make_backend
from .config import get_settings
from .pipeline import Reviewer
from .playbook import default_playbook

app = FastAPI(title="Contract Redline Copilot", version=__version__)


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500_000, description="Contract or clause text")
    include_other: bool = Field(False, description="Also return clauses outside the playbook")


@lru_cache(maxsize=1)
def get_reviewer() -> Reviewer:
    settings = get_settings()
    return Reviewer(backend=make_backend(settings), settings=settings)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "backend": get_settings().backend}


@app.get("/playbook")
def playbook() -> dict:
    pb = default_playbook()
    return {"version": pb.version, "rules": [r.to_dict() for r in pb.rules]}


@app.post("/review")
def review(req: ReviewRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")
    return get_reviewer().review_document(req.text, include_other=req.include_other)
