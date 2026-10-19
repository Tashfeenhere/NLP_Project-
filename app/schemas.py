"""Pydantic models shared by the API and the database layer."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_NOTE_CHARS = 20_000


class PainSeverity(BaseModel):
    raw: str
    value: Optional[int] = None
    range: Optional[list[int]] = None


class RomEntry(BaseModel):
    movement: str
    value: Optional[int] = None
    unit: Optional[str] = None
    quality: Optional[str] = None


class ClinicalRecord(BaseModel):
    """The structured fields a clinician reviews and confirms."""

    body_part: Optional[str] = None
    laterality: Optional[str] = None
    pain_severity: Optional[PainSeverity] = None
    rom: list[RomEntry] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    relieving_factors: list[str] = Field(default_factory=list)
    muscles: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    exercises: list[str] = Field(default_factory=list)
    duration: Optional[str] = None
    frequency: Optional[str] = None


class EntityOut(BaseModel):
    start: int
    end: int
    label: str
    text: str
    source: str
    normalized: str
    negated: bool = False
    laterality: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AnalyzeResponse(BaseModel):
    ready: bool
    warnings: list[str] = Field(default_factory=list)
    fields: ClinicalRecord
    entities: list[EntityOut] = Field(default_factory=list)
    negated_entities: list[EntityOut] = Field(default_factory=list)
    model_version: str


class RecordCreate(BaseModel):
    patient_id: Optional[int] = None
    patient_name: Optional[str] = Field(default=None, max_length=200)
    session_date: Optional[str] = Field(default=None, max_length=32)
    text: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)
    fields: ClinicalRecord
    entities: list[EntityOut] = Field(default_factory=list)
    negated_entities: list[EntityOut] = Field(default_factory=list)
    model_version: Optional[str] = None


class RecordSummary(BaseModel):
    id: int
    patient_id: Optional[int] = None
    patient_name: Optional[str] = None
    session_date: Optional[str] = None
    body_part: Optional[str] = None
    pain_severity: Optional[int] = None
    preview: str
    created_at: str


class RecordOut(RecordSummary):
    text: str
    fields: ClinicalRecord
    entities: list[EntityOut] = Field(default_factory=list)
    negated_entities: list[EntityOut] = Field(default_factory=list)
    model_version: Optional[str] = None


class RecordCreated(BaseModel):
    id: int
    patient_id: Optional[int] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_version: str
    labels: list[str]
    record_count: int
