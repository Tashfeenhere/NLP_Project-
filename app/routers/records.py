"""Record persistence endpoints."""
from __future__ import annotations

import csv
import io
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from .. import db
from ..schemas import (
    RecordCreate,
    RecordCreated,
    RecordOut,
    RecordSummary,
)

router = APIRouter(tags=["records"])


@router.post("/records", response_model=RecordCreated, status_code=201)
def create_record(payload: RecordCreate) -> RecordCreated:
    try:
        record_id, patient_id = db.create_record(
            text=payload.text,
            fields=payload.fields,
            entities=payload.entities,
            negated_entities=payload.negated_entities,
            patient_id=payload.patient_id,
            patient_name=payload.patient_name,
            session_date=payload.session_date,
            model_version=payload.model_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RecordCreated(id=record_id, patient_id=patient_id)


@router.get("/records", response_model=list[RecordSummary])
def list_records(patient_id: Optional[int] = Query(default=None, ge=1)) -> list[RecordSummary]:
    return [RecordSummary.model_validate(row) for row in db.list_records(patient_id)]


@router.get("/records/export.csv")
def export_csv() -> StreamingResponse:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "id",
            "patient_name",
            "session_date",
            "body_part",
            "laterality",
            "pain_severity",
            "created_at",
        ],
    )
    writer.writeheader()
    writer.writerows(db.export_rows())
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="confirmed-records.csv"'},
    )


@router.get("/records/{record_id}", response_model=RecordOut)
def get_record(record_id: int) -> RecordOut:
    record = db.get_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    return RecordOut.model_validate(record)


@router.delete("/records/{record_id}", status_code=204)
def delete_record(record_id: int) -> None:
    if not db.delete_record(record_id):
        raise HTTPException(status_code=404, detail="record not found")
