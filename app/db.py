"""SQLite persistence (stdlib ``sqlite3``, no ORM).

Two tables, matching the proposal:

    patients(id, name, created_at)
    records(id, patient_id, session_date, note_text, fields_json,
            entities_json, negated_entities_json, model_version, created_at)

``fields_json`` keeps the record schema forward-compatible: a later dashboard
can read old rows without a migration when a field is added.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .schemas import ClinicalRecord, EntityOut

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id            INTEGER REFERENCES patients(id) ON DELETE SET NULL,
    session_date          TEXT,
    note_text             TEXT NOT NULL,
    fields_json           TEXT NOT NULL,
    entities_json         TEXT NOT NULL DEFAULT '[]',
    negated_entities_json TEXT NOT NULL DEFAULT '[]',
    model_version         TEXT,
    created_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_patient ON records(patient_id);
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
"""


def db_path() -> Path:
    override = os.environ.get("CNA_DB_PATH")
    if override:
        path = Path(override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """One short-lived connection per request.

    FastAPI runs sync endpoints in a worker thread, so a process-wide
    connection would be unsafe.  Opening SQLite is cheap and WAL keeps the
    reader/writer overlap comfortable for a single-clinician prototype.
    """
    conn = sqlite3.connect(db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
            conn.execute("INSERT INTO schema_version(version) VALUES (1)")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _preview(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _patient_id(conn: sqlite3.Connection, patient_id: Optional[int], patient_name: Optional[str]) -> Optional[int]:
    if patient_id is not None:
        row = conn.execute("SELECT id FROM patients WHERE id = ?", (patient_id,)).fetchone()
        if row is None:
            raise ValueError(f"patient {patient_id} does not exist")
        return patient_id
    name = (patient_name or "").strip()
    if not name:
        return None
    row = conn.execute(
        "SELECT id FROM patients WHERE name = ? COLLATE NOCASE ORDER BY id LIMIT 1",
        (name,),
    ).fetchone()
    if row is not None:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO patients(name, created_at) VALUES (?, ?)",
        (name, _now()),
    )
    return int(cur.lastrowid)


def create_record(
    *,
    text: str,
    fields: ClinicalRecord,
    entities: list[EntityOut],
    negated_entities: list[EntityOut],
    patient_id: Optional[int] = None,
    patient_name: Optional[str] = None,
    session_date: Optional[str] = None,
    model_version: Optional[str] = None,
) -> tuple[int, Optional[int]]:
    with connect() as conn:
        pid = _patient_id(conn, patient_id, patient_name)
        created = _now()
        cur = conn.execute(
            """
            INSERT INTO records(
                patient_id, session_date, note_text, fields_json,
                entities_json, negated_entities_json, model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                session_date or date.today().isoformat(),
                text,
                fields.model_dump_json(),
                json.dumps([e.model_dump() for e in entities], ensure_ascii=False),
                json.dumps([e.model_dump() for e in negated_entities], ensure_ascii=False),
                model_version,
                created,
            ),
        )
        return int(cur.lastrowid), pid


def _summary(row: sqlite3.Row) -> dict:
    fields = json.loads(row["fields_json"])
    pain = fields.get("pain_severity") or {}
    patient_name = row["patient_name"] if "patient_name" in row.keys() else None
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "patient_name": patient_name,
        "session_date": row["session_date"],
        "body_part": fields.get("body_part"),
        "pain_severity": pain.get("value"),
        "preview": _preview(row["note_text"]),
        "created_at": row["created_at"],
    }


_SELECT = """
SELECT r.*, p.name AS patient_name
FROM records r
LEFT JOIN patients p ON p.id = r.patient_id
"""


def list_records(patient_id: Optional[int] = None, limit: int = 200) -> list[dict]:
    with connect() as conn:
        if patient_id is None:
            rows = conn.execute(
                _SELECT + " ORDER BY r.created_at DESC, r.id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                _SELECT + " WHERE r.patient_id = ? ORDER BY r.created_at DESC, r.id DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
    return [_summary(row) for row in rows]


def get_record(record_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(_SELECT + " WHERE r.id = ?", (record_id,)).fetchone()
    if row is None:
        return None
    out = _summary(row)
    out.update(
        {
            "text": row["note_text"],
            "fields": json.loads(row["fields_json"]),
            "entities": json.loads(row["entities_json"]),
            "negated_entities": json.loads(row["negated_entities_json"]),
            "model_version": row["model_version"],
        }
    )
    return out


def delete_record(record_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
    return cur.rowcount > 0


def count_records() -> int:
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])


def export_rows() -> list[dict]:
    """Flat rows for the optional CSV export."""
    rows = list_records(limit=100_000)
    out = []
    for row in rows:
        full = get_record(row["id"])
        if full is None:
            continue
        out.append(
            {
                "id": full["id"],
                "patient_name": full.get("patient_name"),
                "session_date": full.get("session_date"),
                "body_part": full["fields"].get("body_part"),
                "laterality": full["fields"].get("laterality"),
                "pain_severity": row["pain_severity"],
                "created_at": full["created_at"],
            }
        )
    return out


__all__ = [
    "init_db",
    "connect",
    "create_record",
    "list_records",
    "get_record",
    "delete_record",
    "count_records",
    "export_rows",
    "db_path",
]
