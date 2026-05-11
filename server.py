"""
Community Intelligence Platform — FastAPI backend
Run: python server.py
"""
import csv
import io
import json
import os
import threading
from pathlib import Path
from typing import Optional

# Load .env if present (before any Anthropic client is created)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db import init_db, get_conn, get_stats
from app.ingest import ingest_all, DATA_DIR
from app.search import search, simple_filter

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Community Intelligence Platform", version="1.0.0")
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR.mkdir(parents=True, exist_ok=True)
init_db()

# ── Ingest state (in-memory progress tracker) ─────────────────────────────────

_ingest_state = {
    "running": False,
    "step": "",
    "current": 0,
    "total": 0,
    "result": None,
    "error": None,
}
_ingest_lock = threading.Lock()


def _run_ingest():
    def progress(step, current, total):
        with _ingest_lock:
            _ingest_state["step"] = step
            _ingest_state["current"] = current
            _ingest_state["total"] = total

    try:
        result = ingest_all(progress_cb=progress)
        with _ingest_lock:
            _ingest_state["result"] = result
            _ingest_state["error"] = None
    except Exception as e:
        with _ingest_lock:
            _ingest_state["error"] = str(e)
            _ingest_state["result"] = None
    finally:
        with _ingest_lock:
            _ingest_state["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    return get_stats()


@app.post("/api/ingest/start")
def start_ingest():
    with _ingest_lock:
        if _ingest_state["running"]:
            return {"status": "already_running"}
        _ingest_state.update(
            running=True, step="Starting", current=0, total=0,
            result=None, error=None
        )
    thread = threading.Thread(target=_run_ingest, daemon=True)
    thread.start()
    return {"status": "started"}


@app.get("/api/ingest/status")
def ingest_status():
    with _ingest_lock:
        return dict(_ingest_state)


@app.post("/api/upload")
async def upload_csvs(files: list[UploadFile] = File(...)):
    """Accept CSV file uploads and save them to the data directory."""
    saved = []
    errors = []
    for f in files:
        if not f.filename:
            continue
        name = Path(f.filename).name
        if not name.lower().endswith((".csv",)):
            errors.append(f"{name}: not a CSV")
            continue
        dest = DATA_DIR / name
        try:
            content = await f.read()
            dest.write_bytes(content)
            saved.append(name)
        except Exception as e:
            errors.append(f"{name}: {e}")
    return {"saved": saved, "errors": errors, "data_dir": str(DATA_DIR)}


@app.get("/api/upload/list")
def list_uploads():
    """List CSV files currently in the data directory."""
    files = []
    for p in sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.CSV")):
        files.append({"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)})
    return files


@app.delete("/api/upload/{filename}")
def delete_upload(filename: str):
    """Delete a CSV from the data directory."""
    p = DATA_DIR / filename
    if not p.exists() or p.suffix.lower() != ".csv":
        raise HTTPException(status_code=404, detail="File not found")
    p.unlink()
    return {"deleted": filename}


class SearchRequest(BaseModel):
    query: str = ""
    member_type: Optional[str] = None
    tag: Optional[str] = None
    min_events: Optional[int] = None
    has_email: Optional[bool] = None
    limit: int = 50
    offset: int = 0


@app.post("/api/search")
def do_search(req: SearchRequest):
    """
    If query is a natural-language string → Claude NLP path.
    Otherwise → fast structured filter path.
    """
    try:
        if req.query.strip():
            override = {}
            if req.member_type and req.member_type != "all":
                override["member_types"] = [req.member_type]
            if req.tag:
                override["tags"] = [req.tag]
            if req.min_events:
                override["min_events"] = req.min_events
            if req.has_email is not None:
                override["has_email"] = req.has_email
            override["limit"] = req.limit
            return search(req.query, override)
        else:
            return simple_filter(
                member_type=req.member_type,
                tag=req.tag,
                min_events=req.min_events,
                has_email=req.has_email,
                limit=req.limit,
                offset=req.offset,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/members/{member_id}")
def get_member(member_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM members WHERE id=?", (member_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Member not found")
    d = dict(row)
    try:
        d["tags"] = json.loads(d.get("tags") or "[]")
    except Exception:
        d["tags"] = []
    # Fetch attended events
    events = conn.execute("""
        SELECT e.name, e.date FROM member_events me
        JOIN events e ON me.event_id = e.id
        WHERE me.member_id = ?
        ORDER BY e.date DESC
    """, (member_id,)).fetchall()
    d["events"] = [dict(e) for e in events]
    conn.close()
    return d


@app.get("/api/events")
def list_events():
    conn = get_conn()
    rows = conn.execute("""
        SELECT e.*, COUNT(me.member_id) as attendee_count
        FROM events e
        LEFT JOIN member_events me ON e.id = me.event_id
        GROUP BY e.id
        ORDER BY e.date DESC, e.name ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/export")
def export_csv(
    member_type: Optional[str] = None,
    tag: Optional[str] = None,
    min_events: Optional[int] = None,
    has_email: Optional[bool] = None,
    query: Optional[str] = None,
):
    """Export filtered members as a CSV download."""
    try:
        if query:
            override = {"limit": 5000}
            if member_type and member_type != "all":
                override["member_types"] = [member_type]
            if tag:
                override["tags"] = [tag]
            if min_events:
                override["min_events"] = min_events
            if has_email is not None:
                override["has_email"] = has_email
            result = search(query, override)
            members = result["results"]
        else:
            result = simple_filter(
                member_type=member_type, tag=tag,
                min_events=min_events, has_email=has_email, limit=5000
            )
            members = result["results"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    FIELDS = ["name", "email", "company", "title", "member_type",
              "location", "linkedin", "twitter", "website", "event_count", "tags"]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for m in members:
        row = {k: m.get(k, "") for k in FIELDS}
        if isinstance(row["tags"], list):
            row["tags"] = ", ".join(row["tags"])
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),  # BOM for Excel
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=community_export.csv"},
    )


@app.get("/api/tags")
def list_tags():
    """Return all unique tags with counts."""
    conn = get_conn()
    rows = conn.execute("SELECT tags FROM members WHERE tags IS NOT NULL AND tags != '[]'").fetchall()
    conn.close()
    tag_counts: dict[str, int] = {}
    for row in rows:
        try:
            for t in json.loads(row["tags"]):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        except Exception:
            pass
    return sorted(
        [{"tag": k, "count": v} for k, v in tag_counts.items()],
        key=lambda x: -x["count"]
    )


# ── Serve frontend ────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
