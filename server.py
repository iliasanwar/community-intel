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

import secrets
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, UploadFile, File, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Auth ──────────────────────────────────────────────────────────────────────

APP_PASSWORD = os.environ.get("APP_PASSWORD", "Pleasedonothackthis")
COOKIE_NAME  = "ci_session"
# All valid session tokens (in-memory; cleared on restart — just re-login)
_sessions: set[str] = set()

def _check_auth(request: Request) -> bool:
    return request.cookies.get(COOKIE_NAME) in _sessions

LOGIN_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Rosturr</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    * {{ -webkit-font-smoothing: antialiased; }}
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif; }}
    input:focus {{ outline: none; border-color: #1d1d1f !important; box-shadow: 0 0 0 3px rgba(29,29,31,.08); }}
  </style>
</head>
<body class="bg-[#f5f5f7] min-h-screen flex items-center justify-center">
  <div class="bg-white border border-[#e5e7eb] rounded-2xl p-10 w-full max-w-sm shadow-sm">
    <div class="flex items-center gap-2.5 mb-8">
      <div class="w-8 h-8 rounded-lg bg-black flex items-center justify-center shrink-0">
        <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v1h8v-1zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-1a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v1h-3zM4.75 12.094A5.973 5.973 0 004 15v1H1v-1a3 3 0 013.75-2.906z"/></svg>
      </div>
      <span class="text-[#1d1d1f] font-semibold text-[17px] tracking-tight">Rosturr</span>
    </div>
    <form method="post" action="/auth/login">
      <label class="block text-[#6b7280] text-[12px] font-medium mb-1.5 uppercase tracking-wider">Password</label>
      <input name="password" type="password" autofocus
        class="w-full bg-[#f9fafb] border border-[#e5e7eb] text-[#1d1d1f] rounded-xl px-4 py-2.5 mb-3 text-[14px] transition-all"
        placeholder="Enter password" />
      {error}
      <button type="submit"
        class="w-full bg-[#1d1d1f] hover:bg-[#3a3a3c] text-white font-medium py-2.5 rounded-xl text-[14px] transition-colors">
        Sign in
      </button>
    </form>
  </div>
</body>
</html>"""

from app.db import init_db, get_conn, get_stats
from app.ingest import ingest_all, DATA_DIR
from app.search import search, simple_filter

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Community Intelligence Platform", version="1.0.0")
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR.mkdir(parents=True, exist_ok=True)
init_db()


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/auth/login", response_class=HTMLResponse, include_in_schema=False)
def login_page():
    return LOGIN_HTML.format(error="")

@app.post("/auth/login", response_class=HTMLResponse, include_in_schema=False)
async def do_login(request: Request):
    form = await request.form()
    if form.get("password") == APP_PASSWORD:
        token = secrets.token_hex(32)
        _sessions.add(token)
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=60*60*24*30)
        return resp
    return HTMLResponse(LOGIN_HTML.format(
        error='<p class="text-red-500 text-[12px] mb-3">Incorrect password, please try again.</p>'
    ))

@app.get("/auth/logout", include_in_schema=False)
def logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    _sessions.discard(token)
    resp = RedirectResponse("/auth/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


# ── Auth middleware (protects everything except /auth/*) ──────────────────────

from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/auth/"):
            return await call_next(request)
        if not _check_auth(request):
            return RedirectResponse("/auth/login")
        return await call_next(request)

app.add_middleware(AuthMiddleware)


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
        stat = p.stat()
        # Count rows quickly (line count minus header)
        try:
            with open(p, "rb") as f:
                row_count = max(0, sum(1 for _ in f) - 1)
        except Exception:
            row_count = 0
        files.append({
            "name": p.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "row_count": row_count,
            "modified": stat.st_mtime,
        })
    return files


@app.get("/api/upload/{filename}/preview")
def preview_csv(filename: str, limit: int = Query(default=500, le=5000)):
    """Return columns + rows of a CSV file for preview."""
    import chardet, pandas as pd
    p = DATA_DIR / Path(filename).name
    if not p.exists() or p.suffix.lower() != ".csv":
        raise HTTPException(status_code=404, detail="File not found")
    try:
        raw = p.read_bytes()[:20_000]
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        sample = p.read_text(encoding=enc, errors="replace")[:3000]
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        delim = max(counts, key=counts.get)
        df = pd.read_csv(p, encoding=enc, sep=delim, dtype=str,
                         on_bad_lines="skip", nrows=limit)
        df = df.fillna("").apply(lambda col: col.str.strip())
        df = df.loc[:, df.any()]
        return {
            "name": p.name,
            "columns": list(df.columns),
            "rows": df.head(limit).to_dict("records"),
            "total_rows": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    FIELDS = [
        "name", "email", "phone", "company", "title", "member_type",
        "industry", "funding_stage", "location", "city", "country",
        "linkedin", "twitter", "instagram", "facebook", "tiktok", "youtube",
        "website", "bio", "notes", "event_count", "tags"
    ]

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
