"""
CSV ingestion pipeline.
Scans data/ for CSVs → detects columns → normalises → deduplicates → tags → saves.
"""
import json
import os
import re
import time
from pathlib import Path

import anthropic
import chardet
import pandas as pd
from fuzzywuzzy import fuzz

from .db import get_conn, init_db, upsert_member, link_member_event
from .tagger import tag_members

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))

def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Create a .env file in the project root with: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)

# ── Encoding / delimiter detection ───────────────────────────────────────────

def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:20_000]
    return chardet.detect(raw).get("encoding") or "utf-8"


def _detect_delimiter(path: Path, enc: str) -> str:
    try:
        sample = path.read_text(encoding=enc, errors="replace")[:3000]
    except Exception:
        sample = path.read_text(encoding="latin-1", errors="replace")[:3000]
    counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
    return max(counts, key=counts.get)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    enc = _detect_encoding(path)
    delim = _detect_delimiter(path, enc)
    for attempt_enc in [enc, "utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                path, encoding=attempt_enc, sep=delim,
                dtype=str, on_bad_lines="skip", nrows=10_000
            )
            df = df.fillna("").apply(lambda col: col.str.strip())
            # Drop completely empty columns
            df = df.loc[:, df.any()]
            return list(df.columns), df.to_dict("records")
        except Exception:
            continue
    return [], []

# ── Column mapping via Claude ─────────────────────────────────────────────────

CANONICAL = (
    "name first_name last_name email company title "
    "linkedin twitter instagram facebook tiktok youtube "
    "website phone location city country "
    "bio industry funding_stage notes "
    "event_name event_date tags"
).split()

# Heuristic keyword → canonical field (used as fallback when Claude unavailable)
_HEURISTIC: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bfirst.?name\b|\bfirst\b|\bgiven.?name\b|\bfname\b", re.I), "first_name"),
    (re.compile(r"\blast.?name\b|\blast\b|\bsurname\b|\bfamily.?name\b|\blname\b", re.I), "last_name"),
    (re.compile(r"\bfull.?name\b|\bname\b|\battendee\b|\bparticipant\b", re.I), "name"),
    (re.compile(r"\be.?mail\b|\bemail.?address\b|\bcontact.?email\b", re.I), "email"),
    (re.compile(r"\bcompany\b|\borganization\b|\borganisation\b|\bemployer\b|\bfirm\b|\bwork.?place\b", re.I), "company"),
    (re.compile(r"\btitle\b|\bjob.?title\b|\brole\b|\bposition\b|\bdesignation\b|\boccupation\b", re.I), "title"),
    (re.compile(r"\blinkedin\b", re.I), "linkedin"),
    (re.compile(r"\btwitter\b|\bx\.com\b|\bx handle\b", re.I), "twitter"),
    (re.compile(r"\binstagram\b|\big\b|\big handle\b", re.I), "instagram"),
    (re.compile(r"\bfacebook\b|\bfb\b", re.I), "facebook"),
    (re.compile(r"\btiktok\b|\btik.?tok\b", re.I), "tiktok"),
    (re.compile(r"\byoutube\b|\byt\b|\byoutube channel\b", re.I), "youtube"),
    (re.compile(r"\bwebsite\b|\burl\b|\bweb\b|\bhomepage\b", re.I), "website"),
    (re.compile(r"\bphone\b|\bmobile\b|\bcell\b|\btelephone\b|\btel\b", re.I), "phone"),
    (re.compile(r"\blocation\b|\baddress\b|\bregion\b", re.I), "location"),
    (re.compile(r"\bcity\b|\btown\b", re.I), "city"),
    (re.compile(r"\bcountry\b|\bnation\b", re.I), "country"),
    (re.compile(r"\bbio\b|\babout\b|\bdescription\b|\bintro\b|\bsummary\b", re.I), "bio"),
    (re.compile(r"\bindustry\b|\bsector\b|\bvertical\b", re.I), "industry"),
    (re.compile(r"\bfunding.?stage\b|\bstage\b|\bround\b|\bseries\b", re.I), "funding_stage"),
    (re.compile(r"\bevent.?name\b|\bevent\b|\bsession\b", re.I), "event_name"),
    (re.compile(r"\bevent.?date\b|\bdate\b|\bregistered\b", re.I), "event_date"),
    (re.compile(r"\btags?\b|\binterests?\b|\bcategory\b|\bcategories\b", re.I), "tags"),
    (re.compile(r"\bnotes?\b|\bcomments?\b|\bremarks?\b", re.I), "notes"),
]

def _heuristic_map(headers: list[str]) -> dict:
    """Keyword-based fallback when Claude is unavailable."""
    mapping: dict[str, str] = {}
    assigned: set[str] = set()
    for h in headers:
        for pattern, canonical in _HEURISTIC:
            if canonical not in assigned and pattern.search(h):
                mapping[h] = canonical
                assigned.add(canonical)
                break
    return mapping


def _batch_map_headers(header_sets: list[list[str]]) -> list[dict]:
    """Map batches of CSV headers → canonical fields using Claude (one API call).
    Falls back to heuristic mapping if Claude is unavailable."""
    # Deduplicate
    unique: list[tuple] = []
    seen: dict[tuple, int] = {}
    for hs in header_sets:
        key = tuple(h.lower().strip() for h in hs)
        if key not in seen:
            seen[key] = len(unique)
            unique.append(hs)

    prompt = f"""Map CSV column headers to canonical fields.

Canonical fields: {', '.join(CANONICAL)}
Use null for headers that don't match any canonical field.
Multiple headers can map to the same canonical field (e.g. "First" → first_name, "Last" → last_name).

Header sets (one per CSV file):
{json.dumps(unique, indent=2)}

Return ONLY a JSON array with one mapping object per header set.
Example: [{{"First Name": "first_name", "E-mail": "email", "Job Title": "title"}}]"""

    try:
        cl = _get_client()
        resp = cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
            text = m.group(1) if m else text
        mappings = json.loads(text)
        # Expand back to all header_sets (including duplicates)
        result = []
        for hs in header_sets:
            key = tuple(h.lower().strip() for h in hs)
            idx = seen[key]
            result.append(mappings[idx] if idx < len(mappings) else {})
        return result
    except Exception as e:
        print(f"[ingest] Claude header mapping failed ({e}); using heuristic fallback")
        return [_heuristic_map(hs) for hs in header_sets]

# ── Row normalisation ─────────────────────────────────────────────────────────

def _apply_mapping(row: dict, mapping: dict) -> dict:
    """Apply column mapping to a raw CSV row, returning a normalised member dict.
    Columns not mapped to a canonical field are stored in extra_fields."""
    out: dict = {}
    extra_fields: dict = {}
    mapped_orig_cols = set(mapping.keys())

    for orig_col, canonical in mapping.items():
        if not canonical or canonical not in CANONICAL:
            # Not a recognised canonical field — save as extra
            val = row.get(orig_col, "").strip()
            if val:
                extra_fields[orig_col] = val
            continue
        val = row.get(orig_col, "").strip()
        if not val:
            continue
        if canonical in out and len(out[canonical]) >= len(val):
            continue
        out[canonical] = val

    # Any column not mentioned in the mapping at all → also save as extra
    for col, val in row.items():
        if col not in mapped_orig_cols:
            val = str(val).strip() if val else ""
            if val:
                extra_fields[col] = val

    if extra_fields:
        out["extra_fields"] = extra_fields

    # Combine first_name + last_name → name
    if "name" not in out:
        parts = [out.pop("first_name", ""), out.pop("last_name", "")]
        full = " ".join(p for p in parts if p)
        if full:
            out["name"] = full
    else:
        out.pop("first_name", None)
        out.pop("last_name", None)

    # Normalise email
    if "email" in out:
        out["email"] = out["email"].lower()

    # Normalise linkedin
    if "linkedin" in out:
        li = out["linkedin"].strip()
        if li.startswith("http"):
            out["linkedin"] = li.replace("http://", "https://")
        elif li:
            li = re.sub(r"^(www\.)?linkedin\.com/in/", "", li, flags=re.I)
            handle = li.lstrip("@").strip("/")
            out["linkedin"] = f"https://linkedin.com/in/{handle}" if handle else ""

    return out


def _name_company_key(m: dict) -> str:
    name = re.sub(r"\s+", " ", (m.get("name") or "").lower().strip())
    co   = re.sub(r"\s+", " ", (m.get("company") or "").lower().strip())
    return f"{name}||{co}"


def _deduplicate(members: list[dict]) -> list[dict]:
    """Merge duplicates by email, then by fuzzy name+company."""
    by_email: dict[str, dict] = {}
    no_email: list[dict] = []

    for m in members:
        email = (m.get("email") or "").strip().lower()
        if email:
            if email in by_email:
                from .db import _merge
                by_email[email] = _merge(by_email[email], m)
            else:
                by_email[email] = dict(m)
        else:
            no_email.append(dict(m))

    merged = list(by_email.values())

    for m in no_email:
        name = (m.get("name") or "").lower()
        co   = (m.get("company") or "").lower()
        best_score = 0
        best_idx = -1
        for i, ex in enumerate(merged):
            if not ex.get("name"):
                continue
            name_score = fuzz.ratio(name, ex.get("name", "").lower())
            co_score   = fuzz.ratio(co,   ex.get("company", "").lower())
            # Both must be high for a confident merge
            if name_score >= 85 and co_score >= 75:
                combined = (name_score + co_score) / 2
                if combined > best_score:
                    best_score = combined
                    best_idx = i
        if best_idx >= 0:
            from .db import _merge
            merged[best_idx] = _merge(merged[best_idx], m)
        else:
            merged.append(m)

    return merged

# ── Main ingest ───────────────────────────────────────────────────────────────

def ingest_all(progress_cb=None) -> dict:
    """
    Full pipeline: scan data/ → parse CSVs → normalize → deduplicate → tag → save.
    progress_cb(step: str, current: int, total: int)
    """
    init_db()

    csv_files = sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.CSV"))
    if not csv_files:
        return {"status": "no_files", "message": "No CSV files found in data/"}

    if progress_cb:
        progress_cb("Reading CSVs", 0, len(csv_files))

    # Step 1: Read all CSVs
    parsed: list[tuple[Path, list[str], list[dict]]] = []
    for i, path in enumerate(csv_files):
        headers, rows = _read_csv(path)
        if headers and rows:
            parsed.append((path, headers, rows))
        if progress_cb:
            progress_cb("Reading CSVs", i + 1, len(csv_files))

    if not parsed:
        return {"status": "error", "message": "Could not read any CSV files"}

    # Step 2: Map all headers in batches of 30 (one Claude call per batch)
    if progress_cb:
        progress_cb("Mapping columns", 0, len(parsed))

    all_headers = [headers for _, headers, _ in parsed]
    mappings: list[dict] = []
    batch_size = 30
    for start in range(0, len(all_headers), batch_size):
        batch = all_headers[start:start + batch_size]
        mappings.extend(_batch_map_headers(batch))
        if progress_cb:
            progress_cb("Mapping columns", min(start + batch_size, len(parsed)), len(parsed))

    # Step 3: Normalise all rows
    if progress_cb:
        progress_cb("Normalising rows", 0, len(parsed))

    all_members: list[dict] = []
    event_info: dict[str, dict] = {}  # path → {name, date}

    for i, ((path, headers, rows), mapping) in enumerate(zip(parsed, mappings)):
        # Try to extract event name/date from mapping
        sample_row = rows[0] if rows else {}
        sample_mapped = _apply_mapping(sample_row, mapping)
        event_name = (
            sample_mapped.get("event_name")
            or path.stem.replace("_", " ").replace("-", " ").title()
        )
        event_date = sample_mapped.get("event_date", "")
        event_info[str(path)] = {"name": event_name, "date": event_date, "path": str(path)}

        for row in rows:
            m = _apply_mapping(row, mapping)
            if not m.get("name") and not m.get("email"):
                continue
            # Fallback name from email
            if not m.get("name") and m.get("email"):
                m["name"] = m["email"].split("@")[0].replace(".", " ").title()
            m["_source_path"] = str(path)
            m["_event_name"] = event_name
            all_members.append(m)

        if progress_cb:
            progress_cb("Normalising rows", i + 1, len(parsed))

    if not all_members:
        return {"status": "error", "message": "No valid member rows extracted"}

    # Step 4: Deduplicate
    if progress_cb:
        progress_cb("Deduplicating", 0, 1)
    unique_members = _deduplicate(all_members)
    if progress_cb:
        progress_cb("Deduplicating", 1, 1)

    # Step 5: Tag
    if progress_cb:
        progress_cb("Tagging members", 0, len(unique_members))
    tagged = tag_members(unique_members)
    if progress_cb:
        progress_cb("Tagging members", len(tagged), len(tagged))

    # Step 6: Save to DB
    if progress_cb:
        progress_cb("Saving to database", 0, len(tagged))

    conn = get_conn()
    event_id_map: dict[str, int] = {}

    for path_str, info in event_info.items():
        cur = conn.execute(
            "INSERT OR IGNORE INTO events (name, date, source_file) VALUES (?,?,?)",
            (info["name"], info.get("date"), info["path"])
        )
        if cur.lastrowid:
            event_id_map[path_str] = cur.lastrowid
        else:
            row = conn.execute(
                "SELECT id FROM events WHERE source_file=?", (info["path"],)
            ).fetchone()
            if row:
                event_id_map[path_str] = row["id"]

    saved = 0
    for i, member in enumerate(tagged):
        source_path = member.pop("_source_path", None)
        member.pop("_event_name", None)
        try:
            mid = upsert_member(conn, member)
            eid = event_id_map.get(source_path) if source_path else None
            if eid:
                link_member_event(conn, mid, eid)
            saved += 1
        except Exception as e:
            pass  # skip bad rows silently

        if progress_cb and i % 500 == 0:
            progress_cb("Saving to database", i, len(tagged))

    conn.commit()
    conn.close()

    return {
        "status": "ok",
        "files_processed": len(parsed),
        "raw_rows": len(all_members),
        "unique_members": len(tagged),
        "saved": saved,
    }
