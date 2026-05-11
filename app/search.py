"""
Natural-language search powered by Claude tool use.
Claude converts a plain-English query into structured SQL filters,
which are executed against the SQLite + FTS5 database.
"""
import json
import os
import re
import sqlite3
from typing import Optional

import anthropic

from .db import get_conn

def _get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── Tool definition Claude uses to express a search ──────────────────────────

SEARCH_TOOL = {
    "name": "search_community",
    "description": (
        "Search the community member database. "
        "Use this to translate natural-language queries into structured filters."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text_query": {
                "type": "string",
                "description": (
                    "Free-text search across name, company, title, bio, location. "
                    "Leave empty if no keyword search needed."
                ),
            },
            "member_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["founder", "investor", "creator", "operator", "other", "unknown"],
                },
                "description": "Filter by member type. Empty means all types.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Filter to members whose tags include ANY of these. "
                    "Example: ['fintech', 'saas', 'ai/ml']"
                ),
            },
            "min_events": {
                "type": "integer",
                "minimum": 1,
                "description": "Only return members who attended at least this many events.",
            },
            "location_contains": {
                "type": "string",
                "description": "Case-insensitive substring match on the location field.",
            },
            "company_contains": {
                "type": "string",
                "description": "Case-insensitive substring match on company name.",
            },
            "title_contains": {
                "type": "string",
                "description": "Case-insensitive substring match on job title.",
            },
            "has_linkedin": {
                "type": "boolean",
                "description": "If true, only return members with a LinkedIn URL.",
            },
            "has_email": {
                "type": "boolean",
                "description": "If true, only return members with a known email address.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Max results to return. Default 50.",
            },
            "explanation": {
                "type": "string",
                "description": "One sentence explaining what this search finds and why.",
            },
        },
        "required": ["explanation"],
    },
}


# ── Build SQL from structured filters ────────────────────────────────────────

def _build_query(filters: dict) -> tuple[str, list]:
    """
    Convert Claude's structured filters into a SQL query + params.
    Returns (sql, params).
    """
    text_query      = (filters.get("text_query") or "").strip()
    member_types    = filters.get("member_types") or []
    tags            = filters.get("tags") or []
    min_events      = filters.get("min_events")
    location_like   = filters.get("location_contains")
    company_like    = filters.get("company_contains")
    title_like      = filters.get("title_contains")
    has_linkedin    = filters.get("has_linkedin")
    has_email       = filters.get("has_email")
    limit           = min(int(filters.get("limit") or 50), 200)

    params: list = []
    where: list[str] = []

    if text_query:
        # Use FTS5 for keyword search, join back to members
        base_select = """
            SELECT m.*, bm25(members_fts) AS relevance
            FROM members_fts
            JOIN members m ON members_fts.rowid = m.id
        """
        where.append("members_fts MATCH ?")
        # Convert to FTS5 prefix query (handles multi-word)
        fts_query = " ".join(f'"{w}"*' for w in text_query.split() if w)
        params.append(fts_query)
        order = "ORDER BY relevance"
    else:
        base_select = "SELECT m.*, 0.0 AS relevance FROM members m"
        order = "ORDER BY m.event_count DESC, m.name ASC"

    if member_types:
        placeholders = ",".join("?" * len(member_types))
        where.append(f"m.member_type IN ({placeholders})")
        params.extend(member_types)

    if tags:
        tag_conditions = " OR ".join("m.tags LIKE ?" for _ in tags)
        where.append(f"({tag_conditions})")
        params.extend(f"%{t}%" for t in tags)

    if min_events:
        where.append("m.event_count >= ?")
        params.append(min_events)

    if location_like:
        where.append("m.location LIKE ?")
        params.append(f"%{location_like}%")

    if company_like:
        where.append("m.company LIKE ?")
        params.append(f"%{company_like}%")

    if title_like:
        where.append("m.title LIKE ?")
        params.append(f"%{title_like}%")

    if has_linkedin:
        where.append("m.linkedin IS NOT NULL AND m.linkedin != ''")

    if has_email:
        where.append("m.email IS NOT NULL AND m.email != ''")

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"{base_select} {where_clause} {order} LIMIT ?"
    params.append(limit)

    return sql, params


# ── Claude NLP → filters ──────────────────────────────────────────────────────

def _nl_to_filters(query: str, stats: dict) -> dict:
    """Ask Claude to turn a natural-language query into structured filters."""

    system = f"""You are a search assistant for a community intelligence database.
The database has {stats.get('total', 0):,} members:
  - {stats.get('founders', 0):,} founders
  - {stats.get('investors', 0):,} investors
  - {stats.get('creators', 0):,} creators
  - {stats.get('operators', 0):,} operators

Use the search_community tool to express the user's query as structured filters.
Be generous — don't over-filter. When in doubt, include more results."""

    response = _get_client().messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=system,
        tools=[SEARCH_TOOL],
        tool_choice={"type": "tool", "name": "search_community"},
        messages=[{"role": "user", "content": query}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "search_community":
            return block.input

    return {"explanation": "General search", "text_query": query}


# ── Public search API ─────────────────────────────────────────────────────────

def search(query: str, filters_override: Optional[dict] = None) -> dict:
    """
    Main search entry point.

    If query is non-empty, uses Claude to convert it to filters.
    filters_override can add/replace any filter directly (used by UI filter sidebar).

    Returns:
      {
        "results": [...member dicts...],
        "total":   int,
        "filters": {...structured filters used...},
        "explanation": "..."
      }
    """
    conn = get_conn()
    stats_row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(member_type='founder')  as founders,
               SUM(member_type='investor') as investors,
               SUM(member_type='creator')  as creators,
               SUM(member_type='operator') as operators
        FROM members
    """).fetchone()
    stats = dict(stats_row) if stats_row else {}

    # Build filters
    if query.strip():
        filters = _nl_to_filters(query.strip(), stats)
    else:
        filters = {"explanation": "All members", "limit": 50}

    if filters_override:
        filters.update(filters_override)

    sql, params = _build_query(filters)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        # FTS5 query syntax error – fall back to LIKE search
        filters["text_query"] = ""
        sql, params = _build_query(filters)
        rows = conn.execute(sql, params).fetchall()

    conn.close()

    results = []
    for row in rows:
        d = dict(row)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except Exception:
            d["tags"] = []
        results.append(d)

    return {
        "results": results,
        "total": len(results),
        "filters": filters,
        "explanation": filters.get("explanation", ""),
    }


def simple_filter(
    member_type: Optional[str] = None,
    tag: Optional[str] = None,
    min_events: Optional[int] = None,
    has_email: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Fast structured filter without Claude (used by sidebar UI)."""
    conn = get_conn()
    where = []
    params: list = []

    if member_type and member_type != "all":
        where.append("member_type = ?")
        params.append(member_type)

    if tag:
        where.append("tags LIKE ?")
        params.append(f"%{tag}%")

    if min_events:
        where.append("event_count >= ?")
        params.append(min_events)

    if has_email:
        where.append("email IS NOT NULL AND email != ''")

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    total_row = conn.execute(
        f"SELECT COUNT(*) FROM members {where_clause}", params
    ).fetchone()
    total = total_row[0] if total_row else 0

    rows = conn.execute(
        f"SELECT * FROM members {where_clause} ORDER BY event_count DESC, name ASC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        d = dict(row)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except Exception:
            d["tags"] = []
        results.append(d)

    return {"results": results, "total": total}
