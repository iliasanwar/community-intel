import os
import sqlite3
import json
from pathlib import Path

# Allow Railway / Render to point DB at a persistent volume via env var
_default_db = Path(__file__).parent.parent / "community.db"
DB_PATH = Path(os.environ.get("DB_PATH", str(_default_db)))


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# All standard member columns (besides id, member_type, confidence, tags,
# event_count, created_at, updated_at)
MEMBER_FIELDS = [
    "name", "email", "company", "title",
    "linkedin", "twitter", "instagram", "facebook", "tiktok", "youtube",
    "website", "phone", "location", "city", "country",
    "bio", "industry", "funding_stage", "notes",
]


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            date        TEXT,
            source_file TEXT UNIQUE,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS members (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT NOT NULL,
            email          TEXT,
            company        TEXT,
            title          TEXT,
            linkedin       TEXT,
            twitter        TEXT,
            instagram      TEXT,
            facebook       TEXT,
            tiktok         TEXT,
            youtube        TEXT,
            website        TEXT,
            phone          TEXT,
            location       TEXT,
            city           TEXT,
            country        TEXT,
            bio            TEXT,
            industry       TEXT,
            funding_stage  TEXT,
            notes          TEXT,
            extra_fields   TEXT DEFAULT '{}',
            member_type    TEXT DEFAULT 'unknown',
            confidence     REAL DEFAULT 0.0,
            tags           TEXT DEFAULT '[]',
            event_count    INTEGER DEFAULT 0,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_members_email
            ON members(email) WHERE email IS NOT NULL AND email != '';

        CREATE INDEX IF NOT EXISTS idx_members_type ON members(member_type);
        CREATE INDEX IF NOT EXISTS idx_members_company ON members(company);

        CREATE TABLE IF NOT EXISTS member_events (
            member_id INTEGER REFERENCES members(id) ON DELETE CASCADE,
            event_id  INTEGER REFERENCES events(id)  ON DELETE CASCADE,
            PRIMARY KEY (member_id, event_id)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS members_fts USING fts5(
            name, company, title, location, bio, tags, industry, notes,
            content=members, content_rowid=id,
            tokenize='porter ascii'
        );

        CREATE TRIGGER IF NOT EXISTS members_fts_insert AFTER INSERT ON members BEGIN
            INSERT INTO members_fts(rowid, name, company, title, location, bio, tags, industry, notes)
            VALUES (new.id, new.name, COALESCE(new.company,''),
                    COALESCE(new.title,''), COALESCE(new.location,''),
                    COALESCE(new.bio,''), COALESCE(new.tags,''),
                    COALESCE(new.industry,''), COALESCE(new.notes,''));
        END;

        CREATE TRIGGER IF NOT EXISTS members_fts_update AFTER UPDATE ON members BEGIN
            INSERT INTO members_fts(members_fts, rowid, name, company, title, location, bio, tags, industry, notes)
            VALUES ('delete', old.id, old.name, COALESCE(old.company,''),
                    COALESCE(old.title,''), COALESCE(old.location,''),
                    COALESCE(old.bio,''), COALESCE(old.tags,''),
                    COALESCE(old.industry,''), COALESCE(old.notes,''));
            INSERT INTO members_fts(rowid, name, company, title, location, bio, tags, industry, notes)
            VALUES (new.id, new.name, COALESCE(new.company,''),
                    COALESCE(new.title,''), COALESCE(new.location,''),
                    COALESCE(new.bio,''), COALESCE(new.tags,''),
                    COALESCE(new.industry,''), COALESCE(new.notes,''));
        END;

        CREATE TRIGGER IF NOT EXISTS members_fts_delete AFTER DELETE ON members BEGIN
            INSERT INTO members_fts(members_fts, rowid, name, company, title, location, bio, tags, industry, notes)
            VALUES ('delete', old.id, old.name, COALESCE(old.company,''),
                    COALESCE(old.title,''), COALESCE(old.location,''),
                    COALESCE(old.bio,''), COALESCE(old.tags,''),
                    COALESCE(old.industry,''), COALESCE(old.notes,''));
        END;
    """)

    # Migrate existing DB: add any new columns that don't exist yet
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(members)")}
    new_cols = {
        "instagram":     "TEXT",
        "facebook":      "TEXT",
        "tiktok":        "TEXT",
        "youtube":       "TEXT",
        "phone":         "TEXT",
        "city":          "TEXT",
        "country":       "TEXT",
        "industry":      "TEXT",
        "funding_stage": "TEXT",
        "notes":         "TEXT",
        "extra_fields":  "TEXT DEFAULT '{}'",
    }
    for col, col_type in new_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE members ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


def get_stats() -> dict:
    conn = get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(member_type = 'founder')  as founders,
            SUM(member_type = 'investor') as investors,
            SUM(member_type = 'creator')  as creators,
            SUM(member_type = 'operator') as operators,
            SUM(member_type = 'other')    as other,
            SUM(member_type = 'unknown')  as unknown
        FROM members
    """).fetchone()
    events_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    return {**dict(row), "events": events_count}


def upsert_member(conn: sqlite3.Connection, member: dict) -> int:
    """Insert or merge member. Returns member id."""
    email = (member.get("email") or "").strip().lower()

    existing = None
    if email:
        existing = conn.execute(
            "SELECT * FROM members WHERE email = ?", (email,)
        ).fetchone()

    if existing:
        mid = existing["id"]
        merged = _merge(dict(existing), member)
        conn.execute("""
            UPDATE members SET
                name=?, company=?, title=?,
                linkedin=?, twitter=?, instagram=?, facebook=?, tiktok=?, youtube=?,
                website=?, phone=?, location=?, city=?, country=?,
                bio=?, industry=?, funding_stage=?, notes=?,
                extra_fields=?,
                member_type=?, confidence=?, tags=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (
            merged["name"], merged.get("company"), merged.get("title"),
            merged.get("linkedin"), merged.get("twitter"),
            merged.get("instagram"), merged.get("facebook"),
            merged.get("tiktok"), merged.get("youtube"),
            merged.get("website"), merged.get("phone"),
            merged.get("location"), merged.get("city"), merged.get("country"),
            merged.get("bio"), merged.get("industry"), merged.get("funding_stage"),
            merged.get("notes"),
            json.dumps(merged.get("extra_fields") or {}),
            merged.get("member_type", "unknown"),
            merged.get("confidence", 0.0),
            json.dumps(merged.get("tags", [])),
            mid
        ))
        return mid
    else:
        cur = conn.execute("""
            INSERT INTO members (
                name, email, company, title,
                linkedin, twitter, instagram, facebook, tiktok, youtube,
                website, phone, location, city, country,
                bio, industry, funding_stage, notes,
                extra_fields,
                member_type, confidence, tags
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?,
                ?, ?, ?
            )
        """, (
            member["name"], email or None,
            member.get("company"), member.get("title"),
            member.get("linkedin"), member.get("twitter"),
            member.get("instagram"), member.get("facebook"),
            member.get("tiktok"), member.get("youtube"),
            member.get("website"), member.get("phone"),
            member.get("location"), member.get("city"), member.get("country"),
            member.get("bio"), member.get("industry"), member.get("funding_stage"),
            member.get("notes"),
            json.dumps(member.get("extra_fields") or {}),
            member.get("member_type", "unknown"),
            member.get("confidence", 0.0),
            json.dumps(member.get("tags", []))
        ))
        return cur.lastrowid


def _merge(existing: dict, new: dict) -> dict:
    """Merge two member records, preferring longer/richer values."""
    result = dict(existing)
    text_fields = [
        "company", "title", "linkedin", "twitter", "instagram", "facebook",
        "tiktok", "youtube", "website", "phone", "location", "city", "country",
        "bio", "industry", "funding_stage", "notes",
    ]
    for k in text_fields:
        ev = (existing.get(k) or "").strip()
        nv = (new.get(k) or "").strip()
        if len(nv) > len(ev):
            result[k] = nv

    # Merge extra_fields
    try:
        ex_extra = json.loads(existing.get("extra_fields") or "{}")
    except Exception:
        ex_extra = {}
    new_extra = new.get("extra_fields") or {}
    if isinstance(new_extra, str):
        try:
            new_extra = json.loads(new_extra)
        except Exception:
            new_extra = {}
    merged_extra = {**ex_extra}
    for k, v in new_extra.items():
        if v and (not merged_extra.get(k) or len(str(v)) > len(str(merged_extra[k]))):
            merged_extra[k] = v
    result["extra_fields"] = merged_extra

    # Merge tags
    existing_tags = set(json.loads(existing.get("tags") or "[]"))
    new_tags = set(new.get("tags") or [])
    result["tags"] = list(existing_tags | new_tags)

    # Keep better typing
    if new.get("confidence", 0) > existing.get("confidence", 0):
        result["member_type"] = new.get("member_type", existing.get("member_type"))
        result["confidence"] = new.get("confidence", 0)
    return result


def link_member_event(conn: sqlite3.Connection, member_id: int, event_id: int):
    conn.execute(
        "INSERT OR IGNORE INTO member_events (member_id, event_id) VALUES (?,?)",
        (member_id, event_id)
    )
    conn.execute(
        "UPDATE members SET event_count = (SELECT COUNT(*) FROM member_events WHERE member_id=?) WHERE id=?",
        (member_id, member_id)
    )
