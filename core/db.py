"""SQLite storage for garments, outfits, feedback and the user's style vector."""
import json
import sqlite3
from datetime import date

import numpy as np

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS garments(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    category    TEXT,
    slot        TEXT,          -- top | bottom | onepiece | shoes | layer
    colors      TEXT,          -- JSON list
    pattern     TEXT,
    formality   INTEGER,       -- 1 (lounge) .. 5 (very formal)
    warmth      INTEGER,       -- 1 light .. 3 warm
    confidence  REAL,
    image_path  TEXT,
    embedding   BLOB,          -- CLIP image embedding (float32)
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS outfits(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    garment_ids TEXT,          -- JSON list
    occasion    TEXT,
    score       REAL,
    title       TEXT,
    reason      TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    outfit_id   INTEGER REFERENCES outfits(id),
    action      TEXT,          -- like | skip | worn
    day         TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS profile(
    key   TEXT PRIMARY KEY,
    value BLOB
);
"""


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.executescript(SCHEMA)


# ---------------------------------------------------------------- garments
def _row_to_garment(r):
    g = dict(r)
    g["colors"] = json.loads(g["colors"] or "[]")
    g["embedding"] = np.frombuffer(g["embedding"], dtype=np.float32) if g["embedding"] else None
    return g


def add_garment(tags: dict, image_path: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO garments(name, category, slot, colors, pattern, formality, warmth,"
            " confidence, image_path, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (tags["name"], tags["category"], tags["slot"], json.dumps(tags["colors"]), tags["pattern"],
             int(tags["formality"]), int(tags["warmth"]), float(tags.get("confidence", 1.0)),
             image_path, np.asarray(tags["embedding"], dtype=np.float32).tobytes()))
        return cur.lastrowid


def list_garments():
    with _conn() as c:
        rows = c.execute("SELECT * FROM garments ORDER BY id DESC").fetchall()
    return [_row_to_garment(r) for r in rows]


def update_garment(gid: int, **fields):
    if "colors" in fields:
        fields["colors"] = json.dumps(fields["colors"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE garments SET {cols} WHERE id=?", (*fields.values(), gid))


def delete_garment(gid: int):
    with _conn() as c:
        c.execute("DELETE FROM garments WHERE id=?", (gid,))


# ---------------------------------------------------------------- outfits
def save_outfit(garment_ids, occasion, score, title, reason) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO outfits(garment_ids, occasion, score, title, reason) VALUES (?,?,?,?,?)",
                        (json.dumps([int(i) for i in garment_ids]), occasion, float(score), title, reason))
        return cur.lastrowid


def add_feedback(outfit_id: int, action: str, day: date | None = None):
    with _conn() as c:
        c.execute("INSERT INTO feedback(outfit_id, action, day) VALUES (?,?,?)",
                  (outfit_id, action, (day or date.today()).isoformat()))


def last_worn() -> dict:
    """{garment_id: date it was last worn}"""
    with _conn() as c:
        rows = c.execute("SELECT o.garment_ids, f.day FROM feedback f JOIN outfits o ON o.id=f.outfit_id "
                         "WHERE f.action='worn'").fetchall()
    worn = {}
    for r in rows:
        d = date.fromisoformat(r["day"])
        for gid in json.loads(r["garment_ids"]):
            if gid not in worn or d > worn[gid]:
                worn[gid] = d
    return worn


def wear_counts() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT o.garment_ids FROM feedback f JOIN outfits o ON o.id=f.outfit_id "
                         "WHERE f.action='worn'").fetchall()
    counts = {}
    for r in rows:
        for gid in json.loads(r["garment_ids"]):
            counts[gid] = counts.get(gid, 0) + 1
    return counts


def feedback_stats() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT action, COUNT(*) n FROM feedback GROUP BY action").fetchall()
    return {r["action"]: r["n"] for r in rows}


# ---------------------------------------------------------------- style profile
def get_style_vector():
    with _conn() as c:
        r = c.execute("SELECT value FROM profile WHERE key='style_vector'").fetchone()
    return np.frombuffer(r["value"], dtype=np.float32).copy() if r else None


def set_style_vector(v):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO profile(key, value) VALUES ('style_vector', ?)",
                  (np.asarray(v, dtype=np.float32).tobytes(),))