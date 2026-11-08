from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "fitfuel.db"


def db_path() -> Path:
    return Path(os.environ.get("FITFUEL_DB", DEFAULT_DB))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                profile_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                summary TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                is_current INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                trace_json TEXT
            );
            CREATE TABLE IF NOT EXISTS custom_items (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pending_suggestions (
                user_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            """
        )


def create_user(profile: dict) -> str:
    init_db()
    user_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO users (id, created_at, profile_json) VALUES (?, ?, ?)",
            (user_id, _now(), json.dumps(profile)),
        )
    return user_id


def update_profile(user_id: str, profile: dict) -> None:
    init_db()
    with connect() as conn:
        conn.execute("UPDATE users SET profile_json = ? WHERE id = ?", (json.dumps(profile), user_id))


def get_profile(user_id: str) -> dict:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT profile_json FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise KeyError(user_id)
    return json.loads(row["profile_json"])


def list_users() -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT id, created_at, profile_json FROM users ORDER BY created_at DESC").fetchall()
    users = []
    for row in rows:
        profile = json.loads(row["profile_json"])
        users.append({"id": row["id"], "created_at": row["created_at"], "name": profile.get("name", "Friend"), "profile": profile})
    return users


def _insert_plan(user_id: str, plan: dict) -> str:
    plan_id = str(uuid.uuid4())
    summary = plan.get("summary") or "Weekly plan"
    with connect() as conn:
        conn.execute("UPDATE plans SET is_current = 0 WHERE user_id = ?", (user_id,))
        conn.execute(
            "INSERT INTO plans (id, user_id, created_at, summary, plan_json, is_current) VALUES (?, ?, ?, ?, ?, 1)",
            (plan_id, user_id, _now(), summary, json.dumps(plan)),
        )
    return plan_id


def save_plan(user_id: str, plan: dict) -> str:
    init_db()
    return _insert_plan(user_id, plan)


def update_plan(user_id: str, plan: dict) -> str:
    init_db()
    current = get_current_plan(user_id)
    if current is None:
        return _insert_plan(user_id, plan)
    summary = plan.get("summary") or current.get("summary") or "Weekly plan"
    with connect() as conn:
        conn.execute(
            "UPDATE plans SET summary = ?, plan_json = ? WHERE id = ?",
            (summary, json.dumps(plan), current["id"]),
        )
    return current["id"]


def get_plan(plan_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
    return _plan_row(row) if row else None


def get_current_plan(user_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM plans WHERE user_id = ? AND is_current = 1 ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return _plan_row(row) if row else None


def list_plans(user_id: str) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM plans WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [_plan_row(row) for row in rows]


def set_current(user_id: str, plan_id: str) -> None:
    init_db()
    with connect() as conn:
        conn.execute("UPDATE plans SET is_current = 0 WHERE user_id = ?", (user_id,))
        conn.execute("UPDATE plans SET is_current = 1 WHERE user_id = ? AND id = ?", (user_id, plan_id))


def add_message(user_id: str, role: str, content: str, trace: list | None = None) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, created_at, role, content, trace_json) VALUES (?, ?, ?, ?, ?)",
            (user_id, _now(), role, content, json.dumps(trace or [])),
        )


def list_messages(user_id: str) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content, trace_json, created_at FROM messages WHERE user_id = ? ORDER BY id ASC",
            (user_id,),
        ).fetchall()
    messages = []
    for row in rows:
        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "trace": json.loads(row["trace_json"] or "[]"),
                "created_at": row["created_at"],
            }
        )
    return messages


def custom_items() -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT payload_json FROM custom_items").fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def save_custom(item: dict) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO custom_items (id, kind, payload_json) VALUES (?, ?, ?)",
            (item["id"], item["kind"], json.dumps(item)),
        )


def delete_custom(item_id: str) -> None:
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM custom_items WHERE id = ?", (item_id,))


def set_pending(user_id: str, payload: dict) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pending_suggestions (user_id, payload_json) VALUES (?, ?)",
            (user_id, json.dumps(payload)),
        )


def get_pending(user_id: str) -> dict | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT payload_json FROM pending_suggestions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return json.loads(row["payload_json"]) if row else None


def clear_pending(user_id: str) -> None:
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM pending_suggestions WHERE user_id = ?", (user_id,))


def _plan_row(row: sqlite3.Row) -> dict:
    plan = json.loads(row["plan_json"])
    plan["id"] = row["id"]
    plan["created_at"] = row["created_at"]
    plan["summary"] = row["summary"]
    plan["is_current"] = bool(row["is_current"])
    return plan
