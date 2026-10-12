"""SQLite: session cache (L1) + provenance ledger. Embedded, zero setup."""

from __future__ import annotations

import sqlite3
import time


class SqliteSessionStore:
    def __init__(self, path: str = "./memstrand.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, task_id TEXT, role TEXT,
                content TEXT, ts REAL)"""
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS provenance(
                atom_id TEXT PRIMARY KEY, action TEXT,
                detail TEXT, ts REAL)"""
        )
        self.conn.commit()

    def append_turn(
        self, session_id: str, role: str, content: str, task_id: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO turns(session_id, task_id, role, content, ts) VALUES (?,?,?,?,?)",
            (session_id, task_id or "", role, content, time.time()),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, ts, task_id FROM turns WHERE session_id=? ORDER BY ts",
            (session_id,),
        ).fetchall()
        return [{"role": r, "content": c, "ts": t, "task_id": tid} for r, c, t, tid in rows]

    def list_sessions(self, task_id: str | None = None) -> list[str]:
        q = "SELECT DISTINCT session_id FROM turns"
        args: tuple = ()
        if task_id:
            q += " WHERE task_id=?"
            args = (task_id,)
        return [r[0] for r in self.conn.execute(q, args).fetchall()]

    def clear_session(self, session_id: str) -> None:
        self.conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        self.conn.commit()

    def log_provenance(self, atom_id: str, action: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO provenance(atom_id, action, detail, ts) VALUES (?,?,?,?)",
            (atom_id, action, detail, time.time()),
        )
        self.conn.commit()
