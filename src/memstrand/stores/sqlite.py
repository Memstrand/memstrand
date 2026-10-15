"""SQLite: session cache (L1) + append-only provenance ledger.

Shares `memstrand.db` with the graph store and the keyword index; table
names (`turns`, `provenance_log`, `distill_state`) are namespaced to avoid
collisions. WAL mode + an RLock: a daemon may serve several agents
concurrently from one connection.

The provenance ledger is append-only (`provenance_log`). The old
`provenance` table (one row per atom, overwritten via INSERT OR REPLACE)
is left in place untouched for anyone reading a pre-upgrade DB — new code
never reads or writes it again.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time


class SqliteSessionStore:
    def __init__(self, path: str = "./memstrand.db"):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.path = path  # discoverable by Memory, to co-locate the keyword index
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5s for a lock instead of raising "database is locked"
        # immediately — several stores share this file across threads/
        # processes (graph + sessions + keyword index).
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS turns(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT, task_id TEXT, role TEXT,
                    content TEXT, ts REAL)"""
            )
            # legacy single-row-per-atom ledger: untouched, kept for old DBs
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS provenance(
                    atom_id TEXT PRIMARY KEY, action TEXT,
                    detail TEXT, ts REAL)"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS provenance_log(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    atom_id TEXT, action TEXT, detail TEXT, ts REAL)"""
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_provenance_log_atom ON provenance_log(atom_id)"
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS distill_state(
                    session_id TEXT PRIMARY KEY,
                    last_turn_id INTEGER, ts REAL)"""
            )
            self.conn.commit()

    # -- turns ------------------------------------------------------------------

    def append_turn(
        self, session_id: str, role: str, content: str, task_id: str | None = None,
        ts: float | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO turns(session_id, task_id, role, content, ts) VALUES (?,?,?,?,?)",
                (session_id, task_id or "", role, content, ts if ts is not None else time.time()),
            )
            self.conn.commit()

    def get_session(self, session_id: str, after_id: int = 0) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, role, content, ts, task_id FROM turns "
                "WHERE session_id=? AND id > ? ORDER BY id",
                (session_id, after_id),
            ).fetchall()
        return [
            {"id": i, "role": r, "content": c, "ts": t, "task_id": tid or None}
            for i, r, c, t, tid in rows
        ]

    def list_sessions(self, task_id: str | None = None) -> list[str]:
        q = "SELECT DISTINCT session_id FROM turns"
        args: tuple = ()
        if task_id:
            q += " WHERE task_id=?"
            args = (task_id,)
        with self._lock:
            return [r[0] for r in self.conn.execute(q, args).fetchall()]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
            self.conn.commit()

    # -- incremental distillation bookkeeping ------------------------------------

    def last_distilled_turn_id(self, session_id: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT last_turn_id FROM distill_state WHERE session_id=?", (session_id,)
            ).fetchone()
        return row[0] if row else 0

    def mark_distilled(self, session_id: str, last_turn_id: int) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO distill_state(session_id, last_turn_id, ts) VALUES (?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       last_turn_id=excluded.last_turn_id, ts=excluded.ts""",
                (session_id, last_turn_id, time.time()),
            )
            self.conn.commit()

    def undistilled_sessions(self) -> list[str]:
        """Sessions with turns newer than their last distilled turn id
        (including sessions that were never distilled at all)."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT t.session_id FROM turns t
                   LEFT JOIN distill_state d ON d.session_id = t.session_id
                   GROUP BY t.session_id
                   HAVING MAX(t.id) > COALESCE(MAX(d.last_turn_id), 0)"""
            ).fetchall()
        return [r[0] for r in rows]

    # -- provenance ledger (append-only) -----------------------------------------

    def log_provenance(
        self, atom_id: str, action: str, detail: str = "", ts: float | None = None
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO provenance_log(atom_id, action, detail, ts) VALUES (?,?,?,?)",
                (atom_id, action, detail, ts if ts is not None else time.time()),
            )
            self.conn.commit()

    def history(self, atom_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT action, detail, ts FROM provenance_log WHERE atom_id=? ORDER BY id",
                (atom_id,),
            ).fetchall()
        return [{"action": a, "detail": d, "ts": t} for a, d, t in rows]

    def dump_ledger(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT atom_id, action, detail, ts FROM provenance_log ORDER BY id"
            ).fetchall()
        return [{"atom_id": aid, "action": a, "detail": d, "ts": t} for aid, a, d, t in rows]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
