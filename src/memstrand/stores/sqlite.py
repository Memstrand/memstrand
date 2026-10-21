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

import json
import os
import re
import sqlite3
import threading
import time

from ..config import ensure_private_dir, secure_file
from ..schema import decode_scopes, encode_scopes


class SqliteSessionStore:
    def __init__(self, path: str = "./memstrand.db"):
        parent = os.path.dirname(path)
        if parent:
            # deep-review fix #12: directories memstrand creates are 0o700.
            ensure_private_dir(parent)
        self.path = path  # discoverable by Memory, to co-locate the keyword index
        self._lock = threading.RLock()
        is_new = not os.path.exists(path)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5s for a lock instead of raising "database is locked"
        # immediately — several stores share this file across threads/
        # processes (graph + sessions + keyword index).
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()
        if is_new:
            # deep-review fix #12: the db file (and its WAL/SHM siblings,
            # once sqlite creates them) are 0o600 — only chmod files we
            # actually created, never an existing user-supplied db.
            secure_file(path)
            secure_file(f"{path}-wal")
            secure_file(f"{path}-shm")

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS turns(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT, task_id TEXT, role TEXT,
                    content TEXT, ts REAL)"""
            )
            # scoping v2 phase 2: scope keys this turn TOUCHED (repos read/
            # edited via tool calls, paths in the prompt), as encoded scopes
            # ("" = none recorded). Added in place for pre-existing DBs.
            cols = {row[1] for row in self.conn.execute("PRAGMA table_info(turns)")}
            if "scopes" not in cols:
                self.conn.execute("ALTER TABLE turns ADD COLUMN scopes TEXT DEFAULT ''")
            # scoping v2 phase 4: who logged the turn ("claude-code@host", ...)
            if "agent" not in cols:
                self.conn.execute("ALTER TABLE turns ADD COLUMN agent TEXT DEFAULT ''")
            # full-text index over raw turns (session search); optional —
            # falls back to substring search if this SQLite lacks FTS5
            try:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5("
                    "content, turn_id UNINDEXED, session_id UNINDEXED, "
                    "tokenize = 'unicode61 remove_diacritics 2')"
                )
                self._fts = True
                indexed = self.conn.execute("SELECT COUNT(*) FROM turns_fts").fetchone()[0]
                if indexed == 0:
                    self.conn.execute(
                        "INSERT INTO turns_fts(content, turn_id, session_id) "
                        "SELECT content, id, session_id FROM turns"
                    )
            except sqlite3.OperationalError:
                self._fts = False
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
                "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)"
            )
            # scoping v2 phase 3: memories whose scope needs the user's call
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS scope_review(
                    atom_id TEXT PRIMARY KEY, kind TEXT, reason TEXT,
                    candidates TEXT, status TEXT, shown_sessions TEXT,
                    created_ts REAL, resolved_ts REAL)"""
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
        ts: float | None = None, scopes: list[str] | None = None, agent: str | None = None,
    ) -> None:
        touched = [s for s in (scopes or []) if s]
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO turns(session_id, task_id, role, content, ts, scopes, agent) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    session_id, task_id or "", role, content,
                    ts if ts is not None else time.time(),
                    encode_scopes(touched) if touched else "",
                    agent or "",
                ),
            )
            if self._fts:
                self.conn.execute(
                    "INSERT INTO turns_fts(content, turn_id, session_id) VALUES (?,?,?)",
                    (content, cur.lastrowid, session_id),
                )
            self.conn.commit()

    def get_session(self, session_id: str, after_id: int = 0) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, role, content, ts, task_id, scopes, agent FROM turns "
                "WHERE session_id=? AND id > ? ORDER BY id",
                (session_id, after_id),
            ).fetchall()
        return [
            {
                "id": i, "role": r, "content": c, "ts": t, "task_id": tid or None,
                "scopes": [s for s in decode_scopes(sc) if s] if sc else [],
                "agent": ag or None,
            }
            for i, r, c, t, tid, sc, ag in rows
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
            if self._fts:
                self.conn.execute("DELETE FROM turns_fts WHERE session_id=?", (session_id,))
            self.conn.commit()

    def find_sessions(self, ref: str) -> list[str]:
        """Session ids matching `ref`: exact, then `claude-code:<ref>`, then
        any id (with or without that prefix) starting with `ref`."""
        with self._lock:
            for exact in (ref, f"claude-code:{ref}"):
                if self.conn.execute(
                    "SELECT 1 FROM turns WHERE session_id=? LIMIT 1", (exact,)
                ).fetchone():
                    return [exact]
            like = ref.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self.conn.execute(
                "SELECT DISTINCT session_id FROM turns WHERE session_id LIKE ? ESCAPE '\\' "
                "OR session_id LIKE ? ESCAPE '\\' ORDER BY session_id",
                (f"{like}%", f"claude-code:{like}%"),
            ).fetchall()
        return [r[0] for r in rows]

    def search_turns(self, query: str, limit: int = 20, session_id: str | None = None) -> list[dict]:
        """Full-text search over raw logged turns, best match first. Every
        word must appear (FTS5), or the whole query as a substring when FTS5
        isn't available."""
        words = re.findall(r"\w+", query, re.UNICODE)
        if not words:
            return []
        with self._lock:
            if self._fts:
                match = " ".join(f'"{w}"' for w in words)
                sql = (
                    "SELECT t.id, t.session_id, t.role, t.ts, t.content, "
                    "snippet(turns_fts, 0, '[', ']', '…', 16) "
                    "FROM turns_fts JOIN turns t ON t.id = turns_fts.turn_id "
                    "WHERE turns_fts MATCH ?"
                )
                params: list = [match]
                if session_id:
                    sql += " AND t.session_id = ?"
                    params.append(session_id)
                sql += " ORDER BY bm25(turns_fts) LIMIT ?"
                params.append(limit)
                try:
                    rows = self.conn.execute(sql, params).fetchall()
                except sqlite3.OperationalError:
                    return []
            else:
                sql = "SELECT id, session_id, role, ts, content, NULL FROM turns WHERE content LIKE ?"
                params = [f"%{query}%"]
                if session_id:
                    sql += " AND session_id = ?"
                    params.append(session_id)
                sql += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                rows = self.conn.execute(sql, params).fetchall()
        return [
            {"id": i, "session_id": sid, "role": r, "ts": ts, "content": c, "snippet": snip or c[:200]}
            for i, sid, r, ts, c, snip in rows
        ]

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

    def dump_distill_state(self) -> list[dict]:
        """Every session's distillation watermark, for export (fix #9:
        distill_state previously had no export/import round-trip at all,
        so restoring a backup re-distilled every session from scratch)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT session_id, last_turn_id FROM distill_state"
            ).fetchall()
        return [{"session_id": s, "last_turn_id": t} for s, t in rows]

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

    def copy_history(self, from_atom_id: str, to_atom_id: str) -> int:
        """Append copies of `from_atom_id`'s ledger entries (original
        timestamps kept) under `to_atom_id` -- used when an atom is re-keyed
        or merged, so its history follows it. The ledger stays append-only:
        the original rows are left untouched. Rows already present under
        `to_atom_id` are skipped, so re-running an interrupted migration
        doesn't duplicate history. Returns the number copied."""
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO provenance_log(atom_id, action, detail, ts) "
                "SELECT ?, src.action, src.detail, src.ts FROM provenance_log src "
                "WHERE src.atom_id=? AND NOT EXISTS ("
                "  SELECT 1 FROM provenance_log dst WHERE dst.atom_id=? AND dst.action=src.action"
                "  AND dst.detail=src.detail AND dst.ts=src.ts"
                ") ORDER BY src.id",
                (to_atom_id, from_atom_id, to_atom_id),
            )
            self.conn.commit()
            return cur.rowcount

    # -- scope review queue (scoping v2 phase 3) ------------------------------------

    def queue_scope_review(
        self, atom_id: str, kind: str, reason: str, candidates: list[str] | None = None
    ) -> bool:
        """Queue `atom_id` for a scope decision. No-op (False) if it already
        has a review row — pending, or resolved: a user-confirmed scope is
        never re-asked."""
        with self._lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO scope_review(atom_id, kind, reason, candidates, status, "
                "shown_sessions, created_ts, resolved_ts) VALUES (?,?,?,?, 'pending', '[]', ?, NULL)",
                (atom_id, kind, reason, json.dumps(candidates or []), time.time()),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def scope_reviews(self, status: str | None = "pending") -> list[dict]:
        q = ("SELECT atom_id, kind, reason, candidates, status, shown_sessions, created_ts "
             "FROM scope_review")
        args: tuple = ()
        if status is not None:
            q += " WHERE status=?"
            args = (status,)
        with self._lock:
            rows = self.conn.execute(q + " ORDER BY created_ts", args).fetchall()
        return [
            {
                "atom_id": a, "kind": k, "reason": r, "candidates": json.loads(c or "[]"),
                "status": st, "shown_sessions": json.loads(sh or "[]"), "created_ts": ts,
            }
            for a, k, r, c, st, sh, ts in rows
        ]

    def mark_scope_review_shown(self, atom_id: str, session_key: str, expire_after: int) -> None:
        """Record that a review was shown in `session_key`; once it has been
        shown in `expire_after` distinct sessions without an answer it's
        `expired` (the extractor's call stands, and it stops asking)."""
        # Two single-statement UPDATEs (each atomic in SQLite), so concurrent
        # hook processes can't lose an append the way a read-modify-write could.
        with self._lock:
            self.conn.execute(
                "UPDATE scope_review SET shown_sessions = json_insert(shown_sessions, '$[#]', ?) "
                "WHERE atom_id=? AND status='pending' "
                "AND NOT EXISTS (SELECT 1 FROM json_each(scope_review.shown_sessions) WHERE value=?)",
                (session_key, atom_id, session_key),
            )
            self.conn.execute(
                "UPDATE scope_review SET status='expired', resolved_ts=? "
                "WHERE atom_id=? AND status='pending' AND json_array_length(shown_sessions) >= ?",
                (time.time(), atom_id, expire_after),
            )
            self.conn.commit()

    def resolve_scope_review(self, atom_id: str, status: str = "confirmed") -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE scope_review SET status=?, resolved_ts=? WHERE atom_id=? AND status='pending'",
                (status, time.time(), atom_id),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def scope_review_status(self, atom_id: str) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM scope_review WHERE atom_id=?", (atom_id,)
            ).fetchone()
        return row[0] if row else None

    # -- store metadata ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self.conn.commit()

    def dump_ledger(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT atom_id, action, detail, ts FROM provenance_log ORDER BY id"
            ).fetchall()
        return [{"atom_id": aid, "action": a, "detail": d, "ts": t} for aid, a, d, t in rows]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
