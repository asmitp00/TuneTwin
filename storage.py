"""Per-listener persistence for Resona.

Every listener gets their own row for:
  * their Spotify OAuth token (users table), and
  * their adaptive-learning state: LinUCB matrices + Like/Skip history
    (learning_state table).

The learning owner is either a Spotify user ID (signed-in listener) or a
"guest:<random>" ID kept in the browser session, so people who have not
connected Spotify still get their own separate feedback history.

Two backends share the same tables and queries:
  * PostgreSQL (e.g. Neon) when given a postgres:// or postgresql:// URL;
    used when deployed, so data survives restarts.
  * SQLite when given a file path; used for local development.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

# DOUBLE PRECISION keeps full epoch-second timestamps in PostgreSQL (REAL is
# only 4 bytes there); SQLite treats it as an ordinary floating-point column.
SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        display_name  TEXT NOT NULL,
        token         TEXT,
        created_at    DOUBLE PRECISION NOT NULL,
        last_login_at DOUBLE PRECISION NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_state (
        owner_id   TEXT PRIMARY KEY,
        state      TEXT NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    )
    """,
]


def is_postgres_url(target: str) -> bool:
    return target.startswith(("postgres://", "postgresql://"))


class Storage:
    def __init__(self, target: Path | str):
        target = str(target)
        if is_postgres_url(target):
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            self.backend = "postgres"
            # A small pool reuses connections across requests, so each query
            # avoids a fresh network + TLS handshake to the database host.
            # `check` replaces connections the server closed while idle (Neon
            # suspends idle databases). `prepare_threshold=None` keeps queries
            # compatible with PgBouncer-style poolers such as Neon's.
            self._pool = ConnectionPool(
                target,
                min_size=1,
                max_size=8,
                kwargs={"row_factory": dict_row, "prepare_threshold": None},
                check=ConnectionPool.check_connection,
                max_idle=240,
                name="resona",
                open=True,
            )
            self.description = "PostgreSQL"
        else:
            self.backend = "sqlite"
            self.path = Path(target)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.description = f"SQLite ({self.path})"
            with self._connect() as db:
                db.execute("PRAGMA journal_mode=WAL")
        with self._connect() as db:
            for statement in SCHEMA:
                db.execute(statement)

    @contextmanager
    def _connect(self):
        if self.backend == "postgres":
            # Commits on success, rolls back on error, returns it to the pool.
            with self._pool.connection() as db:
                yield db
            return
        # A short-lived connection per call keeps SQLite safe across Flask's
        # request threads without sharing one sqlite3 connection object.
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _execute(self, db, query: str, params: tuple = ()):
        if self.backend == "postgres":
            query = query.replace("?", "%s")
        return db.execute(query, params)

    def close(self) -> None:
        if self.backend == "postgres":
            self._pool.close()

    # ---- users and Spotify tokens -------------------------------------
    def upsert_user(self, user_id: str, display_name: str, token: dict) -> None:
        now = time.time()
        with self._connect() as db:
            self._execute(
                db,
                """
                INSERT INTO users (id, display_name, token, created_at, last_login_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    display_name = excluded.display_name,
                    token = excluded.token,
                    last_login_at = excluded.last_login_at
                """,
                (user_id, display_name, json.dumps(token), now, now),
            )

    def get_user(self, user_id: str) -> dict | None:
        with self._connect() as db:
            row = self._execute(
                db,
                "SELECT id, display_name, created_at, last_login_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_token(self, user_id: str) -> dict | None:
        with self._connect() as db:
            row = self._execute(db, "SELECT token FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row or not row["token"]:
            return None
        try:
            return json.loads(row["token"])
        except (TypeError, ValueError):
            return None

    def save_token(self, user_id: str, token: dict) -> None:
        with self._connect() as db:
            self._execute(db, "UPDATE users SET token = ? WHERE id = ?", (json.dumps(token), user_id))

    def clear_token(self, user_id: str) -> None:
        with self._connect() as db:
            self._execute(db, "UPDATE users SET token = NULL WHERE id = ?", (user_id,))

    # ---- adaptive-learning state --------------------------------------
    def load_state(self, owner_id: str) -> dict | None:
        with self._connect() as db:
            row = self._execute(
                db, "SELECT state FROM learning_state WHERE owner_id = ?", (owner_id,)
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["state"])
        except (TypeError, ValueError):
            return None

    def save_state(self, owner_id: str, state: dict) -> None:
        with self._connect() as db:
            self._execute(
                db,
                """
                INSERT INTO learning_state (owner_id, state, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    state = excluded.state,
                    updated_at = excluded.updated_at
                """,
                (owner_id, json.dumps(state, separators=(",", ":")), time.time()),
            )

    def adopt_state(self, source_owner: str, target_owner: str) -> bool:
        """Move a guest's learning to an account that has none yet."""
        with self._connect() as db:
            if self._execute(
                db, "SELECT 1 FROM learning_state WHERE owner_id = ?", (target_owner,)
            ).fetchone():
                return False
            moved = self._execute(
                db,
                "UPDATE learning_state SET owner_id = ? WHERE owner_id = ?",
                (target_owner, source_owner),
            ).rowcount
        return bool(moved)
