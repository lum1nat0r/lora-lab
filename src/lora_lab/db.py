"""SQLite persistence: runs, jobs, images."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    grid_hash   TEXT,
    profile     TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    meta_json   TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY,
    job_hash    TEXT UNIQUE NOT NULL,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    subject   TEXT NOT NULL,
    lora_id     TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    strength    REAL NOT NULL,
    scenario    TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    condition   TEXT NOT NULL,
    profile     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    prompt_id   TEXT,
    error       TEXT,
    prompt_json TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_run ON jobs(run_id);

CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    filename    TEXT NOT NULL,
    subfolder   TEXT NOT NULL DEFAULT '',
    type        TEXT NOT NULL DEFAULT 'output',
    path        TEXT NOT NULL,
    sha256      TEXT,
    width       INTEGER,
    height      INTEGER,
    bytes       INTEGER,
    created_at  REAL NOT NULL,
    UNIQUE(job_id, idx)
);

CREATE TABLE IF NOT EXISTS scores (
    id          INTEGER PRIMARY KEY,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    scorer      TEXT NOT NULL,
    version     TEXT NOT NULL,
    axis        TEXT NOT NULL,
    value       REAL,
    meta_json   TEXT,
    created_at  REAL NOT NULL,
    UNIQUE(image_id, scorer, version, axis)
);

CREATE VIEW IF NOT EXISTS run_progress AS
SELECT run_id,
       status,
       COUNT(*) AS n
FROM jobs
GROUP BY run_id, status;
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- runs --------------------------------------------------------------
    def upsert_run(self, name: str, grid_hash: str, profile: str, meta: dict[str, Any]) -> int:
        now = time.time()
        row = self.conn.execute("SELECT id FROM runs WHERE name = ?", (name,)).fetchone()
        if row:
            self.conn.execute(
                "UPDATE runs SET grid_hash=?, profile=?, updated_at=?, meta_json=? WHERE id=?",
                (grid_hash, profile, now, json.dumps(meta), row["id"]),
            )
            self.conn.commit()
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO runs(name, grid_hash, profile, created_at, updated_at, meta_json)"
            " VALUES (?,?,?,?,?,?)",
            (name, grid_hash, profile, now, now, json.dumps(meta)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # -- jobs --------------------------------------------------------------
    def add_job(self, run_id: int, job_hash: str, job: Any, prompt: dict[str, Any]) -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO jobs(job_hash, run_id, subject, lora_id, epoch, strength,"
            " scenario, seed, condition, profile, status, prompt_json, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?,?,?)",
            (
                job_hash,
                run_id,
                job.subject,
                job.lora_id,
                job.epoch,
                job.strength,
                job.scenario,
                job.seed,
                job.condition,
                job.profile,
                json.dumps(prompt, ensure_ascii=False),
                now,
                now,
            ),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            row = self.conn.execute(
                "SELECT id FROM jobs WHERE job_hash = ?", (job_hash,)
            ).fetchone()
            return int(row["id"])
        return int(cur.lastrowid)

    def job_status(self, job_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM jobs WHERE job_hash = ?", (job_hash,)
        ).fetchone()
        return str(row["status"]) if row else None

    def set_status(
        self, job_id: int, status: str, *, prompt_id: str | None = None, error: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE jobs SET status=?, prompt_id=COALESCE(?, prompt_id), error=?, updated_at=?"
            " WHERE id=?",
            (status, prompt_id, error, time.time(), job_id),
        )
        self.conn.commit()

    def pending_jobs(self, run_id: int | None = None) -> list[sqlite3.Row]:
        if run_id is None:
            return list(
                self.conn.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id")
            )
        return list(
            self.conn.execute(
                "SELECT * FROM jobs WHERE status='pending' AND run_id=? ORDER BY id", (run_id,)
            )
        )

    # -- images ------------------------------------------------------------
    def add_image(
        self,
        job_id: int,
        idx: int,
        filename: str,
        subfolder: str,
        type_: str,
        path: str,
        sha256: str | None,
        width: int | None,
        height: int | None,
        size: int | None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO images(job_id, idx, filename, subfolder, type, path,"
            " sha256, width, height, bytes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                idx,
                filename,
                subfolder,
                type_,
                path,
                sha256,
                width,
                height,
                size,
                time.time(),
            ),
        )
        self.conn.commit()

    # -- scores ------------------------------------------------------------
    def add_score(
        self,
        image_id: int,
        scorer: str,
        version: str,
        axis: str,
        value: float | None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO scores(image_id, scorer, version, axis, value, meta_json,"
            " created_at) VALUES (?,?,?,?,?,?,?)",
            (image_id, scorer, version, axis, value, json.dumps(meta or {}), time.time()),
        )
        self.conn.commit()

    def has_axes(
        self, image_id: int, scorer: str, version: str, axes: tuple[str, ...]
    ) -> bool:
        if not axes:
            return False
        placeholders = ",".join("?" for _ in axes)
        row = self.conn.execute(
            f"SELECT COUNT(DISTINCT axis) AS n FROM scores WHERE image_id=? AND scorer=?"
            f" AND version=? AND axis IN ({placeholders})",
            (image_id, scorer, version, *axes),
        ).fetchone()
        return bool(row) and int(row["n"]) == len(axes)

    def images_for_run(self, run_id: int | None = None) -> list[sqlite3.Row]:
        if run_id is None:
            query = (
                "SELECT images.*, jobs.subject, jobs.lora_id, jobs.epoch, jobs.strength,"
                " jobs.scenario, jobs.seed, jobs.condition FROM images"
                " JOIN jobs ON jobs.id = images.job_id ORDER BY images.id"
            )
            return list(self.conn.execute(query))
        query = (
            "SELECT images.*, jobs.subject, jobs.lora_id, jobs.epoch, jobs.strength,"
            " jobs.scenario, jobs.seed, jobs.condition FROM images"
            " JOIN jobs ON jobs.id = images.job_id WHERE jobs.run_id = ? ORDER BY images.id"
        )
        return list(self.conn.execute(query, (run_id,)))

    def scored_images(self, run_id: int | None = None) -> list[sqlite3.Row]:
        where = "WHERE j.run_id = ?" if run_id is not None else ""
        params: tuple[Any, ...] = (run_id,) if run_id is not None else ()
        query = f"""
            SELECT i.id, i.path, j.subject, j.lora_id, j.epoch, j.strength, j.scenario, j.seed,
                   j.condition,
                   MAX(CASE WHEN s.axis = 'identity' THEN s.value END)  AS identity,
                   MAX(CASE WHEN s.axis = 'adherence' THEN s.value END) AS adherence,
                   MAX(CASE WHEN s.axis = 'quality' THEN s.value END)   AS quality,
                   MAX(CASE WHEN s.axis = 'critical' THEN s.value END)  AS critical
            FROM images i
            JOIN jobs j ON j.id = i.job_id
            LEFT JOIN scores s ON s.image_id = i.id
            {where}
            GROUP BY i.id
            ORDER BY i.id
        """
        return list(self.conn.execute(query, params))

    def run_by_name(self, name: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE name = ?", (name,)).fetchone()

    # -- reporting ---------------------------------------------------------
    def counts(self, run_id: int | None = None) -> dict[str, int]:
        where = "WHERE run_id=?" if run_id is not None else ""
        params: tuple[Any, ...] = (run_id,) if run_id is not None else ()
        rows = self.conn.execute(
            f"SELECT status, COUNT(*) AS n FROM jobs {where} GROUP BY status", params
        )
        out = {row["status"]: int(row["n"]) for row in rows}
        total = self.conn.execute(f"SELECT COUNT(*) AS n FROM jobs {where}", params).fetchone()
        out["total"] = int(total["n"]) if total else 0
        return out

    def job_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
        return int(row["n"]) if row else 0

    def image_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()
        return int(row["n"]) if row else 0

    def recent_images(self, limit: int = 24) -> Iterable[sqlite3.Row]:
        return self.conn.execute(
            "SELECT images.*, jobs.subject, jobs.lora_id, jobs.scenario, jobs.seed,"
            " jobs.strength, jobs.epoch FROM images JOIN jobs ON jobs.id = images.job_id"
            " ORDER BY images.id DESC LIMIT ?",
            (limit,),
        )

    def runs(self) -> Iterable[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs ORDER BY id DESC")
