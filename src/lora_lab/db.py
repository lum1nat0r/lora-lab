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
    job_hash    TEXT NOT NULL,
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
    updated_at  REAL NOT NULL,
    UNIQUE(run_id, job_hash)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_run ON jobs(run_id);
CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(job_hash);

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
        self._migrate_jobs_unique()
        self.conn.commit()

    def _migrate_jobs_unique(self) -> None:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if not row or "UNIQUE(run_id, job_hash)" in str(row["sql"]):
            return
        self.conn.executescript(
            """
            PRAGMA foreign_keys=OFF;
            DROP VIEW IF EXISTS run_progress;
            CREATE TABLE jobs_new (
                id          INTEGER PRIMARY KEY,
                job_hash    TEXT NOT NULL,
                run_id      INTEGER NOT NULL REFERENCES runs(id),
                subject     TEXT NOT NULL,
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
                updated_at  REAL NOT NULL,
                UNIQUE(run_id, job_hash)
            );
            INSERT INTO jobs_new SELECT * FROM jobs;
            DROP TABLE jobs;
            ALTER TABLE jobs_new RENAME TO jobs;
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_run ON jobs(run_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(job_hash);
            CREATE VIEW IF NOT EXISTS run_progress AS
            SELECT run_id, status, COUNT(*) AS n FROM jobs GROUP BY run_id, status;
            PRAGMA foreign_keys=ON;
            """
        )

    def close(self) -> None:
        # commit as a safety net: an open transaction would be rolled back
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

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
        # no commit here: callers commit once per job/batch instead of per row
        if cur.rowcount == 0:
            row = self.conn.execute(
                "SELECT id FROM jobs WHERE run_id = ? AND job_hash = ?", (run_id, job_hash)
            ).fetchone()
            return int(row["id"])
        return int(cur.lastrowid)

    def job_status(self, job_hash: str, run_id: int | None = None) -> str | None:
        if run_id is not None:
            row = self.conn.execute(
                "SELECT status FROM jobs WHERE run_id = ? AND job_hash = ?",
                (run_id, job_hash),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT status FROM jobs WHERE job_hash = ? ORDER BY id DESC LIMIT 1",
                (job_hash,),
            ).fetchone()
        return str(row["status"]) if row else None

    def reuse_completed_job(
        self, run_id: int, job_hash: str, job: Any, prompt: dict[str, Any]
    ) -> bool:
        """If this job is already done in ``run_id`` or an earlier run, link it to ``run_id``."""

        if self.job_status(job_hash, run_id=run_id) == "done":
            return True
        source = self.conn.execute(
            "SELECT id, prompt_id FROM jobs WHERE job_hash = ? AND status = 'done'"
            " ORDER BY id DESC LIMIT 1",
            (job_hash,),
        ).fetchone()
        if not source:
            return False
        source_images = self.conn.execute(
            "SELECT * FROM images WHERE job_id = ? ORDER BY idx", (source["id"],)
        ).fetchall()
        if not source_images:
            return False

        job_id = self.add_job(run_id, job_hash, job, prompt)
        self.set_status(job_id, "done", prompt_id=source["prompt_id"])
        for img in source_images:
            self.add_image(
                job_id=job_id,
                idx=int(img["idx"]),
                filename=str(img["filename"]),
                subfolder=str(img["subfolder"]),
                type_=str(img["type"]),
                path=str(img["path"]),
                sha256=img["sha256"],
                width=img["width"],
                height=img["height"],
                size=img["bytes"],
            )
            new_img = self.conn.execute(
                "SELECT id FROM images WHERE job_id = ? AND idx = ?",
                (job_id, int(img["idx"])),
            ).fetchone()
            if new_img:
                for sc in self.conn.execute(
                    "SELECT scorer, version, axis, value, meta_json FROM scores WHERE image_id = ?",
                    (img["id"],),
                ):
                    self.conn.execute(
                        "INSERT OR REPLACE INTO scores(image_id, scorer, version, axis, value,"
                        " meta_json, created_at) VALUES (?,?,?,?,?,?,?)",
                        (
                            new_img["id"],
                            sc["scorer"],
                            sc["version"],
                            sc["axis"],
                            sc["value"],
                            sc["meta_json"],
                            time.time(),
                        ),
                    )
        self.conn.commit()
        return True

    def set_status(
        self, job_id: int, status: str, *, prompt_id: str | None = None, error: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE jobs SET status=?, prompt_id=COALESCE(?, prompt_id), error=?, updated_at=?"
            " WHERE id=?",
            (status, prompt_id, error, time.time(), job_id),
        )

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

    def images_for_run(
        self,
        run_id: int | None = None,
        *,
        lora_id: str | None = None,
        epoch: int | None = None,
        strength: float | None = None,
    ) -> list[sqlite3.Row]:
        clauses, params = [], []
        for column, value in (
            ("jobs.run_id", run_id),
            ("jobs.lora_id", lora_id),
            ("jobs.epoch", epoch),
            ("jobs.strength", strength),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = (
            "SELECT images.*, jobs.subject, jobs.lora_id, jobs.epoch, jobs.strength,"
            " jobs.scenario, jobs.seed, jobs.condition FROM images"
            f" JOIN jobs ON jobs.id = images.job_id{where} ORDER BY images.id"
        )
        return list(self.conn.execute(query, params))

    def scored_images(self, run_id: int | None = None) -> list[sqlite3.Row]:
        where = "WHERE j.run_id = ?" if run_id is not None else ""
        params: tuple[Any, ...] = (run_id,) if run_id is not None else ()
        query = f"""
            SELECT i.id, i.path, j.subject, j.lora_id, j.epoch, j.strength, j.scenario, j.seed,
                   j.condition,
                   MAX(CASE WHEN s.axis = 'identity' THEN s.value END)     AS identity,
                   MAX(CASE WHEN s.axis = 'adherence' THEN s.value END)    AS adherence,
                   MAX(CASE WHEN s.axis = 'quality' THEN s.value END)      AS quality,
                   MAX(CASE WHEN s.axis = 'critical' THEN s.value END)     AS critical,
                   MAX(CASE WHEN s.axis = 'identity' THEN s.meta_json END)  AS identity_meta,
                   MAX(CASE WHEN s.axis = 'adherence' THEN s.meta_json END) AS adherence_meta,
                   MAX(CASE WHEN s.axis = 'critical' THEN s.meta_json END)  AS critical_meta
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

    def score_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM scores").fetchone()
        return int(row["n"]) if row else 0

    def score_max_created_at(self) -> float:
        row = self.conn.execute("SELECT MAX(created_at) AS m FROM scores").fetchone()
        return float(row["m"]) if row and row["m"] is not None else 0.0

    def recent_images(self, limit: int = 24) -> Iterable[sqlite3.Row]:
        return self.conn.execute(
            "SELECT images.*, jobs.subject, jobs.lora_id, jobs.scenario, jobs.seed,"
            " jobs.strength, jobs.epoch FROM images JOIN jobs ON jobs.id = images.job_id"
            " ORDER BY images.id DESC LIMIT ?",
            (limit,),
        )

    def runs(self) -> Iterable[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs ORDER BY id DESC")

    # -- cleanup -----------------------------------------------------------
    def _unlink_image_paths(self, data_dir: Path, paths: list[str]) -> int:
        root = (Path(data_dir) / "images").resolve()
        removed = 0
        for rel in paths:
            if not rel:
                continue
            target = (Path(data_dir) / rel).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                continue
            try:
                target.unlink()
                removed += 1
            except OSError:
                continue
            parent = target.parent
            while parent != root and parent.is_dir():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        return removed

    def _shared_paths(self, where: str, params: list[Any]) -> set[str]:
        """Paths of images whose jobs survive the delete about to happen.

        Reused jobs give two runs image rows with the same file path; a file
        must only be unlinked when no surviving row still points at it.
        """

        rows = self.conn.execute(
            "SELECT DISTINCT images.path FROM images"
            " JOIN jobs ON jobs.id = images.job_id"
            f" WHERE NOT ({where})",
            params,
        )
        return {str(row["path"]) for row in rows}

    def delete_lora(
        self,
        data_dir: Path,
        lora_id: str,
        *,
        epoch: int | None = None,
        strength: float | None = None,
        run_id: int | None = None,
    ) -> dict[str, int]:
        clauses = ["jobs.lora_id = ?"]
        params: list[Any] = [lora_id]
        if epoch is not None:
            clauses.append("jobs.epoch = ?")
            params.append(epoch)
        if strength is not None:
            clauses.append("jobs.strength = ?")
            params.append(strength)
        if run_id is not None:
            clauses.append("jobs.run_id = ?")
            params.append(run_id)
        where = " AND ".join(clauses)

        rows = self.conn.execute(
            f"SELECT images.path FROM images JOIN jobs ON jobs.id = images.job_id WHERE {where}",
            params,
        ).fetchall()
        doomed = {str(r["path"]) for r in rows}
        keep = self._shared_paths(where, params) if doomed else set()
        files = self._unlink_image_paths(data_dir, sorted(doomed - keep))

        cur = self.conn.execute(f"DELETE FROM jobs WHERE {where}", params)
        jobs_deleted = cur.rowcount
        self.conn.execute("DELETE FROM runs WHERE id NOT IN (SELECT DISTINCT run_id FROM jobs)")
        self.conn.commit()
        return {"jobs": jobs_deleted, "files": files}

    def delete_run(self, data_dir: Path, run_id: int | None = None) -> dict[str, int]:
        if run_id is None:
            rows = self.conn.execute("SELECT path FROM images").fetchall()
            files = self._unlink_image_paths(data_dir, [str(r["path"]) for r in rows])
            cur = self.conn.execute("DELETE FROM jobs")
            jobs_deleted = cur.rowcount
            self.conn.execute("DELETE FROM runs")
            self.conn.commit()
            return {"jobs": jobs_deleted, "files": files}

        rows = self.conn.execute(
            "SELECT images.path FROM images JOIN jobs ON jobs.id = images.job_id WHERE jobs.run_id = ?",
            (run_id,),
        ).fetchall()
        doomed = {str(r["path"]) for r in rows}
        keep = self._shared_paths("jobs.run_id = ?", [run_id]) if doomed else set()
        files = self._unlink_image_paths(data_dir, sorted(doomed - keep))
        cur = self.conn.execute("DELETE FROM jobs WHERE run_id = ?", (run_id,))
        jobs_deleted = cur.rowcount
        self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.conn.commit()
        return {"jobs": jobs_deleted, "files": files}
