"""SQLite construction and named analytical query execution."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd

from stem_analytics.config import ProjectConfig


def create_database(db_path: Path, schema_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        connection.execute("PRAGMA foreign_keys = ON")


def load_questions(db_path: Path, frame: pd.DataFrame, dataset_revision: str) -> int:
    records = []
    for local_id, row in enumerate(frame.itertuples(index=False), start=1):
        records.append(
            (
                local_id,
                int(row.question_id),
                str(row.question_text),
                str(row.category),
                str(row.src),
                int(row.text_length),
                int(row.duplicate_group_id),
                int(row.duplicate_group_size),
                float(row.duplicate_group_max_similarity),
                str(row.split_name),
                dataset_revision,
            )
        )
    statement = """
        INSERT INTO questions (
            question_id, source_question_id, question_text, subject, source_name,
            text_length, duplicate_group_id, duplicate_group_size,
            duplicate_group_max_similarity, split_name, dataset_revision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.executemany(statement, records)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"foreign-key violations: {violations}")
    return len(records)


def execute_named_query(db_path: Path, query_path: Path) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as connection:
        return pd.read_sql_query(query_path.read_text(encoding="utf-8"), connection)


def build_database_and_export(
    frame: pd.DataFrame, config: ProjectConfig, root: Path = Path(".")
) -> dict[str, int]:
    target = root / config.paths.database
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".sqlite")
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        create_database(temp_path, root / "sql" / "schema.sql")
        loaded = load_questions(temp_path, frame, config.dataset.revision)
        with closing(sqlite3.connect(temp_path)) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(f"SQLite integrity check failed: {integrity}")
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    table_dir = root / config.paths.reports / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    query_count = 0
    for query in sorted((root / "sql" / "analysis").glob("*.sql")):
        execute_named_query(target, query).to_csv(
            table_dir / f"{query.stem}.csv", index=False, encoding="utf-8"
        )
        query_count += 1
    return {"questions_loaded": loaded, "queries_exported": query_count}
