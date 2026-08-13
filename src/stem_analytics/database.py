"""SQLite construction and named analytical query execution."""

from __future__ import annotations

import json
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


def export_named_queries(
    db_path: Path,
    query_dir: Path,
    table_dir: Path,
    names: set[str] | None = None,
) -> int:
    """Export selected named SQL analyses, or all analyses when names is omitted."""
    table_dir.mkdir(parents=True, exist_ok=True)
    query_count = 0
    for query in sorted(query_dir.glob("*.sql")):
        if names is not None and query.name[:2] not in names:
            continue
        execute_named_query(db_path, query).to_csv(
            table_dir / f"{query.stem}.csv", index=False, encoding="utf-8"
        )
        query_count += 1
    return query_count


def load_model_run_and_predictions(
    db_path: Path,
    selection: dict,
    metrics: dict,
    predictions: pd.DataFrame,
    random_seed: int,
) -> tuple[int, int]:
    """Transactionally replace one external run and its final-test predictions."""
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            existing = connection.execute(
                "SELECT run_id FROM model_runs WHERE external_run_id = ?",
                (selection["run_id"],),
            ).fetchone()
            if existing:
                database_run_id = int(existing[0])
                connection.execute(
                    "DELETE FROM predictions WHERE run_id = ?", (database_run_id,)
                )
                connection.execute(
                    """
                    UPDATE model_runs SET model_name = ?, parameters_json = ?,
                        random_seed = ?, cv_macro_f1 = ?, test_macro_f1 = ?,
                        test_weighted_f1 = ?, test_accuracy = ?, runtime_seconds = ?,
                        created_at = ? WHERE run_id = ?
                    """,
                    (
                        selection["selected_model"],
                        json.dumps(selection["best_params"], sort_keys=True),
                        random_seed,
                        selection["cv_macro_f1_mean"],
                        metrics["macro_f1"],
                        metrics["weighted_f1"],
                        metrics["accuracy"],
                        metrics["runtime_seconds"],
                        metrics["evaluated_at_utc"],
                        database_run_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO model_runs (
                        external_run_id, task_name, model_name, parameters_json,
                        random_seed, cv_macro_f1, test_macro_f1, test_weighted_f1,
                        test_accuracy, runtime_seconds, created_at
                    ) VALUES (?, 'subject', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        selection["run_id"],
                        selection["selected_model"],
                        json.dumps(selection["best_params"], sort_keys=True),
                        random_seed,
                        selection["cv_macro_f1_mean"],
                        metrics["macro_f1"],
                        metrics["weighted_f1"],
                        metrics["accuracy"],
                        metrics["runtime_seconds"],
                        metrics["evaluated_at_utc"],
                    ),
                )
                database_run_id = int(cursor.lastrowid)

            question_ids = dict(
                connection.execute(
                    "SELECT source_question_id, question_id FROM questions"
                ).fetchall()
            )
            records = []
            for row in predictions.itertuples(index=False):
                source_id = int(row.source_question_id)
                if source_id not in question_ids:
                    raise ValueError(f"prediction references unknown question: {source_id}")
                confidence = None if pd.isna(row.confidence) else float(row.confidence)
                records.append(
                    (
                        database_run_id,
                        question_ids[source_id],
                        str(row.true_label),
                        str(row.predicted_label),
                        confidence,
                        int(row.is_correct),
                    )
                )
            connection.executemany(
                """
                INSERT INTO predictions (
                    run_id, question_id, true_label, predicted_label, confidence, is_correct
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                records,
            )
    return database_run_id, len(records)


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

    query_count = export_named_queries(
        target,
        root / "sql" / "analysis",
        root / config.paths.reports / "tables",
    )
    return {"questions_loaded": loaded, "queries_exported": query_count}
