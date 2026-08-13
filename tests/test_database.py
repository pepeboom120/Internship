import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from stem_analytics.config import load_config
from stem_analytics.database import (
    build_database_and_export,
    create_database,
    execute_named_query,
    load_questions,
)


def question_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"question_id": 1, "question_text": "Math question", "category": "math", "src": "fixture", "text_length": 13, "duplicate_group_id": 1, "duplicate_group_size": 1, "duplicate_group_max_similarity": 0.0, "split_name": "development"},
            {"question_id": 2, "question_text": "Physics question", "category": "physics", "src": "fixture", "text_length": 16, "duplicate_group_id": 2, "duplicate_group_size": 1, "duplicate_group_max_similarity": 0.0, "split_name": "test"},
        ]
    )


def test_database_enforces_subject_constraint(tmp_path: Path) -> None:
    database = tmp_path / "test.sqlite"
    create_database(database, Path("sql/schema.sql"))
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO questions VALUES (1, 1, 'History', 'history', 'x', 7, 1, 1, 0, 'development', 'rev')"
            )


def test_load_questions_and_named_query(tmp_path: Path) -> None:
    database = tmp_path / "test.sqlite"
    create_database(database, Path("sql/schema.sql"))
    assert load_questions(database, question_frame(), "revision") == 2
    result = execute_named_query(database, Path("sql/analysis/01_subject_distribution.sql"))
    assert result["record_count"].sum() == 2
    assert set(result["subject"]) == {"math", "physics"}


def test_atomic_database_build_replaces_temporary_file(tmp_path: Path) -> None:
    config = load_config(Path("configs/project.yaml"))
    (tmp_path / "sql").mkdir()
    (tmp_path / "sql" / "analysis").mkdir()
    (tmp_path / "sql" / "schema.sql").write_text(
        Path("sql/schema.sql").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "sql" / "analysis" / "01_subject_distribution.sql").write_text(
        Path("sql/analysis/01_subject_distribution.sql").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = build_database_and_export(question_frame(), config, tmp_path)
    assert result == {"questions_loaded": 2, "queries_exported": 1}
    assert (tmp_path / config.paths.database).exists()
