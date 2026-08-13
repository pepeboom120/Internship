from pathlib import Path

import pandas as pd

from stem_analytics.analysis import build_eda_figures, matplotlib
from stem_analytics.database import create_database, load_questions


def test_matplotlib_cache_is_workspace_local() -> None:
    assert Path(matplotlib.get_configdir()).name == "matplotlib-cache"


def test_build_eda_figures_creates_expected_files(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    create_database(database, Path("sql/schema.sql"))
    frame = pd.DataFrame(
        [
            {"question_id": 1, "question_text": "Math question", "category": "math", "src": "fixture", "text_length": 13, "duplicate_group_id": 1, "duplicate_group_size": 1, "duplicate_group_max_similarity": 0.0, "split_name": "development"},
            {"question_id": 2, "question_text": "Physics question", "category": "physics", "src": "fixture", "text_length": 16, "duplicate_group_id": 2, "duplicate_group_size": 1, "duplicate_group_max_similarity": 0.0, "split_name": "test"},
        ]
    )
    load_questions(database, frame, "revision")
    records = build_eda_figures(database, tmp_path / "figures", Path("sql/analysis"))
    assert [record.path.name for record in records] == [
        "01_subject_distribution.png",
        "02_subject_by_source.png",
        "03_question_length.png",
        "04_data_quality.png",
        "05_split_distribution.png",
    ]
    assert all(record.path.stat().st_size > 0 for record in records)
    assert all(record.sample_size > 0 for record in records)
