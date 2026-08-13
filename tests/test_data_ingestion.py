from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import stem_analytics.data_ingestion as ingestion
from stem_analytics.config import load_config
from stem_analytics.data_ingestion import build_source_manifest, select_stem_subjects


def source_frame() -> pd.DataFrame:
    rows = []
    for subject in ["math", "physics", "chemistry", "biology"]:
        for index in range(2):
            rows.append({"question_id": len(rows), "question": f"Synthetic {subject} {index}", "options": ["A", "B"], "answer": "A", "answer_index": 0, "cot_content": "Synthetic", "category": subject, "src": "fixture"})
    rows.append({"question_id": 99, "question": "Synthetic history", "options": ["A"], "answer": "A", "answer_index": 0, "cot_content": "Synthetic", "category": "history", "src": "fixture"})
    return pd.DataFrame(rows)


def test_select_stem_subjects_preserves_official_ids() -> None:
    selected = select_stem_subjects(source_frame(), ["math", "physics", "chemistry", "biology"])
    assert len(selected) == 8
    assert selected["question_id"].is_unique


def test_manifest_records_immutable_source() -> None:
    config = load_config(Path("configs/project.yaml"))
    manifest = build_source_manifest(source_frame(), config, datetime(2026, 8, 13, tzinfo=UTC))
    assert manifest["dataset_revision"] == config.dataset.revision
    assert manifest["source_split"] == "test"
    assert manifest["retrieved_at"] == "2026-08-13T00:00:00+00:00"


def test_loader_enables_system_trust_before_network(monkeypatch) -> None:
    events: list[str] = []

    class FakeDataset:
        def to_pandas(self) -> pd.DataFrame:
            return source_frame()

    monkeypatch.setattr(
        ingestion.truststore, "inject_into_ssl", lambda: events.append("trust")
    )
    monkeypatch.setattr(
        ingestion,
        "load_dataset",
        lambda *args, **kwargs: events.append("network") or FakeDataset(),
    )
    ingestion.load_source_dataset(load_config(Path("configs/project.yaml")))
    assert events[:2] == ["trust", "network"]
