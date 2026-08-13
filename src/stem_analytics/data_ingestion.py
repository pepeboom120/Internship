"""Pinned MMLU-Pro retrieval and source manifest creation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import truststore
from datasets import load_dataset

from stem_analytics.config import ProjectConfig
from stem_analytics.provenance import config_hash, sha256_file, write_json_atomic


class DatasetLike(Protocol):
    def to_pandas(self) -> pd.DataFrame: ...


def select_stem_subjects(frame: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    selected = frame.loc[frame["category"].isin(subjects)].copy()
    return selected.sort_values("question_id", kind="stable").reset_index(drop=True)


def load_source_dataset(config: ProjectConfig) -> pd.DataFrame:
    truststore.inject_into_ssl()
    dataset: DatasetLike = load_dataset(
        config.dataset.name,
        revision=config.dataset.revision,
        split=config.dataset.source_split,
    )
    frame = dataset.to_pandas()
    missing = sorted(set(config.dataset.required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"source dataset is missing required columns: {missing}")
    return select_stem_subjects(frame, config.subjects)


def build_source_manifest(
    frame: pd.DataFrame,
    config: ProjectConfig,
    retrieved_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = retrieved_at or datetime.now(UTC)
    return {
        "dataset_name": config.dataset.name,
        "dataset_revision": config.dataset.revision,
        "source_split": config.dataset.source_split,
        "retrieved_at": timestamp.isoformat(),
        "selected_rows": len(frame),
        "subject_counts": {
            str(key): int(value)
            for key, value in frame["category"].value_counts().sort_index().items()
        },
        "required_columns": config.dataset.required_columns,
        "config_hash": config_hash(config),
        "license": "MIT (per upstream dataset card; review upstream terms before redistribution)",
    }


def fetch_and_save(config: ProjectConfig, root: Path = Path(".")) -> dict[str, Any]:
    frame = load_source_dataset(config)
    output = root / config.paths.raw_data
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest = build_source_manifest(frame, config)
    manifest["source_file"] = output.as_posix()
    manifest["source_sha256"] = sha256_file(output)
    manifest_path = root / config.paths.artifacts / "manifests" / "source_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest
