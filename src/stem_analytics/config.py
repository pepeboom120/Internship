"""Strict project configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictModel):
    name: str
    revision: str
    source_split: str
    required_columns: list[str]


class SplitConfig(StrictModel):
    final_test_folds: int
    cv_folds: int


class DuplicateConfig(StrictModel):
    char_ngram_range: tuple[int, int]
    cosine_similarity_threshold: float


class ModelConfig(StrictModel):
    tfidf_ngram_ranges: list[tuple[int, int]]
    tfidf_min_df: list[int]
    c_values: list[float]
    class_weights: list[Literal["balanced"] | None]


class EvaluationConfig(StrictModel):
    primary_metric: Literal["macro_f1"]
    bootstrap_iterations: int


class PathConfig(StrictModel):
    raw_data: Path
    processed_data: Path
    database: Path
    artifacts: Path
    reports: Path


class ProjectConfig(StrictModel):
    project_name: str
    random_seed: int
    subjects: list[str]
    dataset: DatasetConfig
    split: SplitConfig
    duplicates: DuplicateConfig
    models: ModelConfig
    evaluation: EvaluationConfig
    paths: PathConfig


def load_config(path: Path) -> ProjectConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(payload)

