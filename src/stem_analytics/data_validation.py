"""Data validation, conservative normalization, and duplicate grouping."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from stem_analytics.config import ProjectConfig
from stem_analytics.provenance import config_hash, sha256_file, write_json_atomic


def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def validate_and_clean(
    frame: pd.DataFrame, config: ProjectConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"question_id", "question", "category", "src"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"question data is missing required columns: {missing}")
    work = frame.copy()
    work["question_text"] = work["question"].fillna("").map(normalize_question)
    duplicate_ids = work["question_id"].duplicated(keep="first")
    empty = work["question_text"].eq("")
    invalid = ~work["category"].isin(config.subjects)
    missing_source = work["src"].fillna("").astype(str).str.strip().eq("")
    reasons = pd.Series(pd.NA, index=work.index, dtype="string")
    reasons.loc[empty] = "empty_question"
    reasons.loc[invalid & reasons.isna()] = "invalid_subject"
    reasons.loc[duplicate_ids & reasons.isna()] = "duplicate_question_id"
    reasons.loc[missing_source & reasons.isna()] = "missing_source"
    exact = work["question_text"].duplicated(keep="first") & reasons.isna()
    reasons.loc[exact] = "exact_duplicate"
    excluded = work.loc[reasons.notna()].copy()
    excluded["exclusion_reason"] = reasons.loc[reasons.notna()]
    clean = work.loc[reasons.isna()].copy()
    clean["text_length"] = clean["question_text"].str.len()
    return clean.sort_values("question_id").reset_index(drop=True), excluded.reset_index(drop=True)


def assign_duplicate_groups(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    size = len(result)
    parents = list(range(size))
    observed_similarity = [0.0] * size

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    if size > 1:
        matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=config.duplicates.char_ngram_range
        ).fit_transform(result["question_text"])
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix)
        distances, indices = neighbors.radius_neighbors(
            matrix, radius=1 - config.duplicates.cosine_similarity_threshold
        )
        for row, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
            for distance, other in zip(row_distances, row_indices, strict=True):
                if row != other and distance <= 1 - config.duplicates.cosine_similarity_threshold:
                    similarity = 1 - float(distance)
                    observed_similarity[row] = max(observed_similarity[row], similarity)
                    observed_similarity[int(other)] = max(
                        observed_similarity[int(other)], similarity
                    )
                    union(row, int(other))
    roots = [find(index) for index in range(size)]
    result["duplicate_group_id"] = [int(result.loc[root, "question_id"]) for root in roots]
    result["duplicate_group_size"] = result.groupby("duplicate_group_id")["question_id"].transform("size")
    result["duplicate_group_max_similarity"] = observed_similarity
    result["duplicate_group_max_similarity"] = result.groupby("duplicate_group_id")[
        "duplicate_group_max_similarity"
    ].transform("max")
    return result


def validate_and_save(
    frame: pd.DataFrame, config: ProjectConfig, root: Path = Path(".")
) -> dict[str, Any]:
    from stem_analytics.splitting import assert_split_integrity, assign_project_split

    clean, excluded = validate_and_clean(frame, config)
    grouped = assign_duplicate_groups(clean, config)
    controlled = assign_project_split(grouped, config)
    assert_split_integrity(controlled)

    processed_path = root / config.paths.processed_data
    exclusion_path = root / config.paths.reports / "tables" / "exclusion_register.csv"
    manifest_path = root / config.paths.artifacts / "manifests" / "validation_manifest.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    controlled.to_parquet(processed_path, index=False)
    excluded.to_csv(exclusion_path, index=False, encoding="utf-8")

    crossing = int(
        controlled.groupby("duplicate_group_id")["split_name"].nunique().gt(1).sum()
    )
    manifest: dict[str, Any] = {
        "stage": "validate",
        "dataset_revision": config.dataset.revision,
        "config_hash": config_hash(config),
        "source_rows": len(frame),
        "clean_rows": len(controlled),
        "excluded_rows": len(excluded),
        "exclusion_counts": {
            str(key): int(value)
            for key, value in excluded["exclusion_reason"].value_counts().sort_index().items()
        },
        "subject_counts": {
            str(key): int(value)
            for key, value in controlled["category"].value_counts().sort_index().items()
        },
        "split_counts": {
            str(key): int(value)
            for key, value in controlled["split_name"].value_counts().sort_index().items()
        },
        "duplicate_groups": int(controlled["duplicate_group_id"].nunique()),
        "crossing_duplicate_groups": crossing,
        "near_duplicate_threshold": config.duplicates.cosine_similarity_threshold,
        "processed_file": processed_path.relative_to(root).as_posix(),
        "processed_sha256": sha256_file(processed_path),
        "exclusion_file": exclusion_path.relative_to(root).as_posix(),
        "exclusion_sha256": sha256_file(exclusion_path),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
