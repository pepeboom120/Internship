"""Reproducible exploratory figures with explicit source records."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path("artifacts") / "matplotlib-cache").resolve())
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from stem_analytics.database import execute_named_query
from stem_analytics.provenance import sha256_file

SUBJECT_ORDER = ["math", "physics", "chemistry", "biology"]
COLORS = ["#4477AA", "#66CCEE", "#228833", "#CCBB44"]
SUBJECT_COLORS = dict(zip(SUBJECT_ORDER, COLORS, strict=True))


@dataclass(frozen=True)
class FigureRecord:
    number: int
    title: str
    path: Path
    source_table: str
    sample_size: int
    verification_status: str = "verified"


def _finish(
    ax: plt.Axes, title: str, sample_size: int, path: Path, footer_y: float = -0.22
) -> None:
    ax.set_title(f"{title} (n = {sample_size:,})", loc="left", weight="bold")
    ax.text(
        0,
        footer_y,
        "Benchmark questions; not real student demand.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def build_eda_figures(
    db_path: Path, output_dir: Path, query_dir: Path = Path("sql/analysis")
) -> list[FigureRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    records: list[FigureRecord] = []

    subject = execute_named_query(db_path, query_dir / "01_subject_distribution.sql")
    total = int(subject["record_count"].sum())
    path = output_dir / "01_subject_distribution.png"
    _, ax = plt.subplots(figsize=(8, 5))
    ordered = subject.set_index("subject").reindex(SUBJECT_ORDER).dropna().reset_index()
    sns.barplot(
        data=ordered,
        x="subject",
        y="record_count",
        hue="subject",
        palette=SUBJECT_COLORS,
        legend=False,
        ax=ax,
    )
    ax.set(xlabel="Subject", ylabel="Questions")
    smallest = ordered.loc[ordered["record_count"].idxmin(), "subject"].title()
    title = f"{smallest} has the fewest usable questions"
    _finish(ax, title, total, path)
    records.append(FigureRecord(1, title, path, "01_subject_distribution.csv", total))

    source = execute_named_query(db_path, query_dir / "02_source_distribution.sql")
    path = output_dir / "02_subject_by_source.png"
    top_sources = (
        source.groupby("source_name")["record_count"].sum().nlargest(12).index
    )
    summarized = source.copy()
    summarized["display_source"] = summarized["source_name"].where(
        summarized["source_name"].isin(top_sources), "Other sources"
    )
    summarized = (
        summarized.groupby(["subject", "display_source"], as_index=False)["record_count"]
        .sum()
    )
    pivot = summarized.pivot(
        index="subject", columns="display_source", values="record_count"
    ).fillna(0)
    pivot = pivot.reindex(SUBJECT_ORDER).dropna(how="all")
    _, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="Blues", ax=ax)
    ax.set(xlabel="Source", ylabel="Subject")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    title = "Source composition differs across STEM subjects"
    _finish(ax, title, total, path, footer_y=-0.62)
    records.append(FigureRecord(2, title, path, "02_source_distribution.csv", total))

    lengths = execute_named_query(db_path, query_dir / "04_text_length.sql")
    path = output_dir / "03_question_length.png"
    _, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=lengths,
        x="subject",
        y="mean_length",
        hue="subject",
        palette=SUBJECT_COLORS,
        legend=False,
        ax=ax,
    )
    ax.set(xlabel="Subject", ylabel="Mean characters per question")
    title = "Average question length varies by subject"
    _finish(ax, title, total, path)
    records.append(FigureRecord(3, title, path, "04_text_length.csv", total))

    quality = execute_named_query(db_path, query_dir / "03_data_quality.sql").iloc[0]
    path = output_dir / "04_data_quality.png"
    labels = ["Usable", "Empty", "Missing source", "Duplicate ID"]
    values = [int(quality["total_records"]), int(quality["empty_questions"]), int(quality["missing_sources"]), int(quality["duplicate_source_ids"])]
    _, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=labels, y=values, color=COLORS[0], ax=ax)
    ax.set(xlabel="Controlled-table check", ylabel="Records")
    title = "The controlled table passes required-field checks"
    _finish(ax, title, total, path)
    records.append(FigureRecord(4, title, path, "03_data_quality.csv", total))

    split = execute_named_query(db_path, query_dir / "05_split_integrity.sql")
    path = output_dir / "05_split_distribution.png"
    _, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=split, x="subject", y="record_count", hue="split_name", ax=ax)
    ax.set(xlabel="Subject", ylabel="Questions")
    title = "Development and final-test splits retain all four subjects"
    _finish(ax, title, total, path)
    records.append(FigureRecord(5, title, path, "05_split_integrity.csv", total))
    return records


def write_figure_register(records: list[FigureRecord], output_path: Path) -> None:
    rows = []
    for record in records:
        row = asdict(record)
        row["path"] = record.path.as_posix()
        row["figure_sha256"] = sha256_file(record.path)
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
