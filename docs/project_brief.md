# Project Brief - STEM Subject Classification

## Decision

**Conditional go as a Validated Offline Prototype.** A leakage-safe text classifier can route benchmark questions among mathematics, physics, chemistry, and biology, but cross-source generalization and calibrated automation remain unverified.

## Problem and Data

The project tests whether question text alone can support four-class STEM routing. It pinned `TIGER-Lab/MMLU-Pro` revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`, selected 4499 rows, removed 91 exact duplicates, and retained 4408 controlled records. Answers, options, and explanations are prohibited model features.

## Workflow

`Pinned data -> validation -> duplicate grouping -> grouped split -> SQLite/SQL -> TF-IDF pipelines -> grouped CV tuning -> locked final test -> diagnostics -> verified report`

## Leakage Controls

- 0 duplicate groups cross development and final test.
- TF-IDF is fitted inside each training fold.
- 72 candidate parameter configurations are compared using development-only grouped CV.
- The final test is evaluated once after the model choice and evidence hashes are persisted.

## Verified Results

- Final macro-F1: **0.8755** (95% bootstrap CI 0.8471-0.8996)
- Accuracy: **0.8778** (553/630)
- Development grouped-CV macro-F1: **0.8807 +/- 0.0171**
- Strongest class: Math F1 0.9361; weakest: Chemistry F1 0.8391

## Engineering Evidence

Python package and CLI, 10 named SQL analyses, artifact hashes, locked run metadata, automated tests, GitHub Actions, clean-room installation, numerical consistency gates, 15-page technical report, and page-level PDF QA.

## Capability Boundaries

This is benchmark evidence, not validation of student behavior or semiconductor manufacturing. LinearSVC margins are not calibrated probabilities. The source is strongly confounded with subject, so production deployment and guaranteed domain transfer are not claimed.

## Review Links

[Repository](https://github.com/pepeboom120/Internship) | [Full report](final_report.pdf) | [Interview guide](interview_guide.md) | [Resume assets](resume_assets.md)
