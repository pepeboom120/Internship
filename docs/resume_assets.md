# Resume and Application Assets

## Project Title

**Reproducible End-to-End ML Workflow | Python, SQL, scikit-learn, CI/CD**

## English Resume Bullets

- Built a reproducible end-to-end ML workflow for four-class STEM text routing across 4499 pinned records, covering schema validation, duplicate-aware grouped splitting, SQLite analytics, hyperparameter tuning, error analysis, and automated reporting.
- Evaluated 72 TF-IDF and linear-model parameter configurations using leakage-safe five-fold grouped cross-validation; the locked linear_svc achieved 0.8755 macro-F1 and 0.8778 accuracy on 630 isolated test records, with a 95% bootstrap interval of 0.8471-0.8996.
- Implemented 10 named SQL analyses, artifact hashing, one-time test evaluation, automated tests, clean-room verification, GitHub Actions, and a visually audited 15-page technical report with zero unresolved numerical discrepancies.

## 中文履歷敘述

- 建立可重現的端到端機器學習流程，處理 4499 筆固定版本資料，涵蓋 schema 驗證、重複群組隔離切分、SQLite 分析、超參數調整、錯誤診斷與自動化報告。
- 以五折 grouped cross-validation 比較 72 組 TF-IDF 與線性模型設定；鎖定的 linear_svc 在 630 筆獨立測試資料達到 macro-F1 0.8755、accuracy 0.8778，95% bootstrap 區間為 0.8471-0.8996。
- 實作 10 個具決策目的的 SQL 查詢、artifact hash、一次性測試評估、自動測試、clean-room 驗證、GitHub Actions 與逐頁 QA 的 15 頁技術報告。

## ML / Data Science Application Variant

Emphasize leakage-safe experimental design, grouped CV, macro-F1, bootstrap uncertainty, per-class errors, feature interpretation, and the decision not to claim calibrated confidence.

## IMC / Intelligent Manufacturing Application Variant

Emphasize the transferable workflow: controlled data ingestion, automated quality checks, high-dimensional feature pipelines, reproducible tuning, failure-slice diagnosis, SQL decision support, and lifecycle verification. Do not describe the benchmark itself as manufacturing validation.

## STAR Interview Outline

- **Situation:** A public ML portfolio needed to demonstrate trustworthy workflow quality rather than a frontend demo.
- **Task:** Determine whether question text alone could support reliable four-subject routing without contaminating the final test.
- **Action:** Pinned the source revision; validated, normalized, and deduplicated records; enforced grouped splits; tuned pipelines with grouped CV; locked the selection; evaluated once; diagnosed class, source, and length failures; automated evidence generation and QA.
- **Result:** The locked model reached macro-F1 0.8755 with a 0.8471-0.8996 interval, while the project explicitly documented source confounding, uncalibrated margins, and the next valid experiments.
