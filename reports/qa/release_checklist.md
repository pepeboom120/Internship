# Public Release Checklist

- [x] Pinned public dataset revision and license note recorded
- [x] Raw data, SQLite, predictions, and model binaries ignored
- [x] Final test used only after locked selection
- [x] 0 duplicate groups cross development/final-test split
- [x] Numerical consistency register: 0 failures
- [x] Figure source register: 12 verified figures
- [x] PDF visual audit: 15/15 pages pass
- [x] One-page Project Brief visual audit: 1/1 page passes
- [x] Intent work described as future work only
- [x] README and report generated from authoritative artifacts
- [x] v1.0.0 release notes state the verified result and capability boundary
- [x] GitHub Actions covers Python 3.11 and 3.12 unit tests and Ruff
- [ ] GitHub Actions remote run (verify after merging or opening a PR)

Local release verification on 2026-08-13: Python 3.12.13; main environment 45/45 tests and Ruff passed; isolated `.verify-venv` installation, Ruff, and 45/45 tests passed. `stem-analytics run-all` confirmed that all full-data stage artifacts remain current under provenance checks. The full-data pipeline is exercised locally because CI deliberately does not download the benchmark or run the full hyperparameter grid. Numerical, figure-source, PDF, UTF-8, secret, tracked-file, and file-size gates all passed.
