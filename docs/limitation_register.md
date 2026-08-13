# Limitation Register

| Limitation | Evidence | Consequence | Next action |
|---|---|---|---|
| Benchmark, not learner data | MMLU-Pro provenance | No student-behavior claim | Validate on consented target-domain data |
| Source–subject confounding | Source distribution and holdout eligibility | Cross-source transfer uncertain | Collect multi-subject data per provider |
| LinearSVC is uncalibrated | No probability output | No automated confidence threshold | Predeclare held-out calibration protocol |
| English text only | Feature contract | No image/non-English support | Add separately evaluated modalities |
| Four subjects only | Label constraint | No open-set detection | Add rejection/OOD evaluation |
