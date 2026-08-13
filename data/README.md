# Data source and redistribution

This project downloads `TIGER-Lab/MMLU-Pro` from Hugging Face at immutable
revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`. The official `test` split is
used only as a source pool; this project constructs its own leakage-safe
development and final-test partitions for subject classification.

The upstream dataset card lists an MIT license. MMLU-Pro also consolidates
questions from several sources, so upstream source notes and current terms
must be reviewed before redistribution. Raw and processed rows are therefore
not committed. They are rebuilt locally from the pinned revision.

- Dataset card: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- Paper: https://arxiv.org/abs/2406.01574
