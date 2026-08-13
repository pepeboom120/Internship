from pathlib import Path

import pandas as pd

from stem_analytics.config import load_config
from stem_analytics.data_validation import validate_and_save


def test_validation_stage_writes_controlled_outputs(tmp_path: Path) -> None:
    rows = []
    for subject in ["math", "physics", "chemistry", "biology"]:
        for index in range(8):
            rows.append(
                {
                    "question_id": len(rows),
                    "question": f"Unique {subject} question {index}",
                    "category": subject,
                    "src": "fixture",
                }
            )
    manifest = validate_and_save(pd.DataFrame(rows), load_config(Path("configs/project.yaml")), tmp_path)
    assert manifest["clean_rows"] == 32
    assert manifest["crossing_duplicate_groups"] == 0
    assert (tmp_path / "data/processed/questions.parquet").exists()
    assert (tmp_path / "reports/tables/exclusion_register.csv").exists()
    assert (tmp_path / "artifacts/manifests/validation_manifest.json").exists()
