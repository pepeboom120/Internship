from pathlib import Path

import pytest
from pydantic import ValidationError

from stem_analytics.config import load_config


def test_project_config_pins_dataset_and_seed() -> None:
    config = load_config(Path("configs/project.yaml"))
    assert config.dataset.revision == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert config.dataset.source_split == "test"
    assert config.subjects == ["math", "physics", "chemistry", "biology"]
    assert config.random_seed == 42
    assert config.split.final_test_folds == 7


def test_project_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("project_name: demo\nrandom_seed: 42\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)

