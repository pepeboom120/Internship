import tomllib
from pathlib import Path

from stem_analytics.cli import _stage_outputs, build_parser
from stem_analytics.config import load_config
from stem_analytics.provenance import artifact_is_current, write_json_atomic


def test_run_all_parser_accepts_force() -> None:
    args = build_parser().parse_args(["run-all", "--force"])
    assert args.command == "run-all"
    assert args.force is True


def test_changed_input_invalidates_stage_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "stage.json"
    write_json_atomic(metadata, {"config_hash": "a", "input_hash": "b"})
    assert artifact_is_current(metadata, {"config_hash": "a", "input_hash": "b"})
    assert not artifact_is_current(metadata, {"config_hash": "a", "input_hash": "changed"})


def test_stage_contracts_include_public_report() -> None:
    config = load_config(Path("configs/project.yaml"))
    assert Path("docs/final_report.md") in _stage_outputs("report", config)


def test_public_release_is_version_1_0_0() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "1.0.0"


def test_release_notes_report_verified_result_and_boundary() -> None:
    release_notes = Path("docs/release_notes_v1.0.0.md").read_text(encoding="utf-8")
    assert "0.8755" in release_notes
    assert "offline prototype" in release_notes.lower()
