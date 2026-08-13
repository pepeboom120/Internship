import json
from pathlib import Path

from stem_analytics.provenance import (
    artifact_is_current,
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)


def test_canonical_hash_is_key_order_independent() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_atomic_json_and_file_hash(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "metadata.json"
    write_json_atomic(target, {"stage": "fetch", "count": 8})
    assert json.loads(target.read_text(encoding="utf-8"))["count"] == 8
    assert len(sha256_file(target)) == 64


def test_artifact_currency_requires_exact_metadata(tmp_path: Path) -> None:
    target = tmp_path / "metadata.json"
    write_json_atomic(target, {"stage": "fetch", "config_hash": "abc"})
    assert artifact_is_current(target, {"stage": "fetch", "config_hash": "abc"})
    assert not artifact_is_current(target, {"stage": "fetch", "config_hash": "changed"})
