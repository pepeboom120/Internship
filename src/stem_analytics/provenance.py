"""Deterministic hashing and artifact metadata helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def artifact_is_current(meta_path: Path, expected: Mapping[str, Any]) -> bool:
    if not meta_path.exists():
        return False
    actual = json.loads(meta_path.read_text(encoding="utf-8"))
    return all(actual.get(key) == value for key, value in expected.items())


def config_hash(config: Any) -> str:
    payload = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    return canonical_sha256(payload)
