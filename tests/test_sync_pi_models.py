# SPDX-License-Identifier: Apache-2.0
"""Tests for the Pi model sync script."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1].parent
SCRIPT = ROOT / "omlx" / "scripts" / "sync-pi-models.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_pi_models", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_models_are_not_tool_calling():
    mod = _load_module()
    entry = mod.build_entry("Qwen3-Embedding-8B-4bit-DWQ")
    assert entry["toolCalling"] is False


def test_qwen36_models_remain_tool_calling():
    mod = _load_module()
    entry = mod.build_entry("Qwen3.6-35B-A3B-4bit")
    assert entry["toolCalling"] is True
