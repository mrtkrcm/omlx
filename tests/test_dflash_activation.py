from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE_POOL_PATH = ROOT / "omlx" / "engine_pool.py"
OMLX_DIR = ROOT / "omlx"


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _StubBaseEngine:
    pass


class _StubBatchedEngine:
    pass


class _StubEmbeddingEngine:
    pass


class _StubRerankerEngine:
    pass


class _StubSTTEngine:
    pass


class _StubSTSEngine:
    pass


class _StubTTSEngine:
    pass


class _StubVLMBatchedEngine:
    pass


class _StubDFlashEngine:
    pass


def _load_engine_pool_module():
    if "omlx" not in sys.modules:
        pkg = types.ModuleType("omlx")
        pkg.__path__ = [str(OMLX_DIR)]
        sys.modules["omlx"] = pkg

    engine_pkg = _install_stub_module(
        "omlx.engine",
        __path__=[],
        BaseEngine=_StubBaseEngine,
        BatchedEngine=_StubBatchedEngine,
        EmbeddingEngine=_StubEmbeddingEngine,
        RerankerEngine=_StubRerankerEngine,
        STTEngine=_StubSTTEngine,
        STSEngine=_StubSTSEngine,
        TTSEngine=_StubTTSEngine,
        VLMBatchedEngine=_StubVLMBatchedEngine,
        DFlashEngine=_StubDFlashEngine,
    )
    _install_stub_module("omlx.engine.base", BaseEngine=_StubBaseEngine)
    _install_stub_module("omlx.engine.batched", BatchedEngine=_StubBatchedEngine)
    _install_stub_module("omlx.engine.embedding", EmbeddingEngine=_StubEmbeddingEngine)
    _install_stub_module("omlx.engine.reranker", RerankerEngine=_StubRerankerEngine)
    _install_stub_module("omlx.engine.stt", STTEngine=_StubSTTEngine)
    _install_stub_module("omlx.engine.sts", STSEngine=_StubSTSEngine)
    _install_stub_module("omlx.engine.tts", TTSEngine=_StubTTSEngine)
    _install_stub_module("omlx.engine.vlm", VLMBatchedEngine=_StubVLMBatchedEngine)

    spec = importlib.util.spec_from_file_location("omlx.engine_pool", ENGINE_POOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {ENGINE_POOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["omlx.engine_pool"] = module
    spec.loader.exec_module(module)
    return module


class _DummySettings:
    def __init__(self, dflash_enabled=False, dflash_draft_model=None, dflash_draft_quant_bits=None, dflash_max_ctx=None):
        self.dflash_enabled = dflash_enabled
        self.dflash_draft_model = dflash_draft_model
        self.dflash_draft_quant_bits = dflash_draft_quant_bits
        self.dflash_max_ctx = dflash_max_ctx


def test_engine_pool_selects_dflash_when_enabled(monkeypatch):
    engine_pool = _load_engine_pool_module()
    pool = engine_pool.EnginePool(max_model_memory=None)
    pool._entries["demo"] = SimpleNamespace(
        model_id="demo",
        model_path="demo-path",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        config_model_type="llama",
        thinking_default=None,
        engine=None,
        last_access=0.0,
        is_loading=False,
        is_pinned=False,
        abort_loading=False,
    )
    pool._settings_manager = SimpleNamespace(
        get_settings=lambda mid: _DummySettings(True, "draft-model", 1234, 777)
    )

    created = {}

    class FakeDFlash:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.started = False

        async def start(self):
            self.started = True

        async def stop(self):
            return None

    _install_stub_module("omlx.engine.dflash", DFlashEngine=FakeDFlash)
    monkeypatch.setattr(engine_pool, "BatchedEngine", lambda **kwargs: (_ for _ in ()).throw(AssertionError("BatchedEngine should not be used")))
    monkeypatch.setattr(engine_pool, "get_mlx_executor", lambda: None)

    asyncio.run(pool._load_engine("demo"))

    assert pool._entries["demo"].engine is not None
    assert created["model_name"] == "demo-path"
    assert created["draft_model_path"] == "draft-model"
    assert created["draft_quant_bits"] == 1234
    assert created["fallback_engine_type"] == "batched"
    assert created["scheduler_config"] is pool._scheduler_config
    assert getattr(pool._entries["demo"].engine, "started", False) is True


def test_engine_pool_uses_batched_when_dflash_disabled(monkeypatch):
    engine_pool = _load_engine_pool_module()
    pool = engine_pool.EnginePool(max_model_memory=None)
    pool._entries["demo"] = SimpleNamespace(
        model_id="demo",
        model_path="demo-path",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        config_model_type="llama",
        thinking_default=None,
        engine=None,
        last_access=0.0,
        is_loading=False,
        is_pinned=False,
        abort_loading=False,
    )
    pool._settings_manager = SimpleNamespace(
        get_settings=lambda mid: _DummySettings(False, None, None, None)
    )

    called = {}

    class FakeBatched:
        def __init__(self, **kwargs):
            called.update(kwargs)
            self.started = False

        async def start(self):
            self.started = True

        async def stop(self):
            return None

    monkeypatch.setattr(engine_pool, "BatchedEngine", FakeBatched)
    _install_stub_module(
        "omlx.engine.dflash",
        DFlashEngine=lambda **kwargs: (_for := (_ for _ in ()), (_ for _ in ()))[1].throw(AssertionError("DFlashEngine should not be used")),
    )
    monkeypatch.setattr(engine_pool, "get_mlx_executor", lambda: None)

    asyncio.run(pool._load_engine("demo"))

    assert pool._entries["demo"].engine is not None
    assert called["model_name"] == "demo-path"
    assert called["scheduler_config"] is pool._scheduler_config
    assert called["model_settings"].dflash_enabled is False
    assert getattr(pool._entries["demo"].engine, "started", False) is True
