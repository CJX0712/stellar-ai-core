from __future__ import annotations

import pytest

from stellarai.core.registry import Registry, provider


def test_register_and_build():
    from stellarai.core.registry import registry

    @provider("svc", "a")
    def make():
        return "A"

    registry.register("svc", "b", lambda: "B")
    assert registry.build("svc", "a") == "A"
    assert registry.build("svc", "b") == "B"
    assert set(registry.available("svc")) >= {"a", "b"}


def test_build_unknown_raises():
    reg = Registry()
    with pytest.raises(KeyError):
        reg.build("svc", "missing")


def test_global_registry_has_builtins():
    from stellarai.core.registry import registry

    assert "mock" in registry.available("llm")
    assert "mock" in registry.available("embedder")
    assert "numpy" in registry.available("vector_store")
    assert "calculator" in registry.available("tool")
