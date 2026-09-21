"""能力注册表（Capability Registry）+ 简单依赖注入。

组件按 (kind, name) 注册工厂函数；运行时按配置名字实例化，
实现"一行配置切换 SOTA 实现"，且新增组件不改业务代码。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """轻量组件注册表。"""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], Callable[..., T]] = {}

    def register(self, kind: str, name: str, factory: Callable[..., T]) -> None:
        self._factories[(kind, name)] = factory

    def build(self, kind: str, name: str, *args: Any, **kwargs: Any) -> T:
        key = (kind, name)
        if key not in self._factories:
            raise KeyError(
                f"未注册的 {kind} 实现: '{name}'。可用: {self.available(kind)}"
            )
        return self._factories[key](*args, **kwargs)

    def available(self, kind: str) -> list[str]:
        return [n for (k, n) in self._factories if k == kind]

    def has(self, kind: str, name: str) -> bool:
        return (kind, name) in self._factories


registry = Registry()


def provider(kind: str, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """装饰器：将工厂函数注册为指定 kind/name 的组件。"""

    def deco(factory: Callable[..., T]) -> Callable[..., T]:
        registry.register(kind, name, factory)
        return factory

    return deco
