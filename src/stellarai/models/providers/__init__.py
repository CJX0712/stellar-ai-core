"""内置 provider 工厂注册入口。导入即注册，重依赖在工厂内惰性导入。"""

from stellarai.models.providers import (
    llamacpp,  # noqa: F401
    mock,  # noqa: F401
    openai_compat,  # noqa: F401
)

__all__ = ["mock", "openai_compat", "llamacpp"]
