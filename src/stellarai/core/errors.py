"""统一异常体系。"""

from __future__ import annotations


class StellarError(Exception):
    """所有 StellarAI 错误的基类。"""


class ConfigError(StellarError):
    """配置错误。"""


class ProviderError(StellarError):
    """组件/Provider 构建或调用错误。"""


class RegistryError(StellarError):
    """注册表相关错误。"""


class IngestionError(StellarError):
    """文档摄取错误。"""


class RetrievalError(StellarError):
    """检索错误。"""
