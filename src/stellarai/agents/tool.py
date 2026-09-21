"""工具接口：智能体可调用的最小能力单元。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, action_input: str) -> str:
        """执行工具，返回观察结果文本。"""
