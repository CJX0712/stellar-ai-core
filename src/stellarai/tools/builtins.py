"""内置工具：计算器、检索器、搜索占位。经注册表注入智能体。"""

from __future__ import annotations

import ast
import operator

from stellarai.agents.tool import Tool
from stellarai.core.config import StellarConfig
from stellarai.core.registry import provider

_ALLOWED = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval").body

    def ev(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("仅支持数值")
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED:
            return _ALLOWED[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED:
            return _ALLOWED[type(n.op)](ev(n.operand))
        raise ValueError(f"不支持的表达式: {ast.dump(n)}")

    return ev(node)


@provider("tool", "calculator")
def _make_calculator(config: StellarConfig, store=None) -> Tool:
    return CalculatorTool()


class CalculatorTool(Tool):
    name = "calculator"
    description = "计算数学表达式，例如 2+3*4"

    def run(self, action_input: str) -> str:
        try:
            return str(_safe_eval(action_input))
        except Exception as e:  # noqa: BLE001
            return f"计算错误: {e}"


@provider("tool", "retriever")
def _make_retriever(config: StellarConfig, store=None) -> Tool:
    return RetrieverTool(config, store)


class RetrieverTool(Tool):
    name = "retriever"
    description = "在已摄取的知识库中检索与查询最相关的资料"

    def __init__(self, config: StellarConfig, store=None) -> None:
        self.config = config
        self.store = store

    def run(self, action_input: str) -> str:
        if self.store is None:
            return "未提供向量存储，无法检索。"
        from stellarai.models.embedder import build_embedder

        embedder = build_embedder(self.config)
        vec = embedder.embed_one(action_input)
        results = self.store.query(vec, self.config.top_k)
        if not results:
            return "未检索到相关资料。"
        return "\n".join(f"[{i + 1}] {r.chunk.text}" for i, r in enumerate(results))


@provider("tool", "search")
def _make_search(config: StellarConfig, store=None) -> Tool:
    return SearchStubTool()


class SearchStubTool(Tool):
    name = "search"
    description = "占位搜索工具（演示用，不联网）"

    def run(self, action_input: str) -> str:
        return f"[stub] 针对 '{action_input}' 的搜索结果暂无（演示环境未接入真实搜索引擎）。"
