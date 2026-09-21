"""ReAct 风格智能体：推理-行动-观察 循环，工具经注册表注入。"""

from __future__ import annotations

import re

from stellarai.core.config import StellarConfig
from stellarai.core.registry import registry
from stellarai.core.types import Message
from stellarai.models.llm import LLMProvider


def build_agent(config: StellarConfig, **kwargs) -> Agent:
    return Agent(config, **kwargs)


class Agent:
    def __init__(
        self,
        config: StellarConfig,
        llm: LLMProvider | None = None,
        tools: dict | None = None,
        store=None,
    ) -> None:
        self.config = config
        self.llm: LLMProvider = llm or registry.build("llm", config.llm_provider, config)
        self.tools = tools if tools is not None else self._default_tools(config, store)
        self.max_steps = config.agent_max_steps

    def _default_tools(self, config: StellarConfig, store) -> dict:
        tools: dict = {}
        for name in registry.available("tool"):
            tools[name] = registry.build("tool", name, config, store)
        return tools

    def _system_prompt(self) -> str:
        names = ", ".join(self.tools.keys()) if self.tools else "(无)"
        return (
            "你是一个使用工具的智能体。可用工具: " + names + "。\n"
            "按以下格式推理：\n"
            "Thought: 你的思考\n"
            "Action: 工具名\n"
            "Action Input: 工具输入\n"
            "得到 Observation 后继续，直到能回答时输出：\n"
            "Final Answer: 最终答案\n"
        )

    def run(self, task: str) -> str:
        messages = [
            Message(role="system", content=self._system_prompt()),
            Message(role="user", content=task),
        ]
        last = ""
        for _ in range(self.max_steps):
            resp = self.llm.chat(messages)
            last = resp.text
            messages.append(Message(role="assistant", content=last))
            if "Final Answer:" in last:
                return self._extract_final(last)
            action, action_input = self._parse_action(last)
            if action and action in self.tools:
                observation = self.tools[action].run(action_input)
                messages.append(Message(role="user", content=f"Observation: {observation}"))
            else:
                messages.append(
                    Message(role="user", content="Observation: 未识别到可用工具或格式错误，请重新推理。")
                )
        return self._extract_final(last) or "未在步数限制内完成任务。"

    @staticmethod
    def _parse_action(text: str):
        action_m = re.search(r"Action:\s*(.+)", text)
        input_m = re.search(r"Action Input:\s*(.+)", text)
        action = action_m.group(1).strip() if action_m else None
        action_input = input_m.group(1).strip() if input_m else ""
        return action, action_input

    @staticmethod
    def _extract_final(text: str) -> str:
        m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        return m.group(1).strip() if m else ""
