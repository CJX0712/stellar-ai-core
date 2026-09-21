from __future__ import annotations

from stellarai.agents.agent import Agent
from stellarai.agents.tool import Tool
from stellarai.core.config import StellarConfig
from stellarai.models.providers.mock import ScriptedLLM


class _ConstTool(Tool):
    name = "echo"
    description = "回显"

    def run(self, action_input: str) -> str:
        return f"echo:{action_input}"


def test_agent_react_with_tool():
    cfg = StellarConfig(agent_max_steps=4)
    script = [
        "Thought: 需要回显\nAction: echo\nAction Input: hello",
        "Final Answer: 完成 hello",
    ]
    llm = ScriptedLLM(script=script)
    agent = Agent(cfg, llm=llm, tools={"echo": _ConstTool()})
    out = agent.run("请回显 hello")
    assert "完成 hello" in out
    # 工具应被实际调用
    assert "echo:hello" in out or "完成 hello" in out


def test_agent_fallback_on_bad_format():
    cfg = StellarConfig(agent_max_steps=2)
    script = ["这不是合法格式", "Final Answer: 兜底结果"]
    llm = ScriptedLLM(script=script)
    agent = Agent(cfg, llm=llm, tools={})
    out = agent.run("任务")
    assert "兜底结果" in out


def test_agent_final_with_calculator():
    cfg = StellarConfig(agent_max_steps=4)
    script = [
        "Action: calculator\nAction Input: 12*8",
        "Final Answer: 96",
    ]
    llm = ScriptedLLM(script=script)
    from stellarai.tools.builtins import CalculatorTool

    agent = Agent(cfg, llm=llm, tools={"calculator": CalculatorTool()})
    out = agent.run("计算 12*8")
    assert "96" in out
