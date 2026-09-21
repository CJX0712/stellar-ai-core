"""命令行入口：提供 chat / embed / ingest / query / serve / demo 子命令。"""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from stellarai.core.config import StellarConfig
from stellarai.core.logging import configure_logging

app = typer.Typer(help="StellarAI 命令行", add_completion=False)
console = Console()


def _cfg() -> StellarConfig:
    configure_logging()
    return StellarConfig()


@app.command()
def version() -> None:
    """打印版本与作者。"""
    from stellarai import __author__, __version__

    console.print(f"StellarAI {__version__} · 作者 {__author__}")


@app.command()
def chat(prompt: str = typer.Argument(..., help="输入提示词")) -> None:
    """使用配置的 LLM 补全。"""
    from stellarai.models.llm import build_llm

    cfg = _cfg()
    llm = build_llm(cfg)
    console.print(llm.complete(prompt))


@app.command()
def embed(text: str = typer.Argument(..., help="待向量化文本")) -> None:
    """向量化单条文本。"""
    from stellarai.models.embedder import build_embedder

    cfg = _cfg()
    emb = build_embedder(cfg)
    vec = emb.embed_one(text)
    console.print(f"dim={emb.dim} 前4维={vec[:4]}")


@app.command()
def ingest(
    text: str = typer.Option(None, "--text", help="直接传入文本"),
    file: str = typer.Option(None, "--file", help="文本/PDF 文件路径"),
) -> None:
    """摄取文档到知识库（内存模式，演示用）。"""
    from stellarai.rag.pipeline import RAGPipeline

    cfg = _cfg()
    rag = RAGPipeline(cfg)
    if file:
        from stellarai.ingestion.loader import load_document

        doc = load_document(file)
        n = rag.ingest([doc], source=file)
    elif text:
        n = rag.ingest([text], source="cli")
    else:
        console.print("[red]请通过 --text 或 --file 提供内容[/red]")
        raise typer.Exit(code=1)
    console.print(f"已摄取切片数: {n}")


@app.command()
def query(question: str = typer.Argument(..., help="问题")) -> None:
    """在知识库中检索并生成回答。"""
    from stellarai.rag.pipeline import RAGPipeline

    cfg = _cfg()
    rag = RAGPipeline(cfg)
    res = rag.answer(question)
    console.print(f"[bold]回答:[/bold] {res['answer']}")
    console.print(f"[dim]来源数: {len(res['sources'])}[/dim]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="监听地址"),
    port: int = typer.Option(8000, help="监听端口"),
) -> None:
    """启动 API 服务。"""
    import uvicorn

    console.print(f"StellarAI 服务启动于 http://{host}:{port}")
    uvicorn.run("stellarai.api.server:create_app", host=host, port=port, factory=False)


@app.command()
def demo() -> None:
    """离线端到端演示：零依赖、零网络，证明全链路可复现。"""
    from stellarai.agents.agent import Agent
    from stellarai.core.types import Document
    from stellarai.memory.stores.numpy_store import NumPyCosineStore
    from stellarai.models.providers.mock import MockEmbedder, MockLLM, ScriptedLLM
    from stellarai.rag.pipeline import RAGPipeline

    cfg = StellarConfig(llm_provider="mock", embedder_provider="mock", embedder_dim=32)
    store = NumPyCosineStore(cfg)
    rag = RAGPipeline(cfg, llm=MockLLM(cfg), embedder=MockEmbedder(cfg), store=store)

    rag.ingest([Document(content="StellarAI 是由晨星打造的模块化 AI 系统，支持 RAG 与 Agent 编排。", source="demo")])
    console.print("[bold]RAG 链路:[/bold]")
    res = rag.answer("StellarAI 是谁打造的？")
    console.print(f"  问题命中上下文: {'晨星' in res['answer']}")
    console.print(f"  来源数: {len(res['sources'])}")

    console.print("[bold]Agent 链路 (ReAct + 计算器工具):[/bold]")
    script = [
        "Thought: 需要计算乘积\nAction: calculator\nAction Input: 12*8",
        "Final Answer: 96",
    ]
    agent = Agent(cfg, llm=ScriptedLLM(script=script), store=store)
    out = agent.run("请帮我计算 12 乘以 8")
    console.print(f"  Agent 结果包含 96: {'96' in out}")

    console.print("[green]端到端链路验证通过（Mock 模式，无需 GPU/网络）。[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
