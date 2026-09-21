# StellarAI 系统架构

本文档说明 StellarAI 的模块划分、接口契约、依赖关系与端到端数据流。

## 1. 设计原则

1. **单一职责**：每个模块只做一件事，对外仅暴露抽象接口。
2. **接口驱动**：模块间通过 `LLMProvider` / `Embedder` / `VectorStore` / `Tool` 等抽象基类通信，实现可替换、可独立验证。
3. **能力注册表 + 依赖注入**：组件以 `(kind, name)` 注册工厂；运行时按 `StellarConfig` 实例化，新增 SOTA 实现不改业务代码（一行配置切换）。
4. **零依赖可验证**：内置 Mock 适配器，无 GPU/网络即可跑通全链路并复现。

## 2. 模块与接口

```
CLI / API
   │
   ▼
Model Gateway ─── LLMProvider.chat() / Embedder.embed()
   │
   ▼
RAG Pipeline / Agent Orchestrator
   │
   ▼
Ingestion ──► Vector Store ──► (OSS 适配器: Chroma / FAISS / NumPy)
   │
   ▼
横切：Config & Registry · Observability
```

| 模块 | 抽象接口 | 内置实现 | 可选 OSS 适配 |
|------|----------|----------|----------------|
| LLM | `LLMProvider` | `MockLLM`, `ScriptedLLM` | OpenAI/vLLM/Ollama, llama-cpp-python |
| Embedding | `Embedder` | `MockEmbedder` | sentence-transformers, OpenAI |
| Vector Store | `VectorStore` | `NumPyCosineStore` | Chroma, FAISS |
| Loader | `Loader` | `TxtLoader`, `PdfLoader` | — |
| Chunker | `ChunkingStrategy` | `RecursiveCharChunker` | — |
| Agent | `Agent` / `Tool` | `CalculatorTool`, `RetrieverTool`, `SearchStubTool` | LangGraph(可选) |

## 3. 能力注册表

`core/registry.py` 提供 `Registry`：

```python
registry.register("llm", "mock", lambda config: MockLLM(config))
llm = registry.build("llm", config.llm_provider, config)
```

工厂在模块导入时注册；重依赖（torch / chromadb / llama_cpp）在工厂函数内**惰性导入**，故导入包本身不触发安装，保证核心轻量可装。

## 4. 端到端数据流

### RAG 链路（摄取 → 检索 → 生成）
1. `RAGPipeline.ingest(docs)`：文档经 `Loader` → `ChunkingStrategy` 切片 → `Embedder` 向量化 → `VectorStore.upsert`。
2. `RAGPipeline.query(q)`：`Embedder.embed(q)` → `VectorStore.query(top_k)` → 组装提示 → `LLMProvider.chat`。

### Agent 链路（ReAct）
1. 系统提示列出可用 `Tool`。
2. 循环：LLM 输出 `Thought/Action/Action Input` → 解析 → 调用 `Tool.run` → `Observation` 回灌 → 直到 `Final Answer`。
3. 步数受 `agent_max_steps` 限制。

## 5. 可复现与验证

- 锁版本：`uv.lock`（由 `uv lock` 生成），`pyproject.toml` 声明主依赖与可选依赖组 `[llm-vllm]` `[llm-llamacpp]` `[embeddings]` `[vector]` `[agent]`。
- 测试：全部基于 Mock 适配器，`pytest` 在干净环境离线跑绿（`tests/`）。
- 持续集成：GitHub Actions 在 push/PR 时执行 `uv sync` + `ruff` + `pytest`。

## 6. 扩展指南

新增一个 LLM 提供方（例如自研模型）：

```python
from stellarai.core.config import StellarConfig
from stellarai.core.registry import provider
from stellarai.models.llm import LLMProvider

@provider("llm", "my-model")
def _make_my_llm(config: StellarConfig) -> LLMProvider:
    return MyLLM(config)

class MyLLM(LLMProvider):
    name = "my-model"
    def complete(self, prompt, **kw): ...
    def chat(self, messages, **kw): ...
```

随后设置 `STELLAR_LLM_PROVIDER=my-model` 即可，无需改动任何业务代码。工具扩展同理（`@provider("tool", "name")`）。
