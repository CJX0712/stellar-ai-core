# StellarAI 使用指南

## 1. 命令行（CLI）

安装后提供 `stellarai` 命令：

| 命令 | 说明 | 示例 |
|------|------|------|
| `stellarai version` | 打印版本与作者 | `stellarai version` |
| `stellarai chat` | 调用 LLM 补全 | `stellarai chat "用一句话介绍 RAG"` |
| `stellarai embed` | 文本向量化 | `stellarai embed "人工智能"` |
| `stellarai ingest` | 摄取文档到知识库 | `stellarai ingest --text "StellarAI 由晨星打造"` |
| `stellarai query` | 知识库问答 | `stellarai query "谁打造了 StellarAI？"` |
| `stellarai serve` | 启动 API 服务 | `stellarai serve --port 8000` |
| `stellarai demo` | 离线端到端演示 | `stellarai demo` |

## 2. REST API

启动 `stellarai serve` 后可用（OpenAPI 文档见 `/docs`）：

| 方法 | 路径 | 请求 | 响应 |
|------|------|------|------|
| GET | `/health` | — | `{status, config}` |
| POST | `/chat` | `{messages:[{role,content}]}` | `{text, model, provider}` |
| POST | `/embed` | `{texts:[...]}` | `{embeddings, dim}` |
| POST | `/ingest` | `{documents:[{content,source}]}` | `{ingested_chunks}` |
| POST | `/query` | `{question, top_k?}` | `{answer, sources, scores}` |

示例：

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"谁打造了 StellarAI？"}'
```

## 3. 配置项（`STELLAR_` 前缀环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `STELLAR_LLM_PROVIDER` | `mock` | `mock` / `openai` / `vllm` / `ollama` / `llamacpp` |
| `STELLAR_LLM_MODEL` | `mock-model` | 模型名 |
| `STELLAR_LLM_BASE_URL` | — | OpenAI 兼容端点（vllm/ollama）或 GGUF 路径（llamacpp） |
| `STELLAR_LLM_API_KEY` | — | API Key（如用 OpenAI） |
| `STELLAR_EMBEDDER_PROVIDER` | `mock` | `mock` / `openai` |
| `STELLAR_EMBEDDER_MODEL` | `mock-embed` | 嵌入模型 |
| `STELLAR_EMBEDDER_DIM` | `64` | 向量维度 |
| `STELLAR_VECTOR_STORE` | `numpy` | `numpy` / `chroma` |
| `STELLAR_VECTOR_PERSIST_PATH` | — | 持久化路径 |
| `STELLAR_CHUNK_SIZE` | `256` | 切片大小 |
| `STELLAR_CHUNK_OVERLAP` | `32` | 切片重叠 |
| `STELLAR_TOP_K` | `4` | 检索条数 |
| `STELLAR_AGENT_MAX_STEPS` | `6` | Agent 最大推理步数 |
| `STELLAR_HOST` / `STELLAR_PORT` | `0.0.0.0` / `8000` | 服务监听 |

## 4. Python 代码调用

```python
from stellarai.core.config import StellarConfig
from stellarai.rag.pipeline import RAGPipeline
from stellarai.models.providers.mock import MockLLM, MockEmbedder
from stellarai.memory.stores.numpy_store import NumPyCosineStore

cfg = StellarConfig(llm_provider="mock", embedder_provider="mock", embedder_dim=32)
store = NumPyCosineStore(cfg)
rag = RAGPipeline(cfg, llm=MockLLM(cfg), embedder=MockEmbedder(cfg), store=store)

rag.ingest(["StellarAI 由晨星打造，支持 RAG 与 Agent。"])
print(rag.answer("StellarAI 是谁打造的？")["answer"])
```

## 5. 扩展：新增工具

```python
from stellarai.core.config import StellarConfig
from stellarai.core.registry import provider
from stellarai.agents.tool import Tool

@provider("tool", "weather")
def _make_weather(config, store=None) -> Tool:
    return WeatherTool()

class WeatherTool(Tool):
    name = "weather"
    description = "查询天气"
    def run(self, action_input: str) -> str:
        return f"[weather] {action_input}: 晴 26C"
```

Agent 会在系统提示中自动发现该工具并支持调用。

## 6. 测试

```bash
pytest -q        # 全量（Mock 模式，离线）
pytest tests/test_rag.py::test_rag_answer_includes_context  # 单测
```
