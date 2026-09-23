# StellarAI

<p align="center">
  <a href="https://github.com/CJX0712/stellar-ai-core/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/stellar-ai-core/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/stellar-ai-core/releases"><img src="https://img.shields.io/github/v/release/CJX0712/stellar-ai-core?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/stellar-ai-core/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/stellar-ai-core" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

> 模块化、可编排、接口驱动的端到端 AI 系统 —— 复用业界领先开源成果，避免从零自研。

**作者：晨星**

StellarAI 以 **单一职责 + 清晰接口 + 能力注册表** 为核心设计原则：

- 每个 AI 能力是一个独立模块，通过抽象基类（Protocol/ABC）定义契约，模块间只经接口通信，可独立验证；
- 业界领先 OSS（vLLM / llama-cpp-python / sentence-transformers / Chroma / LangGraph）作为适配器挂载，一行配置热插拔；
- 内置 Mock 适配器，无需 GPU/网络即可跑通全链路并保证可复现。

## 特性

- 模块化解耦：模型网关 / 摄取 / 向量存储 / RAG / Agent / 工具 / API / CLI 各自独立
- 接口驱动：所有模块以抽象基类定义契约，Mock 与真实实现可互换
- 能力注册表 + 依赖注入：按 `STELLAR_*` 环境变量实例化组件
- 零依赖可验证：Mock 模式全链路离线跑绿，复现无忧
- 一键复现：`uv` 锁版本 + `Dockerfile` + `Makefile` + GitHub Actions CI

## 架构

| 模块 | 职责 | 关键接口 |
|------|------|----------|
| Config & Registry | 配置与组件注册/注入 | `Registry` / `StellarConfig` |
| Model Gateway | 统一 LLM / Embedding 抽象 | `LLMProvider` / `Embedder` |
| Ingestion | 文档加载与切片 | `Loader` / `ChunkingStrategy` |
| Vector Store | 向量存取与检索 | `VectorStore` |
| RAG | 检索增强生成 | `RAGPipeline` |
| Agent | ReAct 推理编排 | `Agent` / `Tool` |
| API | FastAPI REST 服务 | `/chat` `/embed` `/ingest` `/query` |
| CLI | Typer 命令行 | `stellarai ...` |

详见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 快速开始

```bash
# 安装（uv，干净环境一键复现）
uv venv && uv pip install -e .

# 离线端到端演示（零依赖、零网络）
stellarai demo

# 启动 API 服务
stellarai serve

# 命令行
stellarai chat "你好"
stellarai ingest --text "StellarAI 由晨星打造"
stellarai query "谁打造了 StellarAI？"
```

## 复用业界领先 OSS

| 能力 | 复用方案 | 配置 |
|------|----------|------|
| 推理 | vLLM / Ollama / OpenAI 兼容 | `STELLAR_LLM_PROVIDER=openai|vllm|ollama` |
| 本地推理 | llama-cpp-python (GGUF) | `STELLAR_LLM_PROVIDER=llamacpp` |
| 嵌入 | sentence-transformers / OpenAI | `STELLAR_EMBEDDER_PROVIDER=openai`(+base_url) |
| 向量库 | Chroma | `STELLAR_VECTOR_STORE=chroma` |
| 编排 | LangGraph（可选） | 依赖组 `agent` |

## 测试

```bash
pytest
```

## 文档

- [ARCHITECTURE.md](./ARCHITECTURE.md) 系统架构
- [DEPLOYMENT.md](./DEPLOYMENT.md) 部署指南
- [USAGE.md](./USAGE.md) 使用指南

## 许可

MIT · 作者 晨星
