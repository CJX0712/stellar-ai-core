# StellarAI 部署指南

## 1. 环境要求

- Python >= 3.10（推荐 3.13）
- `uv` >= 0.12（用于锁版本与隔离安装）：`pip install uv`

## 2. 干净环境一键复现

```bash
# 1) 创建虚拟环境并安装（核心依赖，含 NumPy / FastAPI / Typer）
uv venv
uv pip install -e .

# 2) 生成/校验锁版本（提交 uv.lock 以保证可复现）
uv lock

# 3) 安装可选 SOTA 组件（按需）
uv pip install -e ".[llm-llamacpp]"   # 本地 GGUF 推理
uv pip install -e ".[embeddings]"      # sentence-transformers
uv pip install -e ".[vector]"          # Chroma
# 或一次性全装
uv pip install -e ".[all]"
```

> 核心依赖不含任何重模型库；Mock 模式下 `pytest` 可完全离线跑绿。

## 3. 本地运行

```bash
stellarai demo            # 离线端到端演示
stellarai serve           # 启动 FastAPI（默认 0.0.0.0:8000）
stellarai chat "你好"
```

API 启动后访问 `http://localhost:8000/docs` 查看 OpenAPI 交互文档。

## 4. Docker 部署

```bash
# 构建镜像
docker build -t stellarai:0.1.0 .

# 运行（Mock 模式，零外部依赖）
docker run -p 8000:8000 stellarai:0.1.0
```

`docker-compose.yml` 提供 API 服务；取消注释 `vllm` 段可挂载本地推理侧车：

```bash
docker compose up --build
```

## 5. 接入真实模型（生产）

编辑 `.env` 或导出环境变量：

```bash
# 使用 vLLM / Ollama 等 OpenAI 兼容端点
export STELLAR_LLM_PROVIDER=vllm
export STELLAR_LLM_BASE_URL=http://localhost:8000/v1
export STELLAR_LLM_MODEL=Qwen2.5-0.5B-Instruct

# 嵌入
export STELLAR_EMBEDDER_PROVIDER=openai
export STELLAR_EMBEDDER_BASE_URL=http://localhost:8000/v1
export STELLAR_EMBEDDER_MODEL=bge-small-zh

# 向量库切换为 Chroma
export STELLAR_VECTOR_STORE=chroma
export STELLAR_VECTOR_PERSIST_PATH=./.chroma
```

配置项完整列表见 [USAGE.md](./USAGE.md)。

## 6. 持续集成

`.github/workflows/ci.yml` 在 `push` / `pull_request` 时：

1. 安装 `uv` 与 Python 3.13
2. `uv sync`（含 dev 依赖组）
3. `ruff check` 静态检查
4. `pytest` 运行全量测试

## 7. 发布到 GitHub

仓库已发布至 `CJX0712/stellar-ai-core`，作者署名「晨星」。本地提交与推送：

```bash
git init
git add -A
git commit -m "StellarAI: 模块化可编排 AI 系统 v0.1.0"
git remote add origin git@github.com:CJX0712/stellar-ai-core.git
git push -u origin main
```
