"""StellarAI 性能基准与正确性验证。

设计原则（零重写 / 集大成）：
- 数学底座直接复用 BLAS 加速的 NumPy（世界级线性代数），不重造；
- 检索正确性以「独立实现的余弦近邻参考解」交叉验证，给出可复现不变量；
- 全部基于 Mock 适配器，零 GPU / 零网络，任意干净环境 `uv run python benchmarks/run_bench.py` 即复现。

输出：人类可读表格 + `benchmarks/results.json`（含环境指纹，便于跨机对比）。
"""

from __future__ import annotations

import json
import platform
import sys
import time
from typing import Any

import numpy as np

from stellarai.agents.agent import Agent
from stellarai.core.config import StellarConfig
from stellarai.core.types import Chunk, Document
from stellarai.ingestion.chunker import RecursiveCharChunker
from stellarai.memory.stores.numpy_store import NumPyCosineStore
from stellarai.models.providers.mock import MockEmbedder, MockLLM, ScriptedLLM
from stellarai.rag.pipeline import RAGPipeline
from stellarai.tools.builtins import CalculatorTool


def _timeit(fn, reps: int) -> float:
    """返回单次平均耗时（秒）。"""
    fn()  # warmup（触发惰性构建 / JIT 友好）
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def bench_embedding() -> list[dict]:
    rows = []
    n_texts = 2000
    texts = [f"文档编号 {i}：StellarAI 由晨星打造的模块化 AI 系统。" for i in range(n_texts)]
    for dim in (64, 128, 256, 768):
        cfg = StellarConfig(embedder_provider="mock", embedder_dim=dim)
        emb = MockEmbedder(cfg)
        # 批处理吞吐
        dt = _timeit(lambda emb=emb: emb.embed(texts), reps=5)
        # 单条延迟
        dl = _timeit(lambda emb=emb: emb.embed_one(texts[0]), reps=200)
        rows.append({
            "dim": dim,
            "batch_texts": n_texts,
            "batch_throughput_docs_s": round(n_texts / dt, 1),
            "single_latency_ms": round(dl * 1000, 3),
        })
    return rows


def bench_vector_store() -> list[dict]:
    rows = []
    dim = 64
    top_k = 5
    rng = np.random.default_rng(20260921)
    for n in (1_000, 5_000, 20_000, 50_000):
        cfg = StellarConfig(embedder_dim=dim, vector_persist_path=None)
        store = NumPyCosineStore(cfg)
        # 预生成归一化向量
        vecs = rng.standard_normal((n, dim))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        docs = [Document(content=f"chunk-{i}", source="bench") for i in range(n)]
        # upsert 吞吐
        t0 = time.perf_counter()
        for i in range(n):
            store.upsert(docs[i], vecs[i].tolist())
        upsert_dt = time.perf_counter() - t0
        # 查询延迟（store 内部惰性构建矩阵，首查后缓存）
        q = rng.standard_normal(dim)
        q /= np.linalg.norm(q) or 1.0
        dt_query = _timeit(lambda store=store, q=q, top_k=top_k: store.query(q.tolist(), top_k), reps=200)
        rows.append({
            "vectors": n,
            "dim": dim,
            "top_k": top_k,
            "upsert_throughput_vec_s": round(n / upsert_dt, 1),
            "query_latency_ms": round(dt_query * 1000, 3),
            "query_mode": "brute-force cosine (O(N·d))",
        })
    return rows


def bench_retrieval_parity() -> dict:
    """正确性不变量：自实现检索 vs 独立参考解必须完全一致（top_k 集合与排序）。"""
    dim, n, top_k = 64, 2000, 5
    rng = np.random.default_rng(7)
    cfg = StellarConfig(embedder_dim=dim)
    store = NumPyCosineStore(cfg)
    texts = [f"text-{i}" for i in range(n)]
    arr = rng.standard_normal((n, dim))
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    for i, t in enumerate(texts):
        store.upsert(Chunk(text=t, doc_id="b", chunk_index=i), arr[i].tolist())
    q = rng.standard_normal(dim)
    q /= np.linalg.norm(q) or 1.0
    got = store.query(q.tolist(), top_k)
    # 独立参考解
    sims = arr @ q
    ref_idx = np.argsort(-sims)[:top_k]
    ref_set = {texts[i] for i in ref_idx}
    got_set = {r.chunk.text for r in got}
    got_sorted = [r.chunk.text for r in got]
    ref_sorted = [texts[i] for i in ref_idx]
    return {
        "top_k": top_k,
        "match_set": ref_set == got_set,
        "match_order": got_sorted == ref_sorted,
        "self_cosine_equals_1": abs(float(np.dot(q, q)) - 1.0) < 1e-12,
    }


def bench_ingestion() -> dict:
    para = "StellarAI 是晨星打造的模块化 AI 系统，支持 RAG 与 Agent 编排，复用业界领先开源成果。" * 8
    n_docs = 200
    docs = [Document(content=para, source="bench") for _ in range(n_docs)]
    chunker = RecursiveCharChunker(chunk_size=256, chunk_overlap=32)
    t0 = time.perf_counter()
    total_chunks = 0
    for d in docs:
        total_chunks += sum(1 for _ in chunker.chunk(d))
    dt = time.perf_counter() - t0
    return {
        "n_docs": n_docs,
        "total_chunks": total_chunks,
        "chars_per_doc": len(para),
        "throughput_docs_s": round(n_docs / dt, 1),
        "avg_chunks_per_doc": round(total_chunks / n_docs, 1),
    }


def bench_rag() -> dict:
    dim = 64
    n_docs = 500
    cfg = StellarConfig(embedder_provider="mock", embedder_dim=dim)
    store = NumPyCosineStore(cfg)
    rag = RAGPipeline(cfg, llm=MockLLM(cfg), embedder=MockEmbedder(cfg), store=store)
    corpus = [f"StellarAI 知识点 {i}：由晨星打造，模块 {i} 负责不同能力。" for i in range(n_docs)]
    t0 = time.perf_counter()
    rag.ingest(corpus, source="kb")
    ingest_dt = time.perf_counter() - t0
    n_queries = 100
    q_lat = _timeit(lambda: rag.answer("StellarAI 由谁打造？"), reps=n_queries)
    return {
        "dim": dim,
        "ingested_docs": n_docs,
        "ingest_throughput_docs_s": round(n_docs / ingest_dt, 1),
        "query_latency_ms": round(q_lat * 1000, 3),
        "note": "Mock LLM 为回显，query 延迟即 检索+拼接 延迟上界",
    }


def bench_agent() -> dict:
    cfg = StellarConfig(agent_max_steps=6)
    store = NumPyCosineStore(cfg)
    script = ["Thought: 计算\nAction: calculator\nAction Input: 12*8", "Final Answer: 96"]
    agent = Agent(cfg, llm=ScriptedLLM(script=script), tools={"calculator": CalculatorTool()}, store=store)
    task = "请帮我计算 12 乘以 8"
    reps = 1000
    dt = _timeit(lambda: agent.run(task), reps=reps)
    return {
        "steps": 2,
        "reps": reps,
        "loop_latency_ms": round(dt * 1000, 3),
        "loops_per_s": round(1 / dt, 1),
        "result_contains_96": "96" in agent.run(task),
    }


def run_all() -> dict[str, Any]:
    return {
        "embedding": bench_embedding(),
        "vector_store": bench_vector_store(),
        "retrieval_parity": bench_retrieval_parity(),
        "ingestion": bench_ingestion(),
        "rag": bench_rag(),
        "agent": bench_agent(),
    }


def env_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "n/a",
        "cpu_count": platform.processor() or "n/a",
    }


def main(quiet: bool = False, json_path: str | None = None) -> int:
    results = {"env": env_fingerprint(), "benchmarks": run_all()}
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    if quiet:
        # CI 模式：仅做不变量校验，任一失败即非零退出
        parity = results["benchmarks"]["retrieval_parity"]
        ok = parity["match_set"] and parity["match_order"] and parity["self_cosine_equals_1"]
        return 0 if ok else 1
    # 人类可读表格
    b = results["benchmarks"]
    print("=" * 72)
    print("StellarAI 基准测试（Mock 适配器 · 零 GPU / 零网络 · 全可复现）")
    print(f"环境: Python {results['env']['python']} · NumPy {results['env']['numpy']} · {results['env']['platform']}")
    print("=" * 72)
    print("\n[嵌入吞吐] MockEmbedder（文本哈希确定性向量）")
    print(f"  {'dim':>6} {'批文本':>8} {'批吞吐(docs/s)':>16} {'单条延迟(ms)':>14}")
    for r in b["embedding"]:
        print(f"  {r['dim']:>6} {r['batch_texts']:>8} {r['batch_throughput_docs_s']:>16} {r['single_latency_ms']:>14}")
    print("\n[向量检索] NumPyCosineStore（暴力余弦，O(N·d)）")
    print(f"  {'向量数':>8} {'dim':>5} {'top_k':>6} {'upsert(vec/s)':>15} {'查询延迟(ms)':>13}")
    for r in b["vector_store"]:
        print(f"  {r['vectors']:>8} {r['dim']:>5} {r['top_k']:>6} {r['upsert_throughput_vec_s']:>15} {r['query_latency_ms']:>13}")
    p = b["retrieval_parity"]
    print("\n[正确性不变量] 检索 vs 独立参考解")
    print(f"  top_k 集合一致: {p['match_set']} · 顺序一致: {p['match_order']} · 自相似=1.0: {p['self_cosine_equals_1']}")
    invariants_ok = p["match_set"] and p["match_order"] and p["self_cosine_equals_1"]
    print("\n[摄取切片] RecursiveCharChunker")
    print(f"  {b['ingestion']['n_docs']} 文档 → {b['ingestion']['total_chunks']} 切片 · {b['ingestion']['throughput_docs_s']} docs/s")
    print("\n[RAG 端到端] 摄取+检索+生成（Mock）")
    print(f"  {b['rag']['ingested_docs']} 文档 → 摄取 {b['rag']['ingest_throughput_docs_s']} docs/s · 查询 {b['rag']['query_latency_ms']} ms")
    print("\n[Agent 循环] ReAct + 计算器工具")
    print(f"  {b['agent']['steps']} 步/轮 · {b['agent']['loop_latency_ms']} ms/轮 · {b['agent']['loops_per_s']} loops/s")
    print("=" * 72)
    if not invariants_ok:
        print("[FAIL] 正确性不变量未通过，基准门禁失败。")
        return 1
    print("[OK] 正确性不变量全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
