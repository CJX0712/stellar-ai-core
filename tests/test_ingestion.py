from __future__ import annotations

from stellarai.core.types import Document
from stellarai.ingestion.chunker import RecursiveCharChunker
from stellarai.ingestion.loader import TxtLoader, load_document


def test_txt_loader_raw_text():
    doc = TxtLoader().load("这是直接传入的原始文本")
    assert doc.content == "这是直接传入的原始文本"
    assert doc.source == "<raw-text>"


def test_load_document_pdf_path_missing():
    import pytest

    from stellarai.core.errors import IngestionError

    with pytest.raises(IngestionError):
        load_document("./不存在的文件.pdf")


def test_chunker_respects_size_and_overlap():
    text = "段落一内容。" * 50 + "段落二内容。" * 50
    chunker = RecursiveCharChunker(chunk_size=64, chunk_overlap=8)
    chunks = chunker.chunk(Document(content=text, source="t"))
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 64 + 8 + 1
    # 切片应能拼接出原始文本的关键片段
    joined = " ".join(c.text for c in chunks)
    assert "段落一内容" in joined and "段落二内容" in joined
    # 元数据继承 doc 来源
    assert all(c.doc_id == "t" for c in chunks)
