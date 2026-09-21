"""文档加载：支持本地文件与原始文本，PDF 经 pypdf 惰性导入。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from stellarai.core.errors import IngestionError
from stellarai.core.types import Document


class Loader(ABC):
    name: str = "base"

    @abstractmethod
    def load(self, source: str) -> Document:
        """source 可为文件路径或原始文本。"""


class TxtLoader(Loader):
    name = "txt"

    def load(self, source: str) -> Document:
        if os.path.isfile(source):
            try:
                with open(source, encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(source, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            return Document(content=text, source=source)
        return Document(content=source, source="<raw-text>")


class PdfLoader(Loader):
    name = "pdf"

    def load(self, source: str) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise IngestionError("PDF 加载需要 pypdf，请执行 pip install stellarai") from exc
        if not os.path.isfile(source):
            raise IngestionError(f"PDF 文件不存在: {source}")
        reader = PdfReader(source)
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages)
        meta = {str(k): str(v) for k, v in reader.metadata.items()} if reader.metadata else {}
        return Document(content=text, source=source, metadata=meta)


def load_document(source: str) -> Document:
    if source.lower().endswith(".pdf"):
        return PdfLoader().load(source)
    return TxtLoader().load(source)
