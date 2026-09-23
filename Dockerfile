FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# 注: `uv venv` 默认不安装 pip, 因此用 `uv pip install --python` 直接装入该 venv。
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache-dir .

ENV PATH=/opt/venv/bin:$PATH

EXPOSE 8000

CMD ["stellarai", "serve", "--host", "0.0.0.0", "--port", "8000"]
