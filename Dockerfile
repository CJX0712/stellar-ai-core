FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN uv venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir .

ENV PATH=/opt/venv/bin:$PATH

EXPOSE 8000

CMD ["stellarai", "serve", "--host", "0.0.0.0", "--port", "8000"]
