PY ?= uv

install:
	uv venv && uv pip install -e .

dev:
	uv pip install pytest pytest-asyncio ruff mypy

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

lock:
	uv lock

build:
	uv build

run:
	uv run stellarai serve

demo:
	uv run stellarai demo

docker-build:
	docker build -t stellarai:0.1.0 .

docker-up:
	docker compose up --build

clean:
	rm -rf .venv dist build .pytest_cache .mypy_cache .ruff_cache .chroma
