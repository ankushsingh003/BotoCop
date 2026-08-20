.PHONY: install index run-dev

install:
	uv sync

index:
	uv run python backend/scripts/build_index.py

run-dev:
	uv run python main.py
