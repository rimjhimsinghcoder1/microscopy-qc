.PHONY: install test lint format run

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

run:
	mqc --config configs/baseline.yaml
