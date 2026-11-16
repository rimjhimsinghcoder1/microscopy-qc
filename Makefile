.PHONY: install test lint format data

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

data:
	mqc prepare --config configs/baseline.yaml
