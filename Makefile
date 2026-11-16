.PHONY: install test lint format data train

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

train:
	mqc train --config configs/baseline.yaml
