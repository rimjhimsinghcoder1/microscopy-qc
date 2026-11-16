.PHONY: install test lint format data train report all

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

report:
	mqc report --config configs/baseline.yaml

# Everything from raw download to report, in one command.
all: data train report
