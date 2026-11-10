.PHONY: install lint format test eval drift report run docker-build docker-run

PY ?= python

install:
	$(PY) -m pip install -e ".[dev,mlops]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	$(PY) -m pytest -q

eval:
	redline eval --name dev --out reports/eval_dev.json --mlflow
	redline eval --gold data/gold/blind.jsonl --name blind --out reports/eval_blind.json --mlflow

drift:
	redline drift --out reports/drift_clean.json
	redline drift --inject-shift --out reports/drift_shifted.json

report:
	redline report --mlflow

run:
	uvicorn redline.api:app --reload --port 8000

docker-build:
	docker build -t contract-redline-copilot:local .

docker-run:
	docker run --rm -p 8000:8000 contract-redline-copilot:local
