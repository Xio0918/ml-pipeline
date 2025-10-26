.PHONY: install train test lint

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m src.train --data-path data/winequality-red.csv --mlruns mlruns --output-dir models

test:
	pytest -q
