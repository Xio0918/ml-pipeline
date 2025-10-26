install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. python src/train.py

test:
	PYTHONPATH=. pytest -q
