PYTHON ?= python
export PYTHONPATH := src

install:
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	$(PYTHON) -m pytest -q

phase1-cora:
	$(PYTHON) scripts/run_phase1.py --dataset data/processed/cora.mat --model anemone --k 150 --runs 5
