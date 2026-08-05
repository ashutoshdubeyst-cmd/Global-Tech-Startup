# =========================================================
# Makefile for Data Science Project
# =========================================================

VENV        = venv
PYTHON      = ./$(VENV)/bin/python
SRC         = src
DATA_RAW    = data/raw
DATA_INTERIM= data/interim
DATA_PROC   = data/processed
CONFIG      = configs/model_config.yaml

# -----------------------------------------------------------
# Default target: run the full pipeline end-to-end
# -----------------------------------------------------------
.PHONY: all
all: data features train evaluate

# -----------------------------------------------------------
# Environment setup
# -----------------------------------------------------------
.PHONY: setup
setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install flake8

.PHONY: install
install:
	pip install -r requirements.txt

# -----------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------
.PHONY: data
data:
	$(PYTHON) $(SRC)/data/make_dataset.py \
		--input $(DATA_RAW) \
		--output $(DATA_PROC)

.PHONY: features
features: data
	$(PYTHON) $(SRC)/features/build_features.py \
		--input $(DATA_PROC) \
		--output $(DATA_PROC)/features.csv

# -----------------------------------------------------------
# Modeling
# -----------------------------------------------------------
.PHONY: train
train: features
	$(PYTHON) $(SRC)/models/train.py --config $(CONFIG)

.PHONY: predict
predict:
	$(PYTHON) $(SRC)/models/predict.py --config $(CONFIG)

.PHONY: evaluate
evaluate: train
	$(PYTHON) $(SRC)/models/evaluate.py --config $(CONFIG)

# -----------------------------------------------------------
# Testing & code quality
# -----------------------------------------------------------
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:
	$(PYTHON) -m flake8 $(SRC) tests/

# -----------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------
.PHONY: clean
clean:
	rm -rf $(DATA_INTERIM)/* $(DATA_PROC)/* models/*.pkl models/*.joblib
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: clean-env
clean-env:
	rm -rf $(VENV)

# -----------------------------------------------------------
# Jupyter
# -----------------------------------------------------------
.PHONY: notebook
notebook:
	jupyter notebook notebooks/

# -----------------------------------------------------------
# Help
# -----------------------------------------------------------
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup      - create venv and install dependencies"
	@echo "  make install    - install dependencies into current env"
	@echo "  make data       - build processed dataset from raw data"
	@echo "  make features   - run feature engineering"
	@echo "  make train      - train the model"
	@echo "  make predict    - run inference with trained model"
	@echo "  make evaluate   - evaluate model performance"
	@echo "  make test       - run unit tests"
	@echo "  make lint       - check code style with flake8"
	@echo "  make clean      - remove generated data/model files"
	@echo "  make clean-env  - remove virtual environment"
	@echo "  make notebook   - launch Jupyter notebook server"
	@echo "  make all        - run full pipeline (data -> features -> train -> evaluate)"