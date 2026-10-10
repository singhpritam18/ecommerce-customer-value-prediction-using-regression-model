#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME   = ecommerce-customer-value-prediction
PYTHON         = python
SRC_PKG        = src/ecommerce_customer_value_prediction

#################################################################################
# ENVIRONMENT                                                                   #
#################################################################################

## Create virtual environment (Python 3.10+)
.PHONY: create_environment
create_environment:
	$(PYTHON) -m venv .venv
	@echo ">>> Virtual environment created. Activate with:"
	@echo ">>>   Windows : .venv\\Scripts\\activate"
	@echo ">>>   Unix    : source .venv/bin/activate"

## Install project in editable mode (runs once after env activation)
.PHONY: install
install:
	pip install -e .

## Install project + dev/test extras
.PHONY: install-dev
install-dev:
	pip install -e .
	pip install pytest ruff

#################################################################################
# DATA PIPELINE                                                                 #
#################################################################################

## Run the cleaning pipeline: raw → data/interim/transactions_cleaned.csv
.PHONY: clean-data
clean-data:
	$(PYTHON) -m ecommerce_customer_value_prediction.data.clean_data

## Build customer-level features + target: → data/processed/customer_dataset.csv
## (called automatically by `make train`)
.PHONY: features
features:
	$(PYTHON) -c "from ecommerce_customer_value_prediction.features.build_features import build_customer_dataset, save_customer_dataset; import pandas as pd; df = pd.read_csv('data/interim/transactions_cleaned.csv', parse_dates=['InvoiceDate']); df['Customer ID'] = df['Customer ID'].astype('Int64'); save_customer_dataset(build_customer_dataset(df))"

#################################################################################
# MODELLING                                                                     #
#################################################################################

## Run full training pipeline: features → train/eval → models/ridge_pipeline.pkl
.PHONY: train
train:
	$(PYTHON) -m ecommerce_customer_value_prediction.pipelines.training_pipeline

## Show evaluation metrics from the last training run
.PHONY: metrics
metrics:
	$(PYTHON) -c "import json, pathlib; p=pathlib.Path('reports/metrics/evaluation_metrics.json'); print(json.dumps(json.loads(p.read_text()), indent=2)) if p.exists() else print('No metrics file found. Run: make train')"

#################################################################################
# QUALITY                                                                       #
#################################################################################

## Run all tests
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

## Lint with ruff
.PHONY: lint
lint:
	ruff format --check src/ tests/
	ruff check src/ tests/

## Auto-format with ruff
.PHONY: format
format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

#################################################################################
# HOUSEKEEPING                                                                  #
#################################################################################

## Delete compiled Python files and caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache"  -exec rm -rf {} +

#################################################################################
# SELF-DOCUMENTING COMMANDS                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
lines = '\n'.join([line for line in sys.stdin])
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines)
print('Available rules:\n')
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
