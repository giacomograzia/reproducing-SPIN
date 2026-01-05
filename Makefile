.PHONY: help install test lint format clean train evaluate

help:
	@echo "FACT AI Group 23 - Reproducibility Project"
	@echo "==========================================="
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code with black"
	@echo "  make clean       - Clean temporary files"
	@echo "  make train       - Run training script"
	@echo "  make evaluate    - Run evaluation script"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest black flake8 mypy pytest-cov

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ train.py evaluate.py --max-line-length=100

format:
	black src/ tests/ train.py evaluate.py setup.py

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f .coverage

train:
	python train.py --config config.yaml

evaluate:
	@echo "Usage: make evaluate MODEL=path/to/model.pth"
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL not specified. Example: make evaluate MODEL=results/models/final_model.pth"; \
	else \
		python evaluate.py --model $(MODEL) --config config.yaml; \
	fi

notebook:
	jupyter notebook notebooks/

.DEFAULT_GOAL := help
