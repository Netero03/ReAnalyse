.PHONY: help install dev-install format lint type-check test test-unit test-integration test-cov clean run run-streamlit init-index

help:
	@echo "Financial Analyzer - Available Commands"
	@echo "========================================"
	@echo "install              : Install dependencies"
	@echo "dev-install          : Install dev dependencies"
	@echo "format               : Format code with black and isort"
	@echo "lint                 : Run flake8 linting"
	@echo "type-check           : Run mypy type checking"
	@echo "test                 : Run all tests"
	@echo "test-unit            : Run unit tests"
	@echo "test-integration     : Run integration tests"
	@echo "test-cov             : Run tests with coverage"
	@echo "clean                : Clean up cache and build files"
	@echo "run                  : Run Streamlit app"
	@echo "init-index           : Initialize Pinecone index"

install:
	pip install -r requirements.txt

dev-install: install
	pip install -r requirements-dev.txt

format:
	black src tests
	isort src tests

lint:
	flake8 src tests --max-line-length=100

type-check:
	mypy src --ignore-missing-imports

test: test-unit test-integration

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

run: run-streamlit

run-streamlit:
	streamlit run src/financial_analyzer/ui/streamlit_app.py

init-index:
	python -c "from src.financial_analyzer.vector_store.pinecone_client import initialize_index; initialize_index()"
