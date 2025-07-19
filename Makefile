# Makefile for Cross-Sell Intelligence Platform
# Provides common development tasks

.PHONY: help install test lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "Multi-CRM Cross-Sell Intelligence Platform"
	@echo "========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make test         Run all tests"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-integration Run integration tests"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code with black"
	@echo "  make type-check   Run type checking with mypy"
	@echo "  make coverage     Run tests with coverage"
	@echo "  make clean        Clean up temporary files"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start services with docker-compose"
	@echo "  make docker-down  Stop docker-compose services"
	@echo "  make run          Run the application"
	@echo "  make run-api      Run the API server"
	@echo "  make docs         Generate API documentation"

# Python commands
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy

# Install dependencies
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	$(PYTEST) tests/integration/ -v -m integration

test-ml:
	$(PYTEST) tests/ml/ -v

coverage:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	$(FLAKE8) src/ tests/ --max-line-length=100 --exclude=venv,__pycache__
	$(PYTHON) -m pylint src/ --max-line-length=100 || true

format:
	$(BLACK) src/ tests/ --line-length=100

format-check:
	$(BLACK) src/ tests/ --line-length=100 --check

type-check:
	$(MYPY) src/ --ignore-missing-imports

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

# Docker commands
docker-build:
	docker build -t multi-crm-cross-sell:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Run commands
run:
	$(PYTHON) main.py --run-once

run-scheduled:
	$(PYTHON) main.py --schedule daily

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run src/visualization/dashboard.py

# Database
db-init:
	$(PYTHON) -c "from src.models.database import init_db; init_db()"

db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# Documentation
docs:
	$(PYTHON) -m pdoc --html --output-dir docs src

api-docs:
	@echo "API docs available at http://localhost:8000/api/docs"

# Development
dev-setup: install db-init
	@echo "Development environment ready!"

check-all: format-check lint type-check test
	@echo "All checks passed!"

# CI/CD helpers
ci-test:
	$(PYTEST) tests/ -v --junit-xml=test-results.xml

ci-lint:
	$(FLAKE8) src/ tests/ --max-line-length=100 --format=junit-xml --output-file=lint-results.xml

# Release
version:
	@$(PYTHON) -c "import src; print(src.__version__)"

release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major
