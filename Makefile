.PHONY: help install install-dev clean test lint format setup demo train predict data validate
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Telecoms Customer Churn ML Project$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install project dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	uv pip install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	uv pip install -e ".[dev]"
	pre-commit install

setup: ## Run complete project setup
	@echo "$(BLUE)Running project setup...$(NC)"
	python setup.py

clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf logs/*.log

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=telecoms_churn_ml --cov-report=html --cov-report=term

lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	flake8 telecoms_churn_ml/ scripts/
	mypy telecoms_churn_ml/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black telecoms_churn_ml/ scripts/
	isort telecoms_churn_ml/ scripts/

demo: ## Run demo pipeline
	@echo "$(BLUE)Running demo pipeline...$(NC)"
	python scripts/demo.py

train: ## Train model with sample data
	@echo "$(BLUE)Training model...$(NC)"
	python scripts/train_model.py --generate-sample --verbose

train-custom: ## Train model with custom data (specify DATA_DIR)
	@echo "$(BLUE)Training model with custom data...$(NC)"
	python scripts/train_model.py --data-dir $(DATA_DIR) --verbose

predict: ## Make predictions (specify INPUT and OUTPUT files)
	@echo "$(BLUE)Making predictions...$(NC)"
	churn-predict --input $(INPUT) --output $(OUTPUT) --probability

data: ## Generate sample data
	@echo "$(BLUE)Generating sample data...$(NC)"
	churn-predict generate --output-dir data/raw --customers 2000

validate: ## Validate data quality
	@echo "$(BLUE)Validating data...$(NC)"
	churn-predict process --data-dir data/raw

info: ## Show model information
	@echo "$(BLUE)Model information:$(NC)"
	churn-predict info

notebook: ## Start Jupyter notebook
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook notebooks/

lab: ## Start Jupyter lab
	@echo "$(BLUE)Starting Jupyter lab...$(NC)"
	jupyter lab

check: ## Run all quality checks
	@echo "$(BLUE)Running all quality checks...$(NC)"
	$(MAKE) lint
	$(MAKE) test

build: ## Build package
	@echo "$(BLUE)Building package...$(NC)"
	python -m build

install-package: build ## Install built package
	@echo "$(BLUE)Installing built package...$(NC)"
	pip install dist/*.whl

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t telecoms-churn-ml .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm -v $(PWD)/data:/app/data telecoms-churn-ml

requirements: ## Update requirements.txt
	@echo "$(BLUE)Updating requirements.txt...$(NC)"
	uv pip freeze > requirements.txt

upgrade: ## Upgrade all dependencies
	@echo "$(BLUE)Upgrading dependencies...$(NC)"
	uv pip install --upgrade -e ".[dev]"

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "Documentation generation not implemented yet"

deploy: ## Deploy model (placeholder)
	@echo "$(BLUE)Deploying model...$(NC)"
	@echo "Deployment not implemented yet"

# Examples
example-train: ## Example: Train model with sample data
	@echo "$(GREEN)Example: Training model with sample data$(NC)"
	$(MAKE) train

example-predict: ## Example: Make predictions
	@echo "$(GREEN)Example: Making predictions$(NC)"
	@echo "First, ensure you have a trained model and input data"
	@echo "Then run: make predict INPUT=data/new_customers.csv OUTPUT=predictions.csv"

example-pipeline: ## Example: Run complete pipeline
	@echo "$(GREEN)Example: Complete pipeline$(NC)"
	$(MAKE) data
	$(MAKE) train
	@echo "$(GREEN)Pipeline complete! Check models/model.pkl$(NC)"
