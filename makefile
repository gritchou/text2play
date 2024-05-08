# Simple Makefile for managing the development of a Python-based project

.PHONY: setup install test run clean docs

# Define default target, executed when no target is specified
all: install

# Setup virtual environment
setup:
	python3 -m venv venv
	@echo "Virtual environment created."

# Install dependencies
install:
	@echo "Installing dependencies..."
	./venv/bin/pip install -r requirements.txt

# Run tests
test:
	@echo "Running tests..."
	./venv/bin/pytest

# Run the application
run:
	@echo "Running the application..."
	./venv/bin/python game/scripts/main.py

# Clean up the project: remove Python cache files, virtual environment, etc.
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf venv
	@echo "Cleaned."

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd docs && $(MAKE) html
	@echo "Documentation generated."

# Add additional commands if needed
