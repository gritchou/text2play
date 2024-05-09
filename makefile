include .env
export

.PHONY: setup install test run clean docs preprocess-dataset

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
	find $(PROCESSED_DATA_PATH) -type f -name '*.csv' -delete  # Assuming CSV files in the processed data path
	rm -rf venv
	@echo "Cleaned."

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd docs && $(MAKE) html
	@echo "Documentation generated."

# Preprocess the dataset
preprocess-dataset:
	@echo "Preprocessing dataset..."
	python text2play/data/preprocessing.py $(RAW_DATA_PATH) $(DATASET_RAW_FILE_NAME) $(PROCESSED_DATA_PATH) $(DATASET_PROCESSED_FILE_NAME)

# Clean dataset
clean-dataset:
	@echo "Cleaning dataset..."
	find $(PROCESSED_DATA_PATH) -type f -name '*.csv' -delete  # Modify as needed for actual cleanup tasks
	@echo "Dataset cleaned."
