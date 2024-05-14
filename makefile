include .env
export

.PHONY: setup build install run clean preprocess-dataset docker-build docker-run docker-stop docker-push deploy-gcr undeploy-gcr delete-gar docker-clean start-api test-api test-api-local test-api-prompt deploy-gke undeploy-gke create-gke-cluster create-gpu-node-pool

# Define default target, executed when no target is specified
all: install

# Setup virtual environment
setup:
	python3 -m venv text2play
	@echo "Virtual environment created."

# Install package in editable mode
build:
	@echo "Building and installing the package in editable mode..."
	./text2play/bin/pip install -e .

# Install dependencies
install:
	@echo "Installing dependencies..."
	./text2play/bin/pip install --upgrade pip
	./text2play/bin/pip install -r requirements.txt

# Run the application
run:
	@echo "Running the application..."
	$(PYTHON) main.py

# Clean up the project: remove Python cache files, virtual environment, etc.
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find $(PROCESSED_DATA_PATH) -type f -name '*.csv' -delete
	rm -rf text2play
	@echo "Cleaned."

# Preprocess the dataset
preprocess-dataset:
	@echo "Preprocessing dataset..."
	$(PYTHON) src/data/preprocessing.py $(RAW_DATA_PATH) $(DATASET_RAW_FILE_NAME) $(PROCESSED_DATA_PATH) $(DATASET_PROCESSED_FILE_NAME)

# Clean dataset
clean-dataset:
	@echo "Cleaning dataset..."
	find $(PROCESSED_DATA_PATH) -type f -name '*.csv' -delete
	@echo "Dataset cleaned."

# Build the Docker image
docker-build: preprocess-dataset
	@echo "Building Docker image..."
	docker build -t $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_REPOSITORY_NAME)/text2play-api:$(DOCKER_IMAGE_TAG) .

# Run the Docker image locally
docker-run:
	@echo "Running Docker container..."
	docker run --name $(CONTAINER_NAME) -e PORT=8080 -p 8080:8080 $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_REPOSITORY_NAME)/text2play-api:$(DOCKER_IMAGE_TAG)

# Docker stop command
docker-stop:
	@echo "Stopping Docker container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "Container stopped and removed."

# Docker push command to Google Artifact Registry
docker-push:
	@echo "Pushing Docker image to Google Artifact Registry..."
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_REPOSITORY_NAME)/text2play-api:$(DOCKER_IMAGE_TAG)

# Delete image from Google Artifact Registry
delete-gar:
	@echo "Deleting image from Google Artifact Registry..."
	gcloud artifacts docker images delete $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_REPOSITORY_NAME)/text2play-api:$(DOCKER_IMAGE_TAG)

# Deploy to Google Cloud Run
deploy-gcr:
	@echo "Deploying to Google Cloud Run..."
	gcloud run deploy text2play-api --image $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_REPOSITORY_NAME)/text2play-api:$(DOCKER_IMAGE_TAG) \
	--platform managed --region $(GCP_REGION) --allow-unauthenticated --memory $(GCR_MEMORY)

# Undeploy from Google Cloud Run
undeploy-gcr:
	@echo "Undeploying from Google Cloud Run..."
	gcloud run services delete text2play-api --platform managed --region $(GCP_REGION)

# Docker clean command
docker-clean:
	@echo "Cleaning Docker..."
	docker container prune -f
	docker image prune -a -f
	docker volume prune -f
	docker network prune -f
	@echo "Docker cleaned."

# Start the API locally using Uvicorn
start-api:
	@echo "Starting the API server locally..."
	./text2play/bin/uvicorn src.api.api:app --reload --port 8080

# Test the deployed API
test-api:
	@echo "Testing API..."
	curl $(SERVICE_URL)/ping

# Test the local API
test-api-local:
	@echo "Testing local API..."
	curl http://127.0.0.1:8080/ping

# Test API with prompt and handle image response
test-api-prompt:
	@if [ "$(PROMPT)" = "" ]; then \
		echo "Usage: make test-api-prompt PROMPT='<prompt-text>'"; \
	else \
		./tests/test_api.sh "$(PROMPT)"; \
	fi

# Deploy to GKE
deploy-gke:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml

# Undeploy from GKE
undeploy-gke:
	kubectl delete -f k8s/service.yaml
	kubectl delete -f k8s/deployment.yaml

# Create GKE cluster
create-gke-cluster:
	@echo "Creating GKE cluster..."
	gcloud container clusters create $(GKE_CLUSTER_NAME) --zone $(GKE_ZONE) --machine-type n1-standard-2

# Create GPU Node Pool
create-gpu-node-pool:
	@echo "Creating GPU node pool..."
	gcloud container node-pools create gpu-pool --cluster $(GKE_CLUSTER_NAME) --zone $(GKE_ZONE) --accelerator type=nvidia-tesla-k80,count=1 --num-nodes 1 --min-nodes 0 --max-nodes 3 --machine-type n1-standard-2 --enable-autoscaling

# Get GKE Cluster Credentials
get-gke-credentials:
	@echo "Fetching GKE cluster credentials..."
	gcloud container clusters get-credentials $(GKE_CLUSTER_NAME) --zone $(GKE_ZONE)
