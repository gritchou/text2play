include .env
export

.PHONY: setup install run clean preprocess-dataset docker-build docker-run docker-stop docker-push deploy-gcr undeploy-gcr delete-gar docker-clean start-api

# Define default target, executed when no target is specified
all: install

# Setup virtual environment
setup:
    python3 -m venv text2play
    @echo "Virtual environment created."

# Install dependencies
install:
    @echo "Installing dependencies..."
    ./text2play/bin/pip install -r requirements.txt

# Run the application
run:
    @echo "Running the application..."
    ./text2play/bin/python main.py

# Clean up the project: remove Python cache files, virtual environment, etc.
clean:
    @echo "Cleaning up..."
    find . -type f -name '*.pyc' -delete
    find . -type d -name '__pycache__' -delete
    find $(PROCESSED_DATA_PATH) -type f -name '*.csv' -delete
    rm -rf venv
    @echo "Cleaned."

# Preprocess the dataset
preprocess-dataset:
    @echo "Preprocessing dataset..."
    python text2play/data/preprocessing.py $(RAW_DATA_PATH) $(DATASET_RAW_FILE_NAME) $(PROCESSED_DATA_PATH) $(DATASET_PROCESSED_FILE_NAME)

# Clean dataset
clean-dataset:
    @echo "Cleaning dataset..."
    find $(PROCESSED_DATA_PATH) -type f -name '*.csv' -delete
    @echo "Dataset cleaned."

# Docker build command
docker-build: preprocess-dataset
    @echo "Building Docker image..."
    docker build -t europe-west1-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY_NAME)/text2play-api:$(TAG) .

# Docker run command
docker-run:
    @echo "Running Docker container..."
    docker run --name text2play-api-container -p 80:80 text2play-api

# Docker stop command
docker-stop:
    @echo "Stopping Docker container..."
    -docker stop text2play-api-container
    -docker rm text2play-api-container
    @echo "Container stopped and removed."

# Docker push command to Google Artifact Registry
docker-push:
    @echo "Pushing Docker image to Google Artifact Registry..."
    docker push europe-west1-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY_NAME)/text2play-api:$(TAG)

# Deploy to Google Cloud Run
deploy-gcr:
    @echo "Deploying to Google Cloud Run..."
    gcloud run deploy text2play-api --image europe-west1-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY_NAME)/text2play-api:$(TAG) \
    --platform managed --region europe-west1 --allow-unauthenticated

# Undeploy from Google Cloud Run
undeploy-gcr:
    @echo "Undeploying from Google Cloud Run..."
    gcloud run services delete text2play-api --platform managed --region europe-west1

# Delete image from Google Artifact Registry
delete-gar:
    @echo "Deleting image from Google Artifact Registry..."
    gcloud artifacts docker images delete europe-west1-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY_NAME)/text2play-api:$(TAG)

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
    uvicorn text2play.api.api:app --reload
