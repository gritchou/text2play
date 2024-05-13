#!/bin/bash

# Check if a prompt is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 '<prompt>'"
    exit 1
fi

# API Endpoint URL
URL="http://127.0.0.1:8080/getImage/"

# Payload
DATA="{\"prompt\": \"$1\"}"

# Send the POST request and save the response
RESPONSE=$(curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$DATA")

# Print values
echo "Prompt: $1"
content_image_path=$(echo $RESPONSE | jq -r '.content_image_url')
style_image_path=$(echo $RESPONSE | jq -r '.style_image_url')
echo "Content Image Path: $content_image_path"
echo "Style Image Path: $style_image_path"

# Extract the base64 encoded image from the response
IMAGE_BASE64=$(echo $RESPONSE | jq -r '.stylized_image')

# Check if the image data is not empty
if [ -n "$IMAGE_BASE64" ]; then
  # Decode the image and save to a file
  echo $IMAGE_BASE64 | base64 --decode > src/data/processed/stylized_image.jpeg
  echo "Stylized image saved as src/data/processed/stylized_image.jpeg"
else
  echo "No image data received."
fi
