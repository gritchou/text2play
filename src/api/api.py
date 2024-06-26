from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import base64
from typing import Dict, Optional, List
from google.cloud import error_reporting
from models.prompt2image import prompt2imageURL
from models.style_transfer_cnn import neural_style_transfer
import requests
import numpy as np
import cv2 as cv

app = FastAPI()

# Initialize Google Cloud Error Reporting
client = error_reporting.Client()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://text2play.netlify.app"  # Deployed frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'paintings_dataset.csv')

CONTENT_FILES = {
    'small': os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_small.jpg'),
    'medium': os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_medium.jpg'),
    'large': os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_large.jpg')
}

HEIGHTS = {
    'small': 480,
    'medium': 720,
    'large': 1080
}

class StyleTransferRequest(BaseModel):
    prompt: str
    resolution: str
    content_weight: Optional[float] = None
    style_weight: Optional[float] = None
    num_steps: Optional[int] = None
    content_layers: Optional[List[str]] = None
    style_layers: Optional[List[str]] = None
    optimizer_type: Optional[str] = None
    learning_rate: Optional[float] = None

@app.post("/getImage/")
async def get_image(request: StyleTransferRequest) -> Dict[str, str]:
    """
    Accepts text input, finds a matching image from the dataset based on text similarity,
    and applies style transfer to generate a stylized image.
    """
    try:
        print("Received request:", request)
        df = pd.read_csv(DATA_FILE)
        print("Data file loaded")
        style_image_url = prompt2imageURL(request.prompt, df)
        print("Style image URL:", style_image_url)

        content_img_path = CONTENT_FILES.get(request.resolution)
        if not content_img_path:
            raise HTTPException(status_code=400, detail="Invalid resolution specified")

        print(f"Content image path: {content_img_path}")
        print(f"Style image path: {style_image_url}")

        height = HEIGHTS.get(request.resolution)
        if not height:
            raise HTTPException(status_code=400, detail="Invalid resolution specified")

        if style_image_url.startswith('http://') or style_image_url.startswith('https://'):
            response = requests.get(style_image_url)
            img_array = np.frombuffer(response.content, np.uint8)
            style_img = cv.imdecode(img_array, cv.IMREAD_COLOR)
        else:
            style_img = cv.imread(style_image_url)

        # Ensure the output directory exists
        output_img_dir = os.path.join(BASE_DIR, 'data', 'processed')
        os.makedirs(output_img_dir, exist_ok=True)

        # Prepare parameters for the neural_style_transfer function
        style_transfer_params = {
            "content_img_name": os.path.basename(content_img_path),
            "style_img": style_img,  # Pass the style image directly
            "content_images_dir": os.path.dirname(content_img_path),
            "height": height,
            "content_weight": request.content_weight if request.content_weight is not None else 1e5,
            "style_weight": request.style_weight if request.style_weight is not None else 200000,
            "tv_weight": 1e0,
            "optimizer": request.optimizer_type if request.optimizer_type is not None else 'adam',
            "init_method": 'content',
            "saving_freq": -1,
            "model": 'vgg19',
            "img_format": (4, '.jpg'),
            "num_of_iterations": 800,
            "output_img_dir": output_img_dir  # Ensure this is included
        }

        # Print the parameters, replacing the actual style image with a placeholder
        style_transfer_params_log = style_transfer_params.copy()
        style_transfer_params_log['style_img'] = 'Style image ready'
        print("Style transfer parameters prepared:", style_transfer_params_log)

        img_str = neural_style_transfer(style_transfer_params)
        print("Style transfer completed")

        return {
            "prompt": request.prompt,
            "content_image_url": content_img_path,
            "style_image_url": style_image_url,
            "stylized_image": img_str
        }

    except Exception as e:
        print(f"Error: {e}")
        client.report_exception()  # Report the error to Google Cloud Error Reporting
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    return {"status": "success", "message": "API is up and running!"}
