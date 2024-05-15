from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import base64
from io import BytesIO
from typing import Dict, Optional, List
from google.cloud import error_reporting
from models.prompt2image import prompt2imageURL
from models.style_transfer_cnn import neural_style_transfer
from models.utils.image_utils import save_image, load_image
import requests
from PIL import Image

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

# Paths to different resolutions of the content image
CONTENT_FILES = {
    'small': os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_small.jpg'),
    'medium': os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_medium.jpg'),
    'large': os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_large.jpg')
}

# Heights corresponding to different resolutions
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

        # Determine the height based on the resolution
        height = HEIGHTS.get(request.resolution)
        if not height:
            raise HTTPException(status_code=400, detail="Invalid resolution specified")

        # Download style image if it's a URL
        if style_image_url.startswith('http://') or style_image_url.startswith('https://'):
            response = requests.get(style_image_url)
            style_image_path = os.path.join(BASE_DIR, 'data', 'raw', 'images', 'style', 'temp_style.jpg')
            with open(style_image_path, 'wb') as f:
                f.write(response.content)
        else:
            style_image_path = style_image_url

        # Prepare parameters for the neural_style_transfer function
        style_transfer_params = {
            "content_img_name": os.path.basename(content_img_path),
            "style_img_name": os.path.basename(style_image_path),
            "content_images_dir": os.path.dirname(content_img_path),
            "style_images_dir": os.path.dirname(style_image_path),
            "output_img_dir": os.path.join(BASE_DIR, 'data', 'processed'),
            "height": height,
            "content_weight": request.content_weight if request.content_weight is not None else 1e5,
            "style_weight": request.style_weight if request.style_weight is not None else 200000,
            "tv_weight": 1e0,
            "optimizer": request.optimizer_type if request.optimizer_type is not None else 'adam',
            "init_method": 'content',
            "saving_freq": -1,
            "model": 'vgg19',
            "img_format": (4, '.jpg'),
            "num_of_iterations": 800 # Initial Recommendation "lbfgs" 1000, "adam": 3000
        }

        # Perform style transfer and get the final image object
        results_path = neural_style_transfer(style_transfer_params)
        print("Style transfer completed")

        # Read the generated image from the output path
        output_image_path = os.path.join(results_path, 'final.jpg')
        final_image = Image.open(output_image_path)

        # Convert the image object to a Base64 string
        buffered = BytesIO()
        final_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print("Image converted to base64")

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
    """
    Health check endpoint to confirm the API is up and running.
    """
    return {"status": "success", "message": "API is up and running!"}
