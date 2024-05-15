from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import base64
from io import BytesIO
from typing import Dict, Optional
from google.cloud import error_reporting
from models.prompt2image import prompt2imageURL
from models.style_transfer_cnn import style_transfer

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

class StyleTransferRequest(BaseModel):
    prompt: str
    resolution: str
    content_weight: Optional[float] = None
    style_weight: Optional[float] = None
    num_steps: Optional[int] = None

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

        # Default values if not provided
        content_weight = request.content_weight if request.content_weight is not None else 1
        style_weight = request.style_weight if request.style_weight is not None else 200000
        num_steps = request.num_steps if request.num_steps is not None else 200

        # Perform style transfer and get the final image object
        final_image = style_transfer(
            content_img_path,
            style_image_url,
            num_steps=num_steps,
            content_weight=content_weight,
            style_weight=style_weight
        )
        print("Style transfer completed")

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
