from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd

import os
import base64

from io import BytesIO
from typing import Dict

from ..models.prompt2image import prompt2imageURL
from ..models.style_transfer_cnn import style_transfer

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. Change this to allow only specific origins in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'paintings_dataset.csv')
CONTENT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background.jpg')

class TextPrompt(BaseModel):
    prompt: str

@app.post("/getImage/")
async def get_image(prompt: TextPrompt) -> Dict[str, str]:
    """
    Accepts text input, finds a matching image from the dataset based on text similarity,
    and applies style transfer to generate a stylized image.
    """
    try:
        df = pd.read_csv(DATA_FILE)
        style_image_url = prompt2imageURL(prompt.prompt, df)
        content_img_path = CONTENT_FILE
        style_img_path = style_image_url

        # Perform style transfer and get the final image object
        final_image = style_transfer(content_img_path, style_img_path, num_steps=300, content_weight=1e5, style_weight=1e10)

        # Convert the image object to a Base64 string
        buffered = BytesIO()
        final_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "prompt": prompt.prompt,
            "content_image_url": content_img_path,
            "style_image_url": style_image_url,
            "stylized_image": img_str
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    """
    Health check endpoint to confirm the API is up and running.
    """
    return {"status": "success", "message": "API is up and running!"}
