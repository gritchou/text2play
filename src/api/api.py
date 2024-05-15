from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pandas as pd

import os
import base64

from io import BytesIO
from typing import Dict

from ..models.prompt2image import prompt2imageURL
from ..models.style_transfer_cnn import style_transfer

import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'paintings_dataset.csv')
CONTENT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background3.png')

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
        content_title=content_img_path.replace('.png','')
        style_img_path = style_image_url
        # Perform style transfer and get the final image object
        final_image = style_transfer(content_img_path, style_img_path, num_steps=200,
                       style_weight=200000, content_weight=1)
        # Convert the image object to a Base64 string
        output_img = transforms.ToPILImage()(final_image.cpu().squeeze(0))
        output_img.save(content_title + f"_modified.jpg")
        return {
            "prompt": prompt.prompt,
            "content_image_url": content_img_path,
            "style_image_url": style_image_url,

        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    """
    Health check endpoint to confirm the API is up and running.
    """
    return {"status": "success", "message": "API is up and running!"}
