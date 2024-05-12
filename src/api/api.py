from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

import pandas as pd
import os

from ..models.prompt2image import prompt2imageURL
from ..models.style_transfer_cnn import style_transfer

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'paintings_dataset.csv')
CONTENT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'astronaut.png')
STYLIZED_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'stylized_image.jpg')

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
        print(prompt.prompt)
        style_image_url = prompt2imageURL(prompt.prompt, df)
        print(style_image_url)
        content_img_path = CONTENT_FILE
        print(content_img_path)
        stylized_image_path = STYLIZED_FILE
        print(stylized_image_path)
        style_img_path = style_image_url

        # Call style transfer without device argument
        final_image = style_transfer(content_img_path, style_img_path, stylized_image_path, num_steps=300, content_weight=1e5, style_weight=1e10)

        # Assuming you might store or serve the final image via URL, modify as needed
        stylized_image_url = f"http://example.com/path_to_output/{os.path.basename(stylized_image_path)}"

        return {
            "prompt": prompt.prompt,
            "content_image_url": content_img_path,
            "style_image_url": style_image_url,
            "stylized_image_url": stylized_image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    """
    Health check endpoint to confirm the API is up and running.
    """
    return {"status": "success", "message": "API is up and running!"}
