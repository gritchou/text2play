from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import base64
from io import BytesIO
from typing import Dict, Optional
from google.cloud import error_reporting
from ..models.prompt2image import prompt2imageURL
from ..models.style_transfer_cnn import style_transfer

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
    "small": os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_small.jpg'),
    "medium": os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_medium.jpg'),
    "large": os.path.join(BASE_DIR, 'data', 'raw', 'images', 'content', 'background_large.jpg'),
}

class TextPrompt(BaseModel):
    prompt: str
    num_steps: Optional[int] = 300
    content_weight: Optional[float] = 1e5
    style_weight: Optional[float] = 1e10

@app.post("/getImage/")
async def get_image(
    prompt: TextPrompt,
    resolution: str = Query("medium", enum=["small", "medium", "large"])
) -> Dict[str, str]:
    """
    Accepts text input, finds a matching image from the dataset based on text similarity,
    and applies style transfer to generate a stylized image.
    """
    try:
        print("Received prompt:", prompt.prompt)
        df = pd.read_csv(DATA_FILE)
        print("Data file loaded")
        style_image_url = prompt2imageURL(prompt.prompt, df)
        print("Style image URL:", style_image_url)

        content_img_path = CONTENT_FILES.get(resolution, CONTENT_FILES["medium"])
        print("Content image path:", content_img_path)

        style_img_path = style_image_url
        print("Style image path:", style_img_path)

        # Perform style transfer and get the final image object
        final_image = style_transfer(
            content_img_path, style_img_path,
            num_steps=prompt.num_steps,
            content_weight=prompt.content_weight,
            style_weight=prompt.style_weight
        )
        print("Style transfer completed")

        # Convert the image object to a Base64 string
        buffered = BytesIO()
        final_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        print("Image converted to base64")

        return {
            "prompt": prompt.prompt,
            "content_image_url": content_img_path,
            "style_image_url": style_img_path,
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
