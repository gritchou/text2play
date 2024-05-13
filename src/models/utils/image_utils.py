import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
import torch

def load_image(img_path, device, size=512, scale=None):
    """ Load an image and prepare it for processing. """
    # Check if img_path is a URL
    if img_path.startswith('http://') or img_path.startswith('https://'):
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    if scale:
        size = int(scale * min(image.size))

    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
