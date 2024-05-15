# src/models/utils/image_utils.py
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
import torch

def load_image(img_path, device, size=None, scale=None):
    """ Load an image and prepare it for processing. """
    # Check if img_path is a URL
    if img_path.startswith('http://') or img_path.startswith('https://'):
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    print(f"Original image size: {image.size}")  # Debugging statement

    if scale:
        size = int(scale * min(image.size))
        print(f"Scaled size: {size}")  # Debugging statement

    if isinstance(size, tuple):
        print(f"Resizing to (height, width): {size}")  # Debugging statement
        loader = transforms.Compose([
            transforms.Resize(size),  # Use the size tuple directly (height, width)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif size is not None:
        print(f"Resizing to square size: {size}")  # Debugging statement
        loader = transforms.Compose([
            transforms.Resize(size),  # Resize to make the smaller edge of the image match the size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        print("No resizing applied")  # Debugging statement
        loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    image = loader(image).unsqueeze(0)
    print(f"Image size after loading: {image.size()}")  # Debugging statement

    # Print the actual dimensions after resizing
    _, _, height, width = image.size()
    print(f"Actual resized image dimensions: width={width}, height={height}")  # New debugging statement

    return image.to(device, torch.float)
