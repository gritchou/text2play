import torch
import requests
from io import BytesIO
from torchvision import transforms, models
from PIL import Image
from torch.optim import Adam

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

def get_features(image, model, layers=None):
    """ Extract features from the layers of the model. """
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """ Calculate the Gram matrix of a tensor. """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_transfer(content_img_path, style_img_path, output_path, num_steps=300, content_weight=1e5, style_weight=1e10):
    """ Perform style transfer from style_img to content_img. """
    # Check GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = load_image(content_img_path, device)
    style_img = load_image(style_img_path, device, scale=0.5)
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    vgg.to(device).eval()

    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = Adam([target], lr=0.003)

    for i in range(1, num_steps + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_grams:
            target_gram = gram_matrix(target_features[layer])
            layer_style_loss = torch.mean((target_gram - style_grams[layer])**2)
            style_loss += layer_style_loss / (target_features[layer].shape[1] * target_features[layer].shape[2] * target_features[layer].shape[3])

        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward(retain_graph=(i != num_steps))
        optimizer.step()

        if i % 50 == 0:
            print(f'Step {i}, Total loss: {total_loss.item()}')

    final_img = transforms.ToPILImage()(target.cpu().squeeze(0))
    final_img.save(output_path)
    return final_img
