import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from models.utils.image_utils import load_image
from PIL import Image

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = nn.MSELoss()

    def forward(self, input):
        self.loss_value = self.loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()
        self.loss = nn.MSELoss()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss_value = self.loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers, style_layers):
    normalization = Normalization(normalization_mean, normalization_std).to(style_img.device)

    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i+1]

    return model, style_losses, content_losses

def style_transfer(content_img_path, style_img_path, num_steps=200, content_weight=1, style_weight=200000,
                   content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                   optimizer_type='Adam', learning_rate=0.003, output_path=None):
    """ Perform style transfer from style_img to content_img, returning the result as a PIL Image. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the content image and get its dimensions
    content_img = load_image(content_img_path, device)
    _, _, content_height, content_width = content_img.size()

    # Print the dimensions for debugging
    print(f"Content Image - Width: {content_width}, Height: {content_height}")

    # Load the style image and resize it to match the content image dimensions
    style_img = load_image(style_img_path, device, size=(content_height, content_width))

    # Print the dimensions for debugging
    print(f"Style Image - Width: {style_img.size(3)}, Height: {style_img.size(2)}")

    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img, content_layers, style_layers
    )

    # Initialize the target image with the content image dimensions
    input_img = content_img.clone().requires_grad_(True)

    # Select optimizer
    if optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS([input_img.requires_grad_()], lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam([input_img.requires_grad_()], lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type. Use 'LBFGS' or 'Adam'.")

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss_value
            for cl in content_losses:
                content_score += cl.loss_value

            loss = style_weight * style_score + content_weight * content_score
            if not torch.isnan(loss):
                loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run[0]}:")
                print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")
                print(f"Total Loss: {loss.item():4f}")

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    unloader = transforms.ToPILImage()
    image = input_img.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    # Save the image if an output path is provided
    if output_path:
        image.save(output_path)

    return image
