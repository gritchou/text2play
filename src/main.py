import argparse
import torch

from src.models.style_transfer_cnn import style_transfer

default_content_image = 'src/data/raw/images/content/astronaut.png'
default_style_image = 'src/data/raw/images/style/Van_Gogh_Starry_Night.jpg'
default_output_image = 'src/data/raw/images/stylized/astronaut_stylized.jpg'

def main():
    parser = argparse.ArgumentParser(description='Perform style transfer between two images.')
    parser.add_argument('--content_image', type=str, default=default_content_image, help='Path to the content image')
    parser.add_argument('--style_image', type=str, default=default_style_image, help='Path to the style image')
    parser.add_argument('--output_image', type=str, default=default_output_image, help='Path for saving the stylized output image')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_img = style_transfer(
        args.content_image,
        args.style_image,
        args.output_image,
        device
    )
    final_img.show()

if __name__ == "__main__":
    main()
