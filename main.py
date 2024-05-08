from text2play.models.style_transfer_cnn import style_transfer
import torch

content_image = 'text2play/data/raw/images/content/astronaut.png'
style_image = 'text2play/data/raw/images/style/Van_Gogh_Starry_Night.jpg'
output_image = 'text2play/data/raw/images/stylized/astronaut_stylized.jpg'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_img = style_transfer(
        content_image,
        style_image,
        output_image,
        device
    )
    final_img.show()

if __name__ == "__main__":
    main()
