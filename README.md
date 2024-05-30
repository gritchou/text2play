# Text2Play 🎮🎨

**Bringing Art to Life through AI**

## Overview
Text2Play is an innovative web application that transforms user prompts into a 2D platform video game. By leveraging AI techniques, it converts text into famous paintings and applies style transfer to create a unique game background.

## Features
- **Word Vectorization:** Utilizes BERT for word embeddings to understand and interpret user prompts.
- **Style Transfer:** Applies VGG19 with PyTorch to transfer the style of famous paintings onto the game's background.
- **Game Generation:** Integrates the generated background into a 2D platform game using Phaser.io.

## Tech Stack
- **Backend:** Python, Flask
- **AI Models:** BERT, VGG19, PyTorch
- **Frontend:** React, Phaser.io
- **Deployment:** Google Cloud Run, Docker, MLflow for model orchestration

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/gritchou/Text2Play.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Text2Play
    ```
3. Install dependencies:
    ```sh
    make install
    ```
4. Preprocess the dataset:
    ```sh
    make preprocess-dataset
    ```
5. Run the application:
    ```sh
    make run
    ```

## Frontend Setup
1. Clone the frontend repository:
    ```sh
    git clone https://github.com/gritchou/Text2Play-frontend.git
    ```
2. Navigate to the frontend project directory:
    ```sh
    cd Text2Play-frontend
    ```
3. Install frontend dependencies:
    ```sh
    yarn install
    ```
4. Start the frontend application:
    ```sh
    yarn start
    ```

## Usage
1. Enter a prompt describing the scene.
2. The backend processes the prompt using BERT for word vectorization.
3. VGG19 applies style transfer to create a game background.
4. The frontend displays the 2D platform game with the generated background.

## Contributors
- **Jean-François Grand** - [LinkedIn](https://www.linkedin.com/in/jfgrand) | [GitHub](https://github.com/gritchou)
- **Guillaume Padan**
- **Shuja Khan**

## References
- [Neural-Style algorithm](https://arxiv.org/abs/1508.06576)
- [Neural Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [BERT Model](https://huggingface.co/docs/transformers/en/model_doc/bert)
- [VGG19](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html)

## License
This project is licensed under the MIT License.
