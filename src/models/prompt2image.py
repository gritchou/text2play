import numpy as np
from sentence_transformers import SentenceTransformer, util
from models.utils.text_preparation import clean_text

def prompt2imageURL(promptText, dataset, descriptionColumnName='Description', imageUrlColumnName='Image URL', model_name="all-MiniLM-L6-v2"):
    """
    Finds the most similar image URL in the dataset to the given text prompt after cleaning and encoding the descriptions.

    Parameters:
    - promptText (str): The text prompt to encode.
    - dataset (pd.DataFrame): DataFrame containing the image descriptions and URLs.
    - descriptionColumnName (str): The name of the column containing text descriptions.
    - imageUrlColumnName (str): The name of the column containing image URLs.
    - model_name (str): The name of the SentenceTransformer model to use for encoding.

    Returns:
    - str: URL of the most similar image based on cosine similarity of descriptions.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Clean the prompt text
    promptText = clean_text(promptText)

    # Encode the cleaned prompt text
    prompt_embedding = model.encode(promptText, convert_to_tensor=True)

    # Clean and encode all descriptions in the dataset
    cleaned_descriptions = dataset[descriptionColumnName].apply(clean_text)
    description_embeddings = model.encode(cleaned_descriptions.tolist(), convert_to_tensor=True)

    # Calculate cosine similarities between the prompt and all descriptions
    cosine_scores = util.pytorch_cos_sim(prompt_embedding, description_embeddings)

    # Convert the PyTorch tensor to a NumPy array
    cosine_scores_np = cosine_scores.cpu().detach().numpy()

    # Find the index of the highest scoring painting description
    highest_score_index = np.argmax(cosine_scores_np)

    # Retrieve the most similar image URL
    most_similar_url = dataset.iloc[highest_score_index][imageUrlColumnName]

    return most_similar_url
