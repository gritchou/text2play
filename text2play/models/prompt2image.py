import numpy as np
from sentence_transformers import util


def prompt2imageURL(promptText, dataset):
    """
    Dataset should be containing an encoded 'description' column
    with the same encoder used on promptText.
    """

    cosine_scores = util.pytorch_cos_sim(promptText, dataset)

    # Convert the PyTorch tensor to a NumPy array
    cosine_scores_np = cosine_scores.cpu().detach().numpy()

    # Find the index of the highest scoring painting description
    highest_score_index = np.argmax(cosine_scores_np)

    # Output the most similar description and its score
    most_similar_url = dataset.iloc[highest_score_index]
    imageURL = most_similar_url['Image URL']

    return imageURL
