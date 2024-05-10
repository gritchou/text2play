import numpy as np
from sentence_transformers import util


def prompt2imageURL(promptText, dataset, encodedColumnName='Encoded Description', imageUrlColumnName='Image URL'):
    """
    Finds the most similar image URL in the dataset to the given encoded text.
    """

    cosine_scores = util.pytorch_cos_sim(promptText, dataset[encodedColumnName])

    # Convert the PyTorch tensor to a NumPy array
    cosine_scores_np = cosine_scores.cpu().detach().numpy()

    # Find the index of the highest scoring painting description
    highest_score_index = np.argmax(cosine_scores_np)

    # Retrieve the most similar image URL
    most_similar_url = dataset.iloc[highest_score_index][imageUrlColumnName]

    return most_similar_url
