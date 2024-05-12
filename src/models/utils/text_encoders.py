import re
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer

def encodeText(text, model=None, model_name="all-MiniLM-L6-v2"):
    """
    Encodes the text using a provided SentenceTransformer model.

    Parameters:
    - text (str): The cleaned text to be encoded.
    - model (SentenceTransformer, optional): A pre-loaded SentenceTransformer model. If None, a new model is loaded.
    - model_name (str, optional): The model name, used only if no model is provided.

    Returns:
    - Tensor: Encoded text as a tensor.
    """
    # Load model if not provided
    if model is None:
        model = SentenceTransformer(model_name)

    prompt_embedding = model.encode(text, convert_to_tensor=True)

    return prompt_embedding

def encodeDataFrameColumn(df, input_col='Description', output_col='Encoded Description', model=None, model_name="all-MiniLM-L6-v2"):
    """
    Encodes a specified column in a DataFrame using a SentenceTransformer model.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the text column to be encoded.
    - input_col (str, optional): The name of the column to encode. Defaults to 'Description'.
    - output_col (str, optional): The name for the column to store encoded results. Defaults to 'Encoded Description'.
    - model (SentenceTransformer, optional): A pre-loaded SentenceTransformer model. If None, a new model is loaded.
    - model_name (str, optional): The model name, used only if no model is provided.

    Returns:
    - pd.DataFrame: DataFrame with an additional column containing encoded data.
    """

    # Load the SentenceTransformer model if not provided
    if model is None:
        model = SentenceTransformer(model_name)

    # Encode the specified column
    df[output_col] = df[input_col].apply(lambda x: model.encode(x, convert_to_tensor=True))

    return df
