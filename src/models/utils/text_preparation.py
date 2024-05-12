import re

def clean_text(text):
    """
    Cleans the input text by converting to lowercase and removing non-alphanumeric characters.

    Parameters:
    - text (str): The text to clean.

    Returns:
    - str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
