import re
from sentence_transformers import SentenceTransformer

def encodeDescription(df, encoder):

    #Filter the dataframe for Naan and links
    nan = df['Description'].notna()
    df = df[nan]

    # CLEANING
    df['Encoded Description'] = df['Description'].str.lower()  # Convert to lowercase
    df['Encoded Description'] = df['Encoded Description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation

    descriptions = df['Encoded Description'].tolist()  # Update 'description' if using a different column name

    # Load the BERT model from sentence-transformers
    model = SentenceTransformer(encoder)

    # Generate embeddings for the descriptions
    newDataFrameWithEncodedDescriptionColumn = model.encode(descriptions, convert_to_tensor=True)

    return newDataFrameWithEncodedDescriptionColumn


def encodePrompt(prompt, encoder):

    prompt = prompt.str.lower()
    prompt = re.sub(r'[^\w\s]', '', prompt)

    model = SentenceTransformer(encoder)

    prompt_embedding = model.encode(prompt, convert_to_tensor=True)

    return prompt_embedding
