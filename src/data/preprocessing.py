import pandas as pd
import os
import sys
import re

def process_paintings_dataset(input_path, input_file_name, output_path, output_file_name):
    """
    Processes the paintings dataset by loading it, cleaning the text, removing rows with NA values,
    and saving the processed data to a new location.

    Parameters:
    - input_path (str): Path to the directory containing the raw dataset.
    - output_path (str): Path to the directory where the processed dataset will be saved.
    - file_name (str): Name of the dataset file.
    """

    input_file = os.path.join(input_path, input_file_name)
    output_file = os.path.join(output_path, output_file_name)

    # Load the dataset
    df = pd.read_csv(input_file)

    # Remove na rows then clean the description
    df = remove_na_rows(df)
    df['Description'] = df['Description'].apply(clean_html)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the processed DataFrame
    df.to_csv(output_file, index=False)

    print(f"Processed dataset saved to {output_file}")

def clean_html(text):
    """
    Cleans HTML tags and entities from the input text.

    Parameters:
    - text (str): The HTML text to clean.

    Returns:
    - str: The cleaned text with HTML tags and entities removed.

    Description:
    - This function should remove all HTML tags, handle or decode HTML entities,
      and ensure the text is readable and clean for further processing or analysis.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""  # Return an empty string or handle it as needed

    # Define a pattern to remove HTML tags and other entities
    pattern_to_remove = r'<[^>]+>|\[/?i\]|\[/?b\]|\[/?u\]|\[url href=[^\]]*\][^\[]*\[/url\]'
    cleaned_text = re.sub(pattern_to_remove, '', text)

    return cleaned_text


    return cleaned_text

def remove_na_rows(df):
    """
    Removes rows from the DataFrame where there are NaN values in either
    'Description' or 'Image URL' columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    - pd.DataFrame: The DataFrame with NA rows removed from specified columns.
    """
    # Drop rows where either 'Description' or 'Image URL' is NaN
    cleaned_df = df.dropna(subset=['Description', 'Image URL'])

    return cleaned_df

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python preprocessing.py <input_path> <input_file_name>, <output_path> <output_file_name>")
        sys.exit(1)

    process_paintings_dataset(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
