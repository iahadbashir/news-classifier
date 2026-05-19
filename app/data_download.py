"""Script to download, process, and sample AG News and WELFake datasets."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# AG News Kaggle CSV columns: "Class Index", "Title", "Description"
# Class Index is 1-based: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
AG_NEWS_PATH = "data/raw/ag_news_train.csv"

def process_ag_news():
    """
    Load AG News from a local Kaggle CSV, map classes, and save a stratified sample.

    Download train.csv from:
    https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
    and place it at data/raw/ag_news_train.csv
    """
    print(f"Loading AG News from local file: {AG_NEWS_PATH} ...")

    if not os.path.exists(AG_NEWS_PATH):
        print(f"Error: AG News CSV not found at '{AG_NEWS_PATH}'.")
        print("Please download train.csv from:")
        print("  https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset")
        print("and place it at: data/raw/ag_news_train.csv")
        return

    try:
        # Kaggle CSV has no header row; columns are: Class Index, Title, Description
        df = pd.read_csv(AG_NEWS_PATH, header=None,
                         names=['label', 'title', 'description'])
    except Exception as e:
        print(f"Error reading AG News CSV: {e}")
        return

    # Kaggle labels are 1-based; remap to 0-based
    # 1=World->0=Politics, 2=Sports->1=Sports, 3=Business->2=Business, 4=Sci/Tech->3=Technology
    label_map = {1: 'Politics', 2: 'Sports', 3: 'Business', 4: 'Technology'}
    int_map   = {1: 0,          2: 1,         3: 2,          4: 3}

    df['label_name'] = df['label'].map(label_map)
    df['label']      = df['label'].map(int_map)

    # Combine title + description as the full text
    df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')

    clean_df = df[['text', 'label', 'label_name']].dropna()

    # Print distribution
    print("\nAG News Full Distribution:")
    print(clean_df['label_name'].value_counts())

    # Stratified sample of 200 rows (50 per class)
    _, sample_df = train_test_split(
        clean_df,
        test_size=200,
        stratify=clean_df['label'],
        random_state=42
    )

    # Save sample
    os.makedirs('data/samples', exist_ok=True)
    sample_df.to_csv('data/samples/ag_news_sample.csv', index=False)
    print("\nAG News Sample Distribution (data/samples/ag_news_sample.csv):")
    print(sample_df['label_name'].value_counts())

def process_welfake():
    """Load WELFake dataset from local CSV, process, and save a stratified sample."""
    file_path = 'data/raw/WELFake_Dataset.csv'
    print(f"\nLoading WELFake dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find WELFake dataset at '{file_path}'.")
        print("Please download it and place it in the 'data/raw/' directory.")
        return
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading WELFake CSV: {e}")
        return
        
    # Combine title and text if both exist, otherwise use text
    if 'title' in df.columns and 'text' in df.columns:
        df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    elif 'text' in df.columns:
        df['combined_text'] = df['text'].fillna('')
    else:
        print("Error: WELFake dataset must contain 'text' column.")
        return
        
    # Mapping label: 0=Fake, 1=Real
    label_map = {0: 'Fake', 1: 'Real'}
    df['label_name'] = df['label'].map(label_map)
    
    clean_df = pd.DataFrame({
        'text': df['combined_text'],
        'label': df['label'],
        'label_name': df['label_name']
    })
    
    # Drop rows with NaN labels
    clean_df = clean_df.dropna(subset=['label'])
    clean_df['label'] = clean_df['label'].astype(int)
    
    # Print distribution
    print("\nWELFake Full Distribution:")
    print(clean_df['label_name'].value_counts())
    
    # Stratified sample of 200 rows
    try:
        _, sample_df = train_test_split(
            clean_df, 
            test_size=200, 
            stratify=clean_df['label'], 
            random_state=42
        )
    except ValueError as e:
        print(f"Error creating stratified sample: {e}")
        sample_df = clean_df.sample(n=min(200, len(clean_df)), random_state=42)
    
    # Save sample
    os.makedirs('data/samples', exist_ok=True)
    sample_df.to_csv('data/samples/welfake_sample.csv', index=False)
    print("\nWELFake Sample Distribution (data/samples/welfake_sample.csv):")
    print(sample_df['label_name'].value_counts())

def main():
    process_ag_news()
    process_welfake()

if __name__ == "__main__":
    main()
