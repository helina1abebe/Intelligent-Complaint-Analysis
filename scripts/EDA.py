import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from typing import List

# --- Helper Functions ---

def _clean_text(text: str) -> str:
    """
    Cleans a given text string by converting to lowercase, removing boilerplate,
    special characters, and extra spaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove boilerplate phrases
    text = re.sub(r"i am writing to file a complaint[.,]?", "", text)
    text = re.sub(r"thank you for your (time|attention)[.,]?", "", text)
    
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

# --- Core EDA and Processing Functions ---

def convert_csv_to_parquet(csv_path: str, parquet_path: str):
    """
    Loads data from a CSV file and saves it in Parquet format.
    Handles potential DtypeWarning.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(parquet_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Loading CSV from: {csv_path}...")
        # Use low_memory=False to handle DtypeWarning, common with large CSVs
        df = pd.read_csv(csv_path, low_memory=False)
        print("Saving to Parquet format...")
        df.to_parquet(parquet_path, index=False)
        print(f"✅ Saved Parquet file to: {parquet_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a Parquet file into a pandas DataFrame.
    """
    try:
        print(f"Loading data from: {file_path}...")
        df = pd.read_parquet(file_path)
        print("✅ Data loaded successfully.")
        print("Dataset shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

def plot_product_distribution(df: pd.DataFrame):
    """
    Generates and displays a bar plot of complaint counts by product.
    """
    print("Analyzing distribution of complaints across products...")
    product_counts = df['Product'].value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(x=product_counts.index, y=product_counts.values, palette="viridis")
    plt.title("Distribution of Complaints by Product", fontsize=14)
    plt.xlabel("Product", fontsize=12)
    plt.ylabel("Number of Complaints", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def analyze_narrative_word_count(df: pd.DataFrame):
    """
    Calculates and plots the distribution of word counts in complaint narratives.
    """
    print("\nAnalyzing word count in complaint narratives...")
    pd.options.display.float_format = '{:.2f}'.format
    
    narratives = df['Consumer complaint narrative'].fillna("")
    word_counts = narratives.apply(lambda x: len(x.split()))
    
    print("\nSummary statistics for word counts:")
    print(word_counts.describe())
    
    plt.figure(figsize=(10, 5))
    sns.histplot(word_counts, bins=50, kde=True, color='teal')
    plt.title("Distribution of Word Counts in Complaint Narratives")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.show()

def plot_narrative_availability(df: pd.DataFrame):
    """
    Generates and displays a pie chart showing the proportion of complaints
    with and without narratives.
    """
    print("\nAnalyzing availability of complaint narratives...")
    df['has_narrative'] = df['Consumer complaint narrative'].notna() & (df['Consumer complaint narrative'] != '')
    narrative_counts = df['has_narrative'].value_counts()
    
    with_narratives = narrative_counts.get(True, 0)
    without_narratives = narrative_counts.get(False, 0)
    
    print(f"Complaints with narratives: {with_narratives}")
    print(f"Complaints without narratives: {without_narratives}")
    
    plt.figure(figsize=(6, 6))
    plt.pie(
        narrative_counts, 
        labels=['Has Narrative', 'Missing Narrative'], 
        autopct='%1.1f%%', 
        colors=['skyblue', 'salmon']
    )
    plt.title("Complaints With vs Without Narratives")
    plt.show()

def show_unique_products(df: pd.DataFrame):
    """
    Prints the unique product categories found in the dataset.
    """
    print("\nUnique product categories in the dataset:")
    unique_products = df['Product'].unique()
    for product in unique_products:
        print(f"- {product}")

def filter_and_process_complaints(df: pd.DataFrame, target_products: List[str]) -> pd.DataFrame:
    """
    Filters the DataFrame for target products, removes rows with empty narratives,
    and applies text cleaning.
    """
    print("\nFiltering and cleaning data...")
    
    # Step 1: Filter to include only the specified products
    df_filtered = df[df['Product'].isin(target_products)].copy()
    print(f"Shape after filtering for target products: {df_filtered.shape}")
    
    # Step 2: Remove rows with empty or null 'Consumer complaint narrative'
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notnull()]
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].str.strip() != ""]
    print(f"Shape after removing empty narratives: {df_filtered.shape}")
    
    # Step 3: Apply cleaning to the narrative column
    df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].apply(_clean_text)
    print("✅ Text cleaning applied.")
    
    return df_filtered

def save_data(df: pd.DataFrame, output_path: str):
    """
    Saves the DataFrame to a CSV file.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_csv(output_path, index=False)
        print(f"✅ Filtered and cleaned data saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")

# --- Main Execution Block ---

def main():
    """
    Main function to run the full EDA and processing pipeline.
    """
    # --- Configuration ---
    BASE_DATA_DIR = '../data'
    RAW_CSV_PATH = os.path.join(BASE_DATA_DIR, 'complaints.csv')
    RAW_PARQUET_PATH = os.path.join(BASE_DATA_DIR, 'raw_complaints.parquet')
    FILTERED_CSV_PATH = os.path.join(BASE_DATA_DIR, 'filtered_complaints_2.csv')

    TARGET_PRODUCTS = [
        "Credit card",
        "Credit card or prepaid card",
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
        "Checking or savings account",
        "Money transfers",
        "Money transfer, virtual currency, or money service"
    ]

    # --- Pipeline ---
    
    # 1. Convert CSV to Parquet (more efficient for large datasets)
    if not os.path.exists(RAW_PARQUET_PATH):
        convert_csv_to_parquet(RAW_CSV_PATH, RAW_PARQUET_PATH)

    # 2. Load data for analysis
    df = load_data(RAW_PARQUET_PATH)
    if df is None:
        return

    # 3. Perform Exploratory Data Analysis
    plot_product_distribution(df)
    analyze_narrative_word_count(df)
    plot_narrative_availability(df)
    show_unique_products(df)

    # 4. Filter and clean the data based on requirements
    df_filtered = filter_and_process_complaints(df, TARGET_PRODUCTS)
    print("\nSample of cleaned data:")
    print(df_filtered[['Product', 'Consumer complaint narrative', 'cleaned_narrative']].sample(3, random_state=42))

    # 5. Save the final processed data
    save_data(df_filtered, FILTERED_CSV_PATH)

if __name__ == "__main__":
    main()