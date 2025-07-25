import pandas as pd
import re


def filter_products_and_narratives(df):
    """
    Filters the dataframe to include only specific product categories
    and removes complaints with missing/empty narratives.
    """
    target_products = [
        "Credit card",
        "Credit card or prepaid card",
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
        "Checking or savings account",
        "Money transfers",
        "Money transfer, virtual currency, or money service"
    ]

    # Filter by product
    df_filtered = df[df['Product'].isin(target_products)].copy()

    # Remove missing/empty narratives
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notnull()]
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].str.strip() != ""]

    print(f"✅ After filtering: {df_filtered.shape[0]} complaints")
    return df_filtered


def clean_text(text):
    """
    Cleans a single narrative string:
    - Lowercases text
    - Removes boilerplate phrases
    - Removes special characters
    - Collapses extra spaces
    """
    if pd.isnull(text):
        return ""

    text = text.lower()
    text = re.sub(r"i am writing to file a complaint[.,]?", "", text)
    text = re.sub(r"thank you for your (time|attention)[.,]?", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def clean_narratives_column(df, source_col="Consumer complaint narrative", target_col="cleaned_narrative"):
    """
    Applies text cleaning to the narrative column and adds a new column to the DataFrame.
    """
    df[target_col] = df[source_col].apply(clean_text)
    print(f"🧹 Cleaned narratives stored in '{target_col}' column.")
    return df
