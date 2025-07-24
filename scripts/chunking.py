import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_complaints(df, chunk_size=500, chunk_overlap=100):
    """
    Splits cleaned complaint narratives into overlapping text chunks.

    Args:
        df (pd.DataFrame): DataFrame with columns 'cleaned_narrative', 'Complaint ID', 'Product'
        chunk_size (int): Number of characters per chunk
        chunk_overlap (int): Number of overlapping characters between chunks

    Returns:
        pd.DataFrame: Chunked text with complaint ID and product metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for _, row in df.iterrows():
        text = row['cleaned_narrative']
        complaint_id = row['Complaint ID']
        product = row['Product']

        split_texts = splitter.split_text(text)

        for chunk in split_texts:
            chunks.append({
                "complaint_id": complaint_id,
                "product": product,
                "text_chunk": chunk
            })

    return pd.DataFrame(chunks)

def run_chunking_experiments(df, configurations, output_dir="../data"):
    """
    Run multiple chunking configurations and save the outputs.

    Args:
        df (pd.DataFrame): Filtered and cleaned complaint data
        configurations (list): List of dicts with chunk_size and chunk_overlap
        output_dir (str): Where to save the chunked CSVs
    """
    for config in configurations:
        size = config["chunk_size"]
        overlap = config["chunk_overlap"]

        print(f"\n🔍 Testing chunk_size={size}, chunk_overlap={overlap}")
        chunked_df = chunk_complaints(df, size, overlap)

        total_chunks = len(chunked_df)
        avg_length = chunked_df['text_chunk'].apply(len).mean()

        print(f"Total chunks: {total_chunks}")
        print(f"Average chunk length: {avg_length:.2f} characters")

        # Save to CSV
        filename = f"{output_dir}/chunked_complaints_{size}_{overlap}.csv"
        chunked_df.to_csv(filename, index=False)
        print(f"✅ Saved to {filename}")
