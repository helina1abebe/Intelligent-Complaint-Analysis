import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
from typing import List

# --- Core Processing Functions ---

def chunk_complaints(
    df: pd.DataFrame, 
    text_column: str, 
    id_column: str, 
    product_column: str,
    chunk_size: int = 500, 
    chunk_overlap: int = 100
) -> pd.DataFrame:
    """
    Chunks the text from a specified column in the DataFrame into smaller pieces.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text to be chunked.
        id_column (str): The name of the column with the complaint ID.
        product_column (str): The name of the column with the product category.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        pd.DataFrame: A new DataFrame with original IDs and products, and the text chunks.
    """
    print(f"Chunking text from column '{text_column}'...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = []
    # Ensure the text column is string type and handle NaNs
    df[text_column] = df[text_column].astype(str).fillna('')
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking complaints"):
        text_chunks = text_splitter.split_text(row[text_column])
        for chunk in text_chunks:
            chunks.append({
                'complaint_id': row[id_column],
                'product': row[product_column],
                'text_chunk': chunk
            })
            
    chunk_df = pd.DataFrame(chunks)
    print(f"✅ Created {len(chunk_df)} chunks from {len(df)} complaints.")
    return chunk_df

def create_and_save_faiss_index(
    chunk_df: pd.DataFrame, 
    embedding_model_name: str, 
    vector_store_path: str
):
    """
    Creates a FAISS vector store from chunked text data and saves it to a local path.

    Args:
        chunk_df (pd.DataFrame): DataFrame containing the text chunks and metadata.
        embedding_model_name (str): The name of the Hugging Face sentence-transformer model.
        vector_store_path (str): The local directory path to save the FAISS index.
    """
    # Note: HuggingFaceEmbeddings is deprecated in langchain 0.2.2.
    # The recommended way is: pip install langchain-huggingface
    # from langchain_huggingface import HuggingFaceEmbeddings
    print("Preparing the embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name
    )

    print("📦 Preparing documents with metadata...")
    documents = []
    for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="Creating documents"):
        documents.append(
            Document(
                page_content=row["text_chunk"],
                metadata={
                    "complaint_id": row["complaint_id"],
                    "product": row["product"]
                }
            )
        )
    print(f"✅ Prepared {len(documents)} documents.")
    
    print("⚙️ Embedding and indexing documents...")
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    print(f"💾 Saving FAISS vector store to: {vector_store_path}")
    os.makedirs(vector_store_path, exist_ok=True)
    vector_store.save_local(vector_store_path)
    print("✅ FAISS vector store saved successfully.")

# --- Main Execution Block ---

def main():
    """
    Main function to run the full data chunking and vector store creation pipeline.
    """
    # --- Configuration ---
    BASE_DATA_DIR = '../data'
    FILTERED_CSV_PATH = os.path.join(BASE_DATA_DIR, 'filtered_complaints_2.csv')
    CHUNKED_CSV_PATH = os.path.join(BASE_DATA_DIR, 'chunked_complaints_500_100.csv')
    VECTOR_STORE_PATH = '../vector_store'
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Pipeline ---
    
    # 1. Load the filtered complaints data
    print(f"Loading filtered data from: {FILTERED_CSV_PATH}")
    try:
        df = pd.read_csv(FILTERED_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Filtered data file not found at {FILTERED_CSV_PATH}. Please run the EDA script first.")
        return

    # 2. Chunk the complaint narratives
    chunked_df = chunk_complaints(
        df=df,
        text_column='cleaned_narrative',
        id_column='Complaint ID',
        product_column='Product',
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # 3. Save the intermediate chunked data
    print(f"Saving chunked data to: {CHUNKED_CSV_PATH}")
    chunked_df.to_csv(CHUNKED_CSV_PATH, index=False)
    print("✅ Chunked data saved.")

    # 4. Create and save the FAISS vector store
    create_and_save_faiss_index(
        chunk_df=chunked_df,
        embedding_model_name=EMBEDDING_MODEL,
        vector_store_path=VECTOR_STORE_PATH
    )

if __name__ == "__main__":
    main()