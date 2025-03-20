"""
indexing.py

- One-time script to read your CSV catalog,
- Create embeddings for each product,
- Build a FAISS index,
- Save the index to disk,
- And save a mapping JSON relating each FAISS index position to the row data.

Usage:
    python indexing.py
"""

import os
import re
import json
import openai
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# The embedding model to use
EMBEDDING_MODEL = "text-embedding-3-large"

# Path to your CSV file
CSV_PATH = "BOS_and_HK_Standard_Catalog - Sheet1.csv"

# Where the FAISS index will be stored
INDEX_OUTPUT_PATH = "catalog.index"

# Where the index-to-row mapping will be stored
MAP_OUTPUT_PATH = "catalog_map.json"

def normalize_text(text: str) -> str:
    """
    Lowercase, remove special characters, remove extra spaces.
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for the input text after normalizing it.
    Returns a NumPy float32 array or None if there's an error.
    """
    text = normalize_text(text)
    try:
        response = openai.Embedding.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Embedding error for '{text}': {e}")
        return None

def build_faiss_index():
    """
    1. Load the CSV file into a DataFrame.
    2. Create a normalized description (Brand + Description).
    3. Generate embeddings for each row.
    4. Build the FAISS index.
    5. Save the index to disk.
    6. Create & save a catalog map from index->row data.
    """
    df = pd.read_csv(CSV_PATH).fillna("")
    df["Normalized_Description"] = (
        df["Brand"] + " " + df["Description"]
    ).apply(normalize_text)

    product_texts = df["Normalized_Description"].tolist()

    embeddings_list = []
    valid_indices = []

    for idx, text in enumerate(product_texts):
        emb = get_embedding(text)
        if emb is not None:
            embeddings_list.append(emb)
            valid_indices.append(idx)
        else:
            print(f"Skipping row {idx} due to embedding error.")

    if not embeddings_list:
        raise ValueError("No embeddings were created. Check your CSV or embedding function.")

    embeddings_array = np.vstack(embeddings_list)
    dimension = embeddings_array.shape[1]

    # Create FAISS index (L2)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    print(f"FAISS index saved to {INDEX_OUTPUT_PATH}")

    # Build catalog map
    catalog_map = {}
    for i, idx_in_df in enumerate(valid_indices):
        row_data = df.iloc[idx_in_df].to_dict()
        catalog_map[i] = row_data

    # Save as JSON
    with open(MAP_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog_map, f, ensure_ascii=False, indent=2)
    print(f"Catalog map saved to {MAP_OUTPUT_PATH}")

if __name__ == "__main__":
    build_faiss_index()
