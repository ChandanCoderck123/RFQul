# One time index creation
# Step:
    # Reads your CSV catalog
    # Creates embeddings for all products
    # Builds a FAISS index
    # Saves the FAISS index to disk
    # Saves a separate mapping file (catalog_map.json) relating index positions to product data


import os
import re
import json
import openai
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CSV_PATH = "BOS_and_HK_Standard_Catalog - Sheet1.csv"

INDEX_OUTPUT_PATH = "catalog.index"
MAP_OUTPUT_PATH = "catalog_map.json"

# Utility Functions
def normalize_text(text: str) -> str:
    """
    Perform text normalization: 
    - Lowercase
    - Remove special characters
    - Remove extra spaces
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)            # Remove extra spaces
    return text

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for input text after normalizing it."""
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
    1. Load CSV.
    2. Create normalized description.
    3. For each product, get embedding.
    4. Build FAISS index.
    5. Save the FAISS index and a mapping dictionary.
    """
    # Load the catalog
    df = pd.read_csv(CSV_PATH)
    df = df.fillna("")  # Replace NaN with empty string

    # Create a new column for normalized search
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

    # Create the FAISS index
    embeddings_array = np.vstack(embeddings_list)
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)  
    index.add(embeddings_array)

    # Save the index to disk
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    print(f"FAISS index saved to {INDEX_OUTPUT_PATH}")

    # Create a map from index to product row
    catalog_map = {}
    for i, idx_in_df in enumerate(valid_indices):
        row_data = df.iloc[idx_in_df].to_dict()
        catalog_map[i] = row_data

    # Save the catalog map
    with open(MAP_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog_map, f, ensure_ascii=False, indent=2)
    print(f"Catalog map saved to {MAP_OUTPUT_PATH}")


if __name__ == "__main__":
    """
    Run this script one time (or whenever your catalog changes) 
    to build and save the FAISS index + mapping.
    """
    build_faiss_index()
