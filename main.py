from flask import Flask, request, jsonify
import openai
import faiss
import numpy as np
import pandas as pd
import os
import re
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define models
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

app = Flask(__name__)

# Load the new CSV file
csv_path = "BOS_and_HK_Standard_Catalog - Sheet1.csv"
catalog_df = pd.read_csv(csv_path)

# Fill NaN values
catalog_df = catalog_df.fillna("")

# Function to normalize text for better matching
def normalize_text(text):
    """Perform text normalization: lowercasing, removing special characters, and extra spaces."""
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Normalize catalog product descriptions for better search
catalog_df["Normalized_Description"] = (catalog_df["Brand"] + " " + catalog_df["Description"]).apply(normalize_text)

# Create product text embeddings
product_texts = catalog_df["Normalized_Description"].tolist()
embeddings_list = []
valid_indices = []

for idx, text in enumerate(product_texts):
    emb = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)["data"][0]["embedding"]
    embeddings_list.append(np.array(emb, dtype=np.float32))
    valid_indices.append(idx)

# Create FAISS index
embeddings_array = np.vstack(embeddings_list)
faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
faiss_index.add(embeddings_array)

# Map index to catalog
catalog_map = {i: catalog_df.iloc[valid_indices[i]].to_dict() for i in range(len(valid_indices))}

def get_embedding(text):
    """Generate embedding for input text after normalizing it."""
    text = normalize_text(text)
    try:
        response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def elaborate_rfq_description(rfq_text):
    """Enhancement Layer: Elaborate product descriptions for better semantic search."""
    prompt = f"""
    You are an expert in Indian office supplies. Given a customer's RFQ request, generate a detailed product description 
    to enhance semantic search in the product catalog. Keep it concise but more informative. Don't add any size descriptions unless mentioned in the original string. 

    Example:
    - Input: "Reynolds Pen"
      Output: "Reynolds ballpoint pen, fine tip, blue ink"
    - Input: "HP Printer"
      Output: "HP LaserJet all-in-one printer with wireless connectivity and duplex printing"

    RFQ Request: "{rfq_text}"
    """
    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Expand the product description for better catalog matching."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error elaborating product description: {e}")
        return rfq_text  # Fallback to original string if elaboration fails

def extract_product_details(rfq_text):
    """Extract product details from RFQ using GPT model."""
    prompt = f"""
    Extract product details from the following RFQ request. Identify the item, brand (if available), and quantity (if specified). 
    If no brand is mentioned, return an empty string for brand. If no quantity is mentioned, assume 1.

    Return the output in **valid JSON format** (a list of dictionaries) without any extra text.
    
    Example input:
    - 10 Dell laptops
    - Apple iPhone 14, 5 units
    - HP LaserJet printer

    Example output:
    [
      {{"item": "laptops", "brand": "Dell", "qty": 10}},
      {{"item": "iPhone 14", "brand": "Apple", "qty": 5}},
      {{"item": "LaserJet printer", "brand": "HP", "qty": 1}}
    ]

    Input RFQ:
    {rfq_text}

    Output JSON:
    """
    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Extract structured product information from RFQ text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        extracted_data = response['choices'][0]['message']['content'].strip()
        return json.loads(extracted_data)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Error extracting product details: {e}")

    return []

@app.route('/rfq', methods=['POST'])
def rfq_search():
    """Handle RFQ search and return best match and top 5 matches with both original and enhanced product strings."""
    data = request.get_json()

    if not data or 'rfq' not in data or 'slab' not in data or 'customer_name' not in data:
        return jsonify({"error": "Invalid request. Provide 'customer_name', 'rfq', and 'slab' fields in JSON."}), 400

    customer_name = data['customer_name']
    rfq_input = data['rfq']
    selected_slab = data['slab']
    total_batches = len(rfq_input.split(','))

    extracted_products = extract_product_details(rfq_input)
    matched_products = []

    for product in extracted_products:
        product["qty"] = product.get("qty", 1)

        # Original string from extracted details
        original_string = f"{product['brand']} {product['item']}".strip()
        
        # Generate enhanced product description using the elaboration layer
        enhanced_string = elaborate_rfq_description(original_string)

        query_embedding = get_embedding(enhanced_string)
        if query_embedding is None:
            continue

        _, indices = faiss_index.search(query_embedding.reshape(1, -1), 5)

        top_matches = []
        best_match = None

        for rank, match_idx in enumerate(indices[0]):
            matched_row = catalog_map[match_idx]

            match_entry = {
                "rank": rank + 1,
                "product_id": matched_row.get("SKU", "N/A"),
                "brand": matched_row.get("Brand", "N/A"),
                "product_name": matched_row.get("Description", "N/A"),
                "quantity": product["qty"],
                "slab_sp_excl_tax": matched_row.get(f"Slab_{selected_slab}_SP_Excl_Tax", "N/A")
            }

            if rank == 0:
                best_match = match_entry

            top_matches.append(match_entry)

        matched_products.append({
            "original_string": original_string,
            "enhanced_string": enhanced_string,
            "best_match": best_match,
            "top_5_matches": top_matches
        })

    return jsonify({
        "customer_name": customer_name,
        "total_batches": total_batches,
        "matched_products": matched_products
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
