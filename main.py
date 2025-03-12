# Loads the pre-built FAISS index from disk
# Loads the mapping file (catalog_map.json)
# Defines the Flask app with the endpoint /rfq:
         # Receives the user’s RFQ
         # Extracts product details using GPT
         # Enhances product description (also via GPT)
         # Gets the embedding for that query text
         # Uses the loaded FAISS index to find the nearest neighbors
         # Builds and returns a JSON response consistent with your schema

import os
import re
import json
import openai
import faiss
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv

#  JSON Response Schema
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_name": {"type": "string"},
        "total_batches": {"type": "integer"},
        "matched_products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_string": {"type": "string"},
                    "enhanced_string": {"type": "string"},
                    "best_match": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer"},
                            "product_id": {"type": "string"},
                            "brand": {"type": "string"},
                            "product_name": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "slab_sp_excl_tax": {"type": "string"}
                        },
                        "required": [
                            "rank", 
                            "product_id", 
                            "brand", 
                            "product_name", 
                            "quantity", 
                            "slab_sp_excl_tax"
                        ]
                    },
                    "top_5_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rank": {"type": "integer"},
                                "product_id": {"type": "string"},
                                "brand": {"type": "string"},
                                "product_name": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "slab_sp_excl_tax": {"type": "string"}
                            },
                            "required": [
                                "rank", 
                                "product_id", 
                                "brand", 
                                "product_name", 
                                "quantity", 
                                "slab_sp_excl_tax"
                            ]
                        }
                    }
                },
                "required": [
                    "original_string",
                    "enhanced_string",
                    "best_match",
                    "top_5_matches"
                ]
            }
        }
    },
    "required": ["customer_name", "total_batches", "matched_products"]
}

#  Load environment & set up
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

INDEX_PATH = "catalog.index"
CATALOG_MAP_PATH = "catalog_map.json"

app = Flask(__name__)

#  Load the FAISS index and catalog map (once)
try:
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(CATALOG_MAP_PATH, "r", encoding="utf-8") as f:
        catalog_map = json.load(f)
    print("FAISS index and catalog map loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index or catalog map: {e}")

#  Utility Functions
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_embedding(query_text: str) -> np.ndarray:
    """Generate embedding for the query text."""
    query_text = normalize_text(query_text)
    try:
        response = openai.Embedding.create(
            input=[query_text],
            model=EMBEDDING_MODEL
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Error creating embedding for '{query_text}': {e}")
        return None

def elaborate_rfq_description(rfq_text: str) -> str:
    """
    Enhancement Layer for a more descriptive product string
    (limit ~30 words, no extra sizes).
    """
    prompt = f"""
    You are an expert in Indian office supplies. Given a customer's RFQ request, generate a detailed product description 
    to enhance semantic search in the product catalog. Keep it concise but more informative. 
    Don't add any size descriptions unless mentioned in the original string. Limit to 20 words.

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
        return rfq_text  # fallback

def extract_product_details(rfq_text: str):
    """
    Extract product details (item, brand, qty) from RFQ input.
    Return in structured JSON format (list of dicts).
    """
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

#  Flask Endpoint
@app.route('/rfq', methods=['POST'])
def rfq_search():
    """
    POST /rfq
    Expects JSON of:
      {
        "customer_name": "string",
        "rfq": "string",
        "slab": "string"
      }
    Returns JSON in the schema specified by RESPONSE_SCHEMA.
    """
    data = request.get_json()

    # Basic validation
    if not data or 'rfq' not in data or 'slab' not in data or 'customer_name' not in data:
        return jsonify({"error": "Invalid request. Provide 'customer_name', 'rfq', and 'slab' fields in JSON."}), 400

    customer_name = data['customer_name']
    rfq_input = data['rfq']
    selected_slab = data['slab']

    # If our RFQ is comma-separated for each item, count them:
    total_batches = len(rfq_input.split(','))

    # Extract product data using GPT
    extracted_products = extract_product_details(rfq_input)
    matched_products = []

    for product in extracted_products:
        # Default quantity to 1 if not found
        qty = product.get("qty", 1)
        brand = product.get("brand", "")
        item = product.get("item", "")

        # Build original search string
        original_string = f"{brand} {item}".strip()

        # Enhanced product description
        enhanced_string = elaborate_rfq_description(original_string)

        # Create query embedding
        query_embedding = get_embedding(enhanced_string)
        if query_embedding is None:
            # If embedding fails, skip
            continue

        # Search top 5 matches in FAISS
        _, indices = faiss_index.search(query_embedding.reshape(1, -1), 5)

        top_matches = []
        best_match = None

        for rank, match_idx in enumerate(indices[0]):
            matched_row = catalog_map[str(match_idx)]  # Notice: keys in JSON are strings

            match_entry = {
                "rank": rank + 1,
                "product_id": matched_row.get("SKU", "N/A"),
                "brand": matched_row.get("Brand", "N/A"),
                "product_name": matched_row.get("Description", "N/A"),
                "quantity": qty,
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

    # Prepare response
    response_data = {
        "customer_name": customer_name,
        "total_batches": total_batches,
        "matched_products": matched_products
    }

    return jsonify(response_data), 200

#  App Runner
if __name__ == '__main__':
    # Runs on localhost:5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True)
