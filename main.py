"""
main.py

- Loads the pre-built FAISS index (catalog.index) and the catalog map (catalog_map.json).
- Defines a Flask app with endpoint /rfq to:
  1) Parse incoming JSON (customer_name, rfq, slab).
  2) Split the incoming RFQ text if it is very long (over 300 chars), looking for newlines near 300 characters.
  3) Extract structured product details from each chunk of the RFQ text (GPT).
  4) Optionally elaborate product descriptions (GPT) using chunking (one call per chunk of items).
  5) Convert the elaborated text to embeddings (OpenAI) in a single batch call for each chunk.
  6) Query the FAISS index for top 5 matches per item.
  7) Re-rank if confidence scores differ by 1% or less, favoring higher sale quantity.
  8) If top match’s confidence score < 0.50, replace all best_match fields with “-”.
  9) Return a JSON response following the defined RESPONSE_SCHEMA.

Usage:
    python main.py
    (Then test via Postman or cURL on http://localhost:5000/rfq)
"""

import os
import re
import json
import openai
import faiss
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv


# JSON schema for the final response
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
                            "slab_sp_excl_tax": {"type": "string"},
                            "sale_qty": {"type": "integer"},
                            "confidence_score": {"type": "number"}
                        },
                        "required": [
                            "rank",
                            "product_id",
                            "brand",
                            "product_name",
                            "quantity",
                            "slab_sp_excl_tax",
                            "sale_qty",
                            "confidence_score"
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
                                "slab_sp_excl_tax": {"type": "string"},
                                "sale_qty": {"type": "integer"},
                                "confidence_score": {"type": "number"}
                            },
                            "required": [
                                "rank",
                                "product_id",
                                "brand",
                                "product_name",
                                "quantity",
                                "slab_sp_excl_tax",
                                "sale_qty",
                                "confidence_score"
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

# Load environment variables (OpenAI key, etc.)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define our model references
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o-mini"

# Paths to FAISS index and catalog map
INDEX_PATH = "catalog.index"
CATALOG_MAP_PATH = "catalog_map.json"

# Create the Flask app
app = Flask(__name__)

# Load the FAISS index and the catalog map at startup
try:
    # Read the FAISS index from disk
    faiss_index = faiss.read_index(INDEX_PATH)
    # Load the JSON file that maps index positions to row data
    with open(CATALOG_MAP_PATH, "r", encoding="utf-8") as f:
        catalog_map = json.load(f)
    print("FAISS index and catalog map loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index or catalog map: {e}")

def normalize_text(text: str) -> str:
    """
    Lowercases the input, strips extra spaces, and removes special characters
    so the text is in a consistent format for embedding.
    """
    text = text.lower().strip()
    # Remove non-alphanumeric (but keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def get_embeddings_batch(strings: list[str]) -> list[np.ndarray]:
    """
    Create embeddings for a list of strings in a single API call to reduce time complexity.
    Returns a list of np.float32 arrays, one embedding per input string.
    If there's an error, returns an empty or partial list.
    """
    # Normalize each string before sending to the Embedding endpoint
    cleaned = [normalize_text(s) for s in strings]
    try:
        # Make a single call to OpenAI's Embedding.create
        response = openai.Embedding.create(
            input=cleaned,
            model=EMBEDDING_MODEL
        )
        # We'll collect the embeddings from response["data"]
        result_embeddings = []
        for emb_obj in response["data"]:
            arr = np.array(emb_obj["embedding"], dtype=np.float32)
            result_embeddings.append(arr)
        return result_embeddings
    except Exception as e:
        print(f"Error creating embeddings for batch: {e}")
        return []

def elaborate_rfq_descriptions_batch(original_strings):
    """
    Creates a single GPT call to elaborate up to 15 items at once.
    Returns a list of the same length, each item is an "enhanced" string.
    If GPT fails or returns invalid format, we fallback to the original strings.
    """
    # Build lines enumerating each item for the prompt
    lines = []
    for idx, orig in enumerate(original_strings, start=1):
        lines.append(f"{idx}) {orig}")

    # Prompt instructing GPT to return a JSON array of elaborations
    chunk_prompt = f"""
You are an expert in Indian office supplies. 
We have {len(original_strings)} items below. For each, create a concise, keyword-rich 
description. Follow the rules:

You are an expert in Indian office supplies. Your primary goal is to convert customer queries into highly precise and concise keyword-rich product descriptions optimized for semantic search against a product catalog. Customer queries will be shared in an array, process them sequentially. You must also normalize ambiguous or implicit size descriptions into explicit, widely recognized sizes. When customers use synonyms or variant descriptions (e.g., “broom stick with metal handle” instead of just “broom”), incorporate generally accepted synonyms or standardized terms to improve matching accuracy. Follow these strict rules:

Prioritize Exact Product Identification:

Focus on extracting and representing the core product and its defining characteristics.
Avoid generating keywords that lead to related but distinct products (e.g., a query for “stapler” should not emphasize “stapler pin”).
Adhere to Explicit Specifications and Convert Implicit Sizes:

Explicit Size is Paramount: If a customer mentions a specific size (e.g., “A4 paper,” “king size envelope,” “10x12 inch file”), include that size prominently in the output.
Convert Implicit Sizes: If the query uses an implicit size term (e.g., “pocket diary,” “legal size paper,” “desktop calendar”), use your domain knowledge to convert it to a widely recognized explicit size (e.g., “A6 diary” or “A5 diary,” “foolscap paper,” “standard desktop calendar”). When multiple common sizes exist, select the most likely one based on general usage.
Other Explicit Details: Always include any additional specifications such as type, color, or specific features as mentioned.
Brand Prioritization: Focus primarily on the product rather than the brand. Include the brand only if explicitly mentioned. However, add “workstore” as a secondary generic brand for non-technical products where it aids in semantic matching.
Infer Defaults Cautiously and Only When Necessary:

Apply default inferences (e.g., blue ink for pens, A4 for paper, LaserJet for HP printers, 24-sheet capacity for staplers) only when the customer query lacks these details and no size or specification is provided.
Ignore defaults if the customer provides an alternative specification or an implicit size.
Maintain Numeric Precision:

Always include numeric details exactly as specified or inferred (e.g., “0.7mm tip,” “75GSM,” or provided dimensions).
Output Concise and Focused Keywords:

Aim for 8-12 comma-separated keywords/phrases that are most crucial for accurate semantic matching.
Exclude unnecessary adjectives, adverbs, or extraneous descriptions that do not significantly enhance product distinction.
Emphasize Core Product Keywords, Minimize Verbosity:

Prioritize the core product name and its key attributes (including explicit size if available or converted).
Omit non-essential descriptors that might dilute the semantic signal.
Avoid Matching Parts Instead of Whole Products:

For queries referring to whole products (e.g., “stapler,” “pen”), do not generate keywords for individual components (e.g., “stapler pin,” “pen refill”) unless the component is explicitly requested.
Incorporate Synonyms for Ambiguous Descriptions:

Where customer descriptions use synonyms or less common terms (e.g., “broom stick with metal handle”), include the generally accepted synonym or standardized term (e.g., “broom”) in the output.
Use commonly used synonyms to resolve ambiguities and improve matching with the catalog.

Items:
{lines}

Return a valid JSON array of {len(original_strings)} strings. No extra text or explanation.
"""

    print("Items to elaborate:", lines)
    try:
        # Single GPT call
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Provide a concise product description for each item "
                        "in a single JSON array, no extra text."
                    )
                },
                {"role": "user", "content": chunk_prompt}
            ],
            temperature=0.2
        )
        raw_answer = response["choices"][0]["message"]["content"]
        # Remove triple backticks if present
        raw_answer = raw_answer.replace("```json", "").replace("```", "").strip()

        # Attempt to find a JSON array in the response using a regex
        match = re.search(r'\[.*\]', raw_answer, re.DOTALL)
        print("GPT raw answer for batch elaboration:", raw_answer)

        # If no match found, fallback
        if not match:
            print("No JSON array found in GPT response. Falling back to original strings.")
            return original_strings

        data_str = match.group(0)

        # Parse the JSON array
        data = json.loads(data_str)
        # Must be a list of the same length
        if not isinstance(data, list) or len(data) != len(original_strings):
            print("GPT batch elaboration didn't return correct JSON array length. Falling back.")
            return original_strings

        return data

    except Exception as e:
        print(f"Error in elaborate_rfq_descriptions_batch: {e}")
        # Fallback to original strings if there's any error
        return original_strings

def compute_confidence_score(distance: float) -> float:
    """
    Convert L2 distance to a raw confidence score. 
    Score formula: score = 1/(1+distance).
    """
    return 1.0 / (1.0 + distance)

def split_rfq_text(rfq_text: str, chunk_size: int = 300) -> list[str]:
    """
    If the RFQ text is very long, this function splits it into smaller chunks.
    Specifically, we aim to split at around 300 characters, looking for the next newline
    after 300 characters to avoid cutting a sentence in half. If no newline is found,
    we break at 300.
    """
    chunks = []
    start = 0
    length = len(rfq_text)

    # Loop until we've covered the entire string
    while start < length:
        end = start + chunk_size
        if end >= length:
            # If we are beyond the text length, just add the remainder
            chunks.append(rfq_text[start:])
            break
        else:
            # Attempt to find a newline after the 300th character
            newline_pos = rfq_text.find('\n', end)
            if newline_pos == -1:
                # No newline found; we'll split exactly at chunk_size
                chunks.append(rfq_text[start:end])
                start = end
            else:
                # Found a newline, so chunk up to the newline
                chunks.append(rfq_text[start:newline_pos])
                start = newline_pos + 1
    return chunks

def extract_product_details_chunk(rfq_text: str) -> list[dict]:
    """
    Uses GPT to parse the raw RFQ text. Returns a list of dictionaries like:
      [{"item": ..., "brand": ..., "qty": ...}, ...]
    If GPT fails or returns invalid JSON, returns an empty list.
    This function processes a single chunk of text (under ~300 chars).
    """
    prompt = f"""
    Extract product details from the following RFQ request. The RFQ can contain one or more product requests...
    If brand is missing, brand=""; if qty is missing, qty=1. Return as a valid JSON list only.
    Example input: "Dell laptops 10, HP Printer"
    Example output: [
      {{"item": "Dell laptops", "brand": "Dell", "qty": 10}},
      {{"item": "HP Printer", "brand": "HP", "qty": 1}}
    ]

    Input:
    {rfq_text}

    Output JSON only:
    """
    try:
        # Single GPT call for chunk
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Extract structured product info from RFQ text. Return valid JSON array."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw_json = response["choices"][0]["message"]["content"].strip()
        # Remove potential triple backticks
        raw_json = raw_json.replace("```json", "").replace("```", "").strip()
        if not raw_json:
            print("GPT returned empty for extraction. Using empty list.")
            return []

        parsed = json.loads(raw_json)
        if not isinstance(parsed, list):
            print("GPT extraction didn't return a list. Fallback to empty list.")
            return []
        return parsed
    except json.JSONDecodeError as e:
        print("JSON parsing error during extraction:", e)
        return []
    except Exception as e:
        print("Error extracting product details:", e)
        return []

def extract_product_details(rfq_text: str) -> list[dict]:
    """
    If the RFQ text is longer than 300 characters, we split it into smaller chunks,
    extract each chunk, and merge the results. Returns a combined list of items.
    """
    # 1. Split the text into smaller chunks
    chunks = split_rfq_text(rfq_text, 300)

    # 2. For each chunk, call GPT extraction
    all_extracted = []
    for ch in chunks:
        chunk_extracted = extract_product_details_chunk(ch)
        all_extracted.extend(chunk_extracted)

    return all_extracted

@app.route('/rfq', methods=['POST'])
def rfq_search():
    """
    Endpoint to handle the RFQ search logic.
    We parse the request, chunk the RFQ if it is long, extract items, chunk them for GPT elaboration,
    embed them, query the FAISS index, re-rank, apply the confidence threshold, and build the final response.
    """
    data = request.get_json()
    # Validate input fields
    if not data or 'rfq' not in data or 'slab' not in data or 'customer_name' not in data:
        return jsonify({"error": "Invalid request. Provide 'customer_name', 'rfq', and 'slab'."}), 400

    # Read relevant fields
    customer_name = data['customer_name']
    rfq_input = data['rfq']
    selected_slab = data['slab']

    # Step 1) GPT extraction of product details (using chunking if >300 chars)
    extracted_products = extract_product_details(rfq_input)
    total_batches = len(extracted_products)
    print(f"DEBUG: Extracted products => {extracted_products}")

    # We'll store final matched data in matched_products
    matched_products = []

    # If no products were extracted, return empty response
    if not extracted_products:
        response_data = {
            "customer_name": customer_name,
            "total_batches": 0,
            "matched_products": []
        }
        return jsonify(response_data), 200

    # Step 2) chunk the extracted products in groups of 15 for elaboration & embedding
    CHUNK_SIZE = 15
    chunked_products = [
        extracted_products[i:i+CHUNK_SIZE]
        for i in range(0, len(extracted_products), CHUNK_SIZE)
    ]

    # Step 3) For each chunk, do a single GPT call for elaboration, and a single embedding call
    for chunk in chunked_products:
        # A) Build original_strings from brand+item for each product
        original_strings = []
        for product in chunk:
            qty = product.get("qty", 1)
            brand = product.get("brand", "")
            item_name = product.get("item", "")
            orig_str = f"{brand} {item_name}".strip()
            original_strings.append(orig_str)

        # B) Single GPT call to elaborate all items in the chunk
        elaborated_list = elaborate_rfq_descriptions_batch(original_strings)
        print("Elaborated list:", elaborated_list)

        # C) Single embedding call for the chunk's elaborated strings
        embeddings_list = get_embeddings_batch(elaborated_list)

        # If for some reason embeddings_list is shorter or empty, fallback to None
        if len(embeddings_list) != len(elaborated_list):
            embeddings_list = [None] * len(elaborated_list)

        # D) For each product in this chunk, we do the FAISS search using the batch's embeddings
        for idx, product in enumerate(chunk):
            qty = product.get("qty", 1)
            emb = embeddings_list[idx]
            if emb is None:
                # If embedding failed, skip or fallback
                matched_products.append({
                    "original_string": original_strings[idx],
                    "enhanced_string": elaborated_list[idx],
                    "best_match": {
                        "rank": "-",
                        "product_id": "-",
                        "brand": "-",
                        "product_name": "-",
                        "quantity": "-",
                        "slab_sp_excl_tax": "-",
                        "sale_qty": "-",
                        "confidence_score": "-"
                    },
                    "top_5_matches": []
                })
                continue

            # Query FAISS for top 5 matches
            distances, indices = faiss_index.search(emb.reshape(1, -1), 5)

            # Build top 5 matches
            top_matches_temp = []
            for rank, match_idx in enumerate(indices[0]):
                distance = float(distances[0][rank])
                # Convert distance -> confidence score, round to 2 decimals
                confidence = round(float(compute_confidence_score(distance)), 2)

                # Look up the matching row in catalog_map
                matched_row = catalog_map.get(str(match_idx), {})
                product_id = matched_row.get("SKU", "N/A")
                product_brand = matched_row.get("Brand", "N/A")
                product_name = matched_row.get("Description", "N/A")
                sales_qty = matched_row.get("sale_qty", 0)
                slab_price = matched_row.get(f"Slab_{selected_slab}_SP_Excl_Tax", "N/A")

                # Add the match entry
                match_entry = {
                    "rank": rank + 1,
                    "product_id": product_id,
                    "brand": product_brand,
                    "product_name": product_name,
                    "quantity": qty,
                    "slab_sp_excl_tax": slab_price,
                    "sale_qty": int(sales_qty) if sales_qty else 0,
                    "confidence_score": confidence
                }
                top_matches_temp.append(match_entry)

            # Sort the matches by confidence (descending)
            top_matches_temp.sort(key=lambda x: x["confidence_score"], reverse=True)

            # Re-rank if confidence difference <= 1% among consecutive items, favor higher sale_qty
            i = 0
            while i < len(top_matches_temp) - 1:
                curr = top_matches_temp[i]
                nxt = top_matches_temp[i+1]
                c1 = curr["confidence_score"]
                c2 = nxt["confidence_score"]

                # Only if c1 > 0 to avoid division by zero
                if c1 > 0 and abs(c1 - c2) / c1 <= 0.01:
                    # If next item has a higher sale_qty, swap them
                    if nxt["sale_qty"] > curr["sale_qty"]:
                        top_matches_temp[i], top_matches_temp[i+1] = nxt, curr
                        # Step back to re-check the previous item for consecutive re-sorting
                        if i > 0:
                            i -= 1
                        continue
                i += 1

            # Reassign rank after final reorder
            for new_rank, item_obj in enumerate(top_matches_temp, start=1):
                item_obj["rank"] = new_rank

            # Identify the best match (top of the sorted list)
            best_match = top_matches_temp[0] if top_matches_temp else None

            # If best_match confidence is below 0.50, replace all fields with "-"
            if best_match and best_match["confidence_score"] < 0.50:
                best_match = {
                    "rank": "-",
                    "product_id": "-",
                    "brand": "-",
                    "product_name": "-",
                    "quantity": "-",
                    "slab_sp_excl_tax": "-",
                    "sale_qty": "-",
                    "confidence_score": "-"
                }

            # Collect final data for this item
            matched_products.append({
                "original_string": original_strings[idx],
                "enhanced_string": elaborated_list[idx],
                "best_match": best_match,
                "top_5_matches": top_matches_temp
            })

    # Build the final JSON response
    response_data = {
        "customer_name": customer_name,
        "total_batches": total_batches,
        "matched_products": matched_products
    }
    return jsonify(response_data), 200

if __name__ == '__main__':
    # Runs the Flask app on localhost:5000 in debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)
