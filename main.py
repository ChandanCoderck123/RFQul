"""
main.py

- Loads the pre-built FAISS index (catalog.index) and the catalog map (catalog_map.json).
- Defines a Flask app with endpoint /rfq to:
  1) Parse incoming JSON (customer_name, rfq, slab).
  2) Extract structured product details from the rfq text (GPT).
  3) Optionally elaborate product descriptions (GPT) using chunking (one call per chunk of items).
  4) Convert the elaborated text to embeddings (OpenAI) in a single batch call for each chunk.
  5) Query the FAISS index for top 5 matches per item.
  6) Adjust final ranking if confidence scores are within 5% difference, favoring higher sale quantity.
  7) Return a JSON response following the defined RESPONSE_SCHEMA.

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
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_embeddings_batch(strings: list[str]) -> list[np.ndarray]:
    """
    Create embeddings for a list of strings in a single API call to reduce time complexity.
    Returns a list of np.float32 arrays, one embedding per input string.
    If there's an error, returns an empty or partial list.
    """
    # Normalize each string
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
            # Convert the returned embedding to float32 for FAISS
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
    If GPT fails or returns invalid format, fallback to the original strings.
    """

    # Build lines enumerating each item
    lines = []
    for idx, orig in enumerate(original_strings, start=1):
        lines.append(f"{idx}) {orig}")

    # Construct a chunk prompt that instructs GPT to respond with a JSON array.
    # We re-include the necessary instructions so GPT can produce short, keyword-rich outputs.
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
Examples:

Input: ["1) Reynolds Pen blue ink 0.5mm","2) HP Color LaserJet printer with duplex printing", "3) White paper legal size 80GSM","4) Heavy-duty stapler that can handle 30 sheets", "5) Pocket diary with ruled pages"]
Output: ["ballpoint pen, 0.5mm tip, blue ink, Reynolds, workstore","Color LaserJet printer, duplex, HP","Foolscap white copier paper, 80GSM, workstore","Heavy-duty stapler, 30-sheet capacity","A6 diary, ruled pages"]

Input: ["1) Stamp pad red color"]
Output: ["stamp pad standard, red color, workstore"] (Avoid “stamp pad ink” unless explicitly mentioned)

Input: ["1) Sketch pen set of 12 assorted colors"]
Output: ["sketch pen, set of 12, assorted colors"] (Prioritize “sketch pen” over just “pen”)

Input: ["1) Glory small blades"]
Output: ["small blades, cutter, multi purpose cutter"] (If a more specific standard size exists, use that)

Input: ["1) Small notebook"]
Output: ["Small notebook, Size A5,A6 or A7"] (If a common explicit size for “small notebook” is known, e.g., “A7 notebook,” use that)

Items:
{lines}

Return a valid JSON array of {len(original_strings)} strings. 
No extra text or explanation.
"""
    print(lines)
    try:
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
        print(raw_answer)
        # ---- FIX BELOW: strip out any triple backticks so JSON decoding won't fail ----
        raw_answer = raw_answer.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\[.*\]', raw_answer, re.DOTALL)

        # Print or log the raw GPT answer to debug
        print("GPT raw answer for batch elaboration:", raw_answer)

        # If the raw_answer is empty, fallback
        if not raw_answer:
            print("GPT returned empty. Falling back to original strings.")
            return original_strings

        if not match:
            print("No JSON array found in GPT response. Falling back to original strings.")
            return original_strings

        # Attempt to parse only the matched portion (the array)
        data_str = match.group(0)
        try:
            data = json.loads(data_str)
        except Exception as json_error:
            print(f"Error parsing GPT batch elaboration as JSON: {json_error}")
            print("Raw answer was:", raw_answer)
            print("Falling back to original strings.")
            return original_strings

        # If it's not a list or doesn't match our length, fallback
        if not isinstance(data, list) or len(data) != len(original_strings):
            print("GPT batch elaboration didn't return correct JSON array length. Falling back.")
            return original_strings

        return data

    except Exception as e:
        print(f"Error in elaborate_rfq_descriptions_batch: {e}")
        # Fallback to original strings if there's any error
        return original_strings

def extract_product_details(rfq_text: str) -> list[dict]:
    """
    Uses GPT to parse the raw RFQ text. Returns a list of dictionaries like:
      [{"item": ..., "brand": ..., "qty": ...}, ...]
    If GPT fails or returns invalid JSON, returns an empty list.
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
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Extract structured product info from RFQ text. Return valid JSON array."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw_json = response["choices"][0]["message"]["content"].strip()
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

def compute_confidence_score(distance: float) -> float:
    """
    Convert L2 distance to a confidence score in [0..1].
    Simple approach: score = 1/(1+distance).
    """
    return 1.0 / (1.0 + distance)

@app.route('/rfq', methods=['POST'])
def rfq_search():
    """
    Endpoint to handle the RFQ search logic.
    We parse the request, extract items, chunk them, and do batch GPT calls
    & batch embedding calls for each chunk to reduce time complexity.
    """
    data = request.get_json()
    # Validate input fields
    if not data or 'rfq' not in data or 'slab' not in data or 'customer_name' not in data:
        return jsonify({"error": "Invalid request. Provide 'customer_name', 'rfq', and 'slab'."}), 400

    # Read relevant fields
    customer_name = data['customer_name']
    rfq_input = data['rfq']
    selected_slab = data['slab']

    # Step 1) GPT extraction of product details (un-chunked)
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

    # Step 3) For each chunk, do a single GPT call for elaboration, single Embedding call for all items
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
        print(elaborated_list)

        # C) Single embedding call for the chunk's elaborated strings
        embeddings_list = get_embeddings_batch(elaborated_list)

        # If for some reason embeddings_list is shorter or empty, fallback to None
        if len(embeddings_list) != len(elaborated_list):
            embeddings_list = [None]*len(elaborated_list)

        # D) For each product in this chunk, we do the FAISS search using the batch's embeddings
        for idx, product in enumerate(chunk):
            qty = product.get("qty", 1)
            emb = embeddings_list[idx]
            if emb is None:
                # If embedding failed, skip
                continue

            # Query FAISS for top 5 matches
            distances, indices = faiss_index.search(emb.reshape(1, -1), 5)

            # Build top 5 matches
            top_matches_temp = []
            for rank, match_idx in enumerate(indices[0]):
                # Convert L2 distance to float, then compute a confidence score
                distance = float(distances[0][rank])
                confidence = float(compute_confidence_score(distance))

                # Look up the matching row in catalog_map
                matched_row = catalog_map.get(str(match_idx), {})
                product_id = matched_row.get("SKU", "N/A")
                product_brand = matched_row.get("Brand", "N/A")
                product_name = matched_row.get("Description", "N/A")
                sales_qty = matched_row.get("sale_qty", 0)
                slab_price = matched_row.get(f"Slab_{selected_slab}_SP_Excl_Tax", "N/A")

                # Each match entry
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

            # Sort the matches by confidence desc
            top_matches_temp.sort(key=lambda x: x["confidence_score"], reverse=True)

            # Re-rank if confidence difference <=5% among consecutive items, favor higher sale_qty
            i = 0
            while i < len(top_matches_temp) - 1:
                curr = top_matches_temp[i]
                nxt = top_matches_temp[i+1]
                c1 = curr["confidence_score"]
                c2 = nxt["confidence_score"]
                if c1 > 0 and abs(c1 - c2) / c1 <= 0.05:
                    if nxt["sale_qty"] > curr["sale_qty"]:
                        top_matches_temp[i], top_matches_temp[i+1] = nxt, curr
                        if i > 0:
                            i -= 1
                        continue
                i += 1

            # Reassign rank after final reorder
            for new_rank, item_obj in enumerate(top_matches_temp, start=1):
                item_obj["rank"] = new_rank

            best_match = top_matches_temp[0] if top_matches_temp else None

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
