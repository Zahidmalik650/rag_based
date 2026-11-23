from pathlib import Path
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(r"E:\rag_base")

INDEX_FILE = BASE_DIR / "data" / "faiss.index"

METADATA_FILE = BASE_DIR / "data" / "faiss_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# ==========================================
# LOAD FAISS INDEX
# ==========================================

print("Loading FAISS index...")

index = faiss.read_index(str(INDEX_FILE))

print("Index loaded successfully!")
print("Total vectors:", index.ntotal)


# ==========================================
# LOAD METADATA
# ==========================================

print("Loading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Metadata loaded:", len(metadata), "records")


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully!")


# ==========================================
# SEARCH FUNCTION
# ==========================================

def search_database(query, top_k=TOP_K):

    # Generate query embedding
    query_vector = model.encode(
        query,
        convert_to_numpy=True
    )

    # Convert to float32
    query_vector = np.array(
        [query_vector],
        dtype=np.float32
    )

    # Search FAISS
    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        record = metadata[idx]

        results.append({
            "rank": rank + 1,
            "chunk_id": record["chunk_id"],
            "distance": float(distances[0][rank]),
            "text": record["text"],
            "metadata": record.get("metadata", {})
        })

    return results


# ==========================================
# INTERACTIVE SEARCH
# ==========================================

print("\n===================================")
print("DATABASE SEMANTIC SEARCH")
print("===================================")

print("Type your question.")
print("Type 'exit' to quit.")

while True:

    query = input("\nYour Question: ").strip()

    if query.lower() == "exit":
        print("Search system closed.")
        break

    if not query:
        print("Please enter a question.")
        continue

    print("\nSearching...")

    results = search_database(query)

    print("\n===================================")
    print("SEARCH RESULTS")
    print("===================================")

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Distance: {result['distance']:.4f}")

        print("\nText:")
        print(result["text"])

        print("\n-----------------------------------")