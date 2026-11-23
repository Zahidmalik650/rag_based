from pathlib import Path
import json
import numpy as np
import faiss

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(r"E:\rag_base")

EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.json"

INDEX_FILE = BASE_DIR / "data" / "faiss.index"

METADATA_FILE = BASE_DIR / "data" / "faiss_metadata.json"


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

print("Loading embeddings...")

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

print(f"Total records: {len(data)}")


# ==========================================
# PREPARE VECTORS
# ==========================================

print("Preparing vectors...")

vectors = np.array(
    [item["embedding"] for item in data],
    dtype=np.float32
)

dimension = vectors.shape[1]

print(f"Vector shape: {vectors.shape}")
print(f"Dimensions: {dimension}")


# ==========================================
# CREATE FAISS INDEX
# ==========================================

print("Creating FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(vectors)

print(f"Vectors added: {index.ntotal}")


# ==========================================
# SAVE INDEX
# ==========================================

print("Saving FAISS index...")

faiss.write_index(index, str(INDEX_FILE))


# ==========================================
# SAVE TEXT AND METADATA
# ==========================================

print("Saving chunk metadata...")

metadata_records = []

for item in data:

    metadata_records.append({
        "chunk_id": item["chunk_id"],
        "text": item["text"],
        "metadata": item.get("metadata", {})
    })

with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(
        metadata_records,
        file,
        ensure_ascii=False,
        indent=2
    )


# ==========================================
# VERIFICATION
# ==========================================

print("\n===================================")
print("FAISS INDEX VERIFICATION")
print("===================================")

print("Total vectors:", index.ntotal)
print("Vector dimensions:", dimension)
print("Index type:", type(index).__name__)
print("Index file:", INDEX_FILE)
print("Metadata file:", METADATA_FILE)

print("\nSUCCESS! FAISS index created.")