import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(r"E:\rag_base")

INDEX_FILE = BASE_DIR / "data" / "faiss.index"
METADATA_FILE = BASE_DIR / "data" / "faiss_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading FAISS...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

query = input("\nEnter your query: ")

query_vector = model.encode(
    query,
    convert_to_numpy=True
)

query_vector = np.array(
    [query_vector],
    dtype=np.float32
)

# Retrieve top 10 results
distances, indices = index.search(query_vector, 10)

print("\n========== TOP 10 RESULTS ==========\n")

for rank, idx in enumerate(indices[0], start=1):

    if idx == -1:
        continue

    record = metadata[idx]

    print(f"Rank       : {rank}")
    print(f"Chunk ID   : {record['chunk_id']}")
    print(f"Distance   : {distances[0][rank-1]:.4f}")
    print("Text:")

    print(record["text"][:1000])

    print("-" * 80)