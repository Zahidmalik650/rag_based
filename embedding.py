import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path(r"E:\rag_base\data\chunks.json")
OUTPUT_FILE = Path(r"E:\rag_base\data\embeddings.json")

MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print("Loading chunks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    results = []

    for chunk, vector in zip(chunks, vectors):
        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "embedding": vector.tolist()
        })

    print("Saving embeddings...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print("\nSUCCESS!")
    print("Total embeddings:", len(results))
    print("Dimensions:", len(vectors[0]))
    print("Saved file:", OUTPUT_FILE)

if __name__ == "__main__":
    main()