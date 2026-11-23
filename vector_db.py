from pathlib import Path
import json
import chromadb

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(r"E:\rag_base")

EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.json"

DATABASE_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "modern_database"


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

print("Loading embeddings...")

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

print(f"Total embeddings loaded: {len(data)}")


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ==========================================
# STORE DATA
# ==========================================

print("Storing embeddings in ChromaDB...")

batch_size = 500

for start in range(0, len(data), batch_size):

    batch = data[start:start + batch_size]

    ids = [
        str(item["chunk_id"])
        for item in batch
    ]

    embeddings = [
        item["embedding"]
        for item in batch
    ]

    documents = [
        item["text"]
        for item in batch
    ]

    metadatas = [
        item.get("metadata", {})
        for item in batch
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Stored {min(start + batch_size, len(data))}/{len(data)}")


# ==========================================
# VERIFY DATABASE
# ==========================================

print("\n===================================")
print("DATABASE VERIFICATION")
print("===================================")

print("Collection:", COLLECTION_NAME)
print("Total records:", collection.count())
print("Database location:", DATABASE_DIR)

print("\nSUCCESS! Embeddings stored in ChromaDB.")