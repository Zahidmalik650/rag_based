
import time
import json
import os
import re
from pathlib import Path

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(r"E:\rag_base")

INDEX_FILE = BASE_DIR / "data" / "faiss.index"
METADATA_FILE = BASE_DIR / "data" / "faiss_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

TOP_K = 10


# ==========================================
# LOAD FAISS
# ==========================================

print("Loading FAISS index...")

index = faiss.read_index(str(INDEX_FILE))

print("FAISS loaded:", index.ntotal)


# ==========================================
# LOAD METADATA
# ==========================================

print("Loading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Metadata loaded:", len(metadata))


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")


# ==========================================
# CONNECT GEMINI
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

client = None

if api_key:

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=15000)
    )

    print("Gemini client initialized!")

else:
    print("WARNING: GEMINI_API_KEY not found. Retrieval mode is available.")


# ==========================================
# HYBRID SEARCH
# ==========================================

def search_database(query, top_k=TOP_K):

    # STEP 1: Semantic search

    query_vector = model.encode(
        query,
        convert_to_numpy=True
    )

    query_vector = np.asarray(
        [query_vector],
        dtype=np.float32
    )

    actual_k = min(max(top_k * 5, 50), index.ntotal)

    distances, indices = index.search(
        query_vector,
        actual_k
    )

    # STEP 2: Extract query keywords

    stop_words = {
        "what", "is", "are", "the", "a", "an",
        "of", "in", "on", "to", "for", "and",
        "explain", "define", "describe", "how"
    }

    query_words = set(
        re.findall(r"\b\w+\b", query.lower())
    ) - stop_words

    results = []

    # STEP 3: Keyword scoring

    for semantic_rank, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        record = metadata[idx]

        text_content = record.get("text", "")

        document_words = set(
            re.findall(r"\b\w+\b", text_content.lower())
        )

        if query_words:

            matched_words = query_words.intersection(
                document_words
            )

            keyword_score = (
                len(matched_words) / len(query_words)
            )

        else:
            keyword_score = 0.0

        # STEP 4: Combined ranking

        semantic_score = 1 / (semantic_rank + 1)

        final_score = (
            0.65 * semantic_score
            + 0.35 * keyword_score
        )

        results.append({

            "chunk_id": record.get("chunk_id", idx),

            "distance": float(
                distances[0][semantic_rank]
            ),

            "keyword_score": keyword_score,

            "final_score": final_score,

            "text": text_content,

            "metadata": record.get("metadata", {})

        })

    # STEP 5: Sort results

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    # STEP 6: Final ranking

    results = results[:top_k]

    for rank, result in enumerate(results):

        result["rank"] = rank + 1

    return results


# ==========================================
# SOURCE INFORMATION
# ==========================================

def get_source_label(source):

    info = source.get("metadata", {})

    page = (
        info.get("page")
        or info.get("page_number")
        or info.get("page_num")
    )

    if page is not None:

        return (
            f"Chunk {source['chunk_id']} | Page {page}"
        )

    return f"Chunk {source['chunk_id']}"


# ==========================================
# GEMINI GENERATION
# ==========================================

def ask_gemini(prompt):

    if client is None:

        return None, "GEMINI_API_KEY is not configured."

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response.text:

            return None, "Gemini returned an empty response."

        return response.text, None

    except Exception as error:

        return None, str(error)


# ==========================================
# RAG ANSWER
# ==========================================

def generate_answer(query):

    start_time = time.time()

    print("\nSearching book...")

    search_start = time.time()

    results = search_database(query)

    search_time = time.time() - search_start

    if not results:

        return "No relevant information found.", []

    # PREPARE CONTEXT

    context = ""

    for result in results:

        context += (
            f"\n\n[{get_source_label(result)}]\n"
            f"{result['text']}"
        )

    # PROMPT

    prompt = f"""
You are an educational book assistant.

Answer the question using ONLY the provided
book context.

Rules:
1. Explain in simple and clear language.
2. Do not invent information.
3. If context is insufficient, say so.
4. Mention source chunk IDs where relevant.
5. Never invent page numbers.

BOOK CONTEXT:

{context}

USER QUESTION:

{query}
"""

    # GENERATE ANSWER

    print("Generating answer with Gemini...")

    api_start = time.time()

    answer, api_error = ask_gemini(prompt)

    api_time = time.time() - api_start

    total_time = time.time() - start_time

    # PERFORMANCE REPORT

    print("\n===================================")
    print("PERFORMANCE REPORT")
    print("===================================")

    print(f"Search Time : {search_time:.2f} seconds")
    print(f"Gemini Time : {api_time:.2f} seconds")
    print(f"Total Time  : {total_time:.2f} seconds")

    if api_error:

        answer = (
            "Gemini API Error:\n"
            + api_error
            + "\n\nRetrieved sources are shown below."
        )

    return answer, results


# ==========================================
# INTERACTIVE CHAT
# ==========================================

def run_chat():

    print("\n===================================")
    print("DATABASE BOOK RAG SYSTEM")
    print("===================================")

    print("Ask questions from your database book.")
    print("Type 'exit' to close.")

    while True:

        query = input("\nYour Question: ").strip()

        if query.lower() == "exit":

            print("RAG system closed.")
            break

        if not query:

            print("Please enter a question.")
            continue

        try:

            answer, sources = generate_answer(query)

            print("\n===================================")
            print("AI ANSWER")
            print("===================================")

            print(answer)

            print("\n===================================")
            print("RETRIEVED SOURCES")
            print("===================================")

            for source in sources:

                print(
                    f"\n{get_source_label(source)} | "
                    f"Distance: {source['distance']:.4f} | "
                    f"Keyword Score: {source['keyword_score']:.2f}"
                )

                print(source["text"][:400])

                print("-" * 50)

        except Exception as error:

            print("\nError:", error)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    run_chat()
