
from pathlib import Path
import json
from pypdf import PdfReader

# CONFIGURATION

PDF_FILE = Path(
    r"E:\rag_base\data\modern+database+10th+edition.pdf"
)

OUTPUT_FILE = Path(
    r"E:\rag_base\data\chunks.json"
)

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200


# CHUNKING

def create_chunks():

    if not PDF_FILE.exists():
        print("PDF file not found:", PDF_FILE)
        return

    pdf = PdfReader(str(PDF_FILE))

    full_text = ""

    for page in pdf.pages:
        full_text += (page.extract_text() or "") + "\n"

    full_text = full_text.strip()

    if not full_text:
        print("No text found in PDF.")
        return

    chunks = []
    start = 0
    chunk_id = 1

    while start < len(full_text):

        end = min(start + CHUNK_SIZE, len(full_text))

        text = full_text[start:end].strip()

        if text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "source": PDF_FILE.name,
                    "character_count": len(text)
                }
            })

            chunk_id += 1

        if end == len(full_text):
            break

        start = end - CHUNK_OVERLAP

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print("Chunking completed!")
    print("Total chunks:", len(chunks))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    create_chunks()
