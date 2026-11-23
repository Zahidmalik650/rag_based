# RAG-Based Database Book Assistant

## Project Overview
A Retrieval-Augmented Generation (RAG) application that answers questions using content extracted from *Modern Database Management, 10th Edition*.

## Technologies Used
- Python
- Streamlit
- PyPDF
- LangChain Text Splitter
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS Vector Database
- Google Gemini API
- JSON

## Project Workflow
PDF Extraction → Text Chunking → Embeddings → FAISS Indexing → Semantic Search → Context Retrieval → Gemini Response → Streamlit Interface

## Main Features
- PDF-based question answering
- Semantic search
- Retrieval-only mode
- Gemini AI mode
- Source chunk references
- Search performance timing
- Chat history during the active session

## Run the Project

Activate virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install streamlit pypdf langchain-text-splitters sentence-transformers faiss-cpu google-genai numpy
```

Start application:

```powershell
python -m streamlit run app.py
```

Open in browser:

```text
http://localhost:8501
```

## Note
Gemini AI mode requires a valid API key and available API quota. Retrieval-only mode works without Gemini API access.