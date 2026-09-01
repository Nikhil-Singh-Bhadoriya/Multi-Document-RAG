# MultiDocChat

A conversational **Multi-Document RAG application** built with **FastAPI, LangChain, FAISS, Hugging Face, Groq, and Google Gemini**.

## Features

- Upload and chat with multiple `.pdf`, `.docx`, and `.txt` files.
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Session-based FAISS vector indexes.
- MMR retrieval for relevant and diverse results.
- Conversational question contextualization.
- Groq and Google Gemini LLM support.
- Pydantic validation, logging, and custom exception handling.

## Configuration

```yaml
embedding_model:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

retriever:
  top_k: 10
  search_type: "mmr"
  fetch_k: 20
  lambda_mult: 0.5

llm:
  groq:
    provider: "groq"
    model_name: "openai/gpt-oss-20b"
    temperature: 0
    max_output_tokens: 2048

  google:
    provider: "google"
    model_name: "gemini-3.6-flash"
    temperature: 0
    max_output_tokens: 2048
```

## Run Locally

```bash
git clone https://github.com/<username>/MultiDocChat.git
cd MultiDocChat
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `.env`:
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

Start the application:
```bash
uvicorn main:app --reload
```

Open **http://localhost:8000**

Swagger API: **http://localhost:8000/docs**

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/upload` | Upload and index documents |
| POST | `/chat` | Ask questions using `session_id` |

## Storage

```text
data/<session_id>/           # Uploaded documents
faiss_index/<session_id>/    # FAISS index
```

Chat history is stored in memory and resets when the server restarts.

## GitHub

Do not commit:

```text
.env
.venv/
data/
faiss_index/
logs/
__pycache__/
```
