from fastapi import APIRouter
from ..schemas.request import AskRequest
from ..schemas.response import AskResponse
from ..services.git import clone_repo, list_code_files
from ..services.db import get, set
from ..services.rag import RAGPipeline
from pathlib import Path
import os
import uuid
import shutil

router = APIRouter()

# Application configuration pulled from .env
RA_CFG = {
    "api_key": os.getenv("PINECONE_API_KEY"),
    "env": os.getenv("PINECONE_ENV"),
    "index_name": os.getenv("PINECONE_INDEX"),
}
LLM_CFG = {
    "ollama_url": os.getenv("OLLAMA_URL"),
    "model": os.getenv("OLLAMA_MODEL"),
}
RA = RAGPipeline(RA_CFG, LLM_CFG)

# Chunking constants – tuned for the PRD
CHUNK_WINDOW  = 250   # tokens per chunk
CHUNK_OVERLAP = 50    # overlapping tokens

def repo_key(url: str) -> str:
    """Canonical key used in the tiny KV store."""
    return url.replace("https://github.com/", "").replace("/", "_").replace(".git", "")

@router.post("/", response_model=AskResponse)
async def ask_item(q: AskRequest):
    """POST /ask – index a repo if needed, then answer a question."""
    key = repo_key(q.repo)

    # 1️⃣ Pull the repo metadata – does the repo exist in the cache?
    meta = get(key)
    if not meta or meta.get("needs_ingest", True):
        # 1. Clone the repository
        repo_path = clone_repo(q.repo, os.getenv("GITHUB_TOKEN"))

        # 2. List all supported file types
        code_files = list_code_files(repo_path)

        # 3. Chunk each file (naïve line‑based chunking)
        chunks = []
        for fp in code_files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            start = 0
            while start < len(lines):
                end = min(start + CHUNK_WINDOW, len(lines))
                chunk_text = "".join(lines[start:end])
                chunk_meta = {
                    "file": str(fp),
                    "startLine": start + 1,
                    "endLine": end,
                    "text": chunk_text,
                }
                chunk_id = str(uuid.uuid4())
                chunks.append({"id": chunk_id, "metadata": chunk_meta, "text": chunk_text})
                start = end - CHUNK_OVERLAP

        # 4. Ingest to Pinecone
        RA.ingest(chunks)

        # 5. Record minimal repo metadata (last indexed, vector count)
        set(key, {"indexed_at": str(uuid.uuid1()), "vector_count": len(chunks)})

        # 6. Cleanup the cloned repo
        shutil.rmtree(repo_path)

    # 7️⃣ Finally, answer the user question via the RAG pipeline
    answer, sources = RA.ask(q.question)
    return AskResponse(answer=answer, sources=sources)

