# backend/app/routers/ask.py
from fastapi import APIRouter
from ..schemas.request import AskRequest
from ..schemas.response import AskResponse
from ..services.repo_processing import process_repository
from ..services.db import get, set
from ..services.rag import RAGPipeline
import os
import uuid
import shutil

router = APIRouter()

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

CHUNK_WINDOW  = 250
CHUNK_OVERLAP = 50

def repo_key(url: str) -> str:
    return url.replace("https://github.com/", "").replace("/", "_").replace(".git", "")

@router.post("/", response_model=AskResponse)
async def ask_item(q: AskRequest):
    key = repo_key(q.repo)
    meta = get(key)
    if not meta or meta.get("needs_ingest", True):
        filtered_paths, temp_dir = process_repository(q.repo, os.getenv("GITHUB_TOKEN"))
        files = filtered_paths

        chunks = []
        for fp in files:
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

        RA.ingest(chunks)
        set(key, {"indexed_at": str(uuid.uuid1()), "vector_count": len(chunks)})
        shutil.rmtree(temp_dir, ignore_errors=True)

    answer, sources = RA.ask(q.question)
    return AskResponse(answer=answer, sources=sources)
