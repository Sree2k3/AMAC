import os, pathlib, json
from fastapi import APIRouter
from ..services.pinecone import PineconeService

router = APIRouter()

RA_CFG = dict(
    api_key=os.getenv("PINECONE_API_KEY"),
    env=os.getenv("PINECONE_ENV"),
    index_name=os.getenv("PINECONE_INDEX"),
)

@router.get("/")
async def status():
    # Path to the tiny local cache file
    cache_file = pathlib.Path(__file__).resolve().parent.parent / "services" / "cache.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8") or "{}")
        repos_indexed = len(data)
    else:
        repos_indexed = 0

    # Pinecone vector count
    pc = PineconeService(**RA_CFG)
    vectors_stored = pc.count()

    return {"reposIndexed": repos_indexed, "vectorsStored": vectors_stored}

