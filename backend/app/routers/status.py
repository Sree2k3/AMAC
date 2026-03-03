from fastapi import APIRouter
import os
from ..services.pinecone import PineconeService

router = APIRouter()

# Pull the Pinecone configuration from the environment
RA_CFG = {
    "api_key": os.getenv("PINECONE_API_KEY"),
    "env": os.getenv("PINECONE_ENV"),
    "index_name": os.getenv("PINECONE_INDEX"),
}

@router.get("/")
async def status():
    """
    Returns a tiny health-check for the vector store.
    When the real Pinecone client is unavailable we fall back to the
    in-memory stub, so this endpoint always works.
    """
    pc = PineconeService(**RA_CFG)
    return {
        "reposIndexed": 0,                     # (stub – we don’t track repo count yet)
        "vectorsStored": pc.count(),
    }
