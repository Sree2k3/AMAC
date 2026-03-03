import os
from typing import List, Dict

# New Pinecone SDK (v5+)
try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_AVAILABLE = True
except Exception:
    # If the import fails (e.g. older library) we will fallback to a stub
    _PINECONE_AVAILABLE = False


class PineconeService:
    """
    Wrapper that works with the **new** Pinecone client.
    If no API key is supplied (or the library cannot be imported) the
    class automatically switches to an in-memory stub so the rest of the
    system keeps functioning for local testing.
    """

    def __init__(self, api_key: str = None, env: str = "us-west1-gcp", index_name: str = "amac-index"):
        self.index_name = index_name
        self._use_stub = False

        # ------------------------------------------------------------------
        # 1??  Decide whether we can talk to the real Pinecone service
        # ------------------------------------------------------------------
        if not _PINECONE_AVAILABLE or not api_key:
            # --------------------------------------------------------------
            #   FALLBACK: simple in-memory store (dict)
            # --------------------------------------------------------------
            self._use_stub = True
            self._vectors: Dict[str, tuple[list[float], dict]] = {}   # id -> (values, metadata)
            return

        # ------------------------------------------------------------------
        # 2??  Initialise the real Pinecone client
        # ------------------------------------------------------------------
        self.pc = Pinecone(api_key=api_key)

        # `env` is expected as "<region>-<cloud>", e.g. "us-west1-gcp"
        # If the format is different we fall back to a safe default.
        if "-" in env:
            region, cloud = env.split("-", 1)
        else:
            region, cloud = "us-west1", "aws"

        # --------------------------------------------------------------
        #   Create the index if it does not exist yet
        # --------------------------------------------------------------
        existing = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing:
            spec = ServerlessSpec(cloud=cloud, region=region)
            self.pc.create_index(
                name=index_name,
                dimension=384,               # MiniLM-L6-v2 dimension
                metric="cosine",
                spec=spec,
            )

        # --------------------------------------------------------------
        #   Grab a handle to the index
        # --------------------------------------------------------------
        self.index = self.pc.Index(index_name)

    # ------------------------------------------------------------------
    #   Public methods – they behave the same whether we are using the
    #   real Pinecone service or the in-memory stub.
    # ------------------------------------------------------------------
    def upsert(self, vectors: List[Dict]):
        if self._use_stub:
            for vec in vectors:
                self._vectors[vec["id"]] = (vec["values"], vec["metadata"])
            return
        self.index.upsert(vectors)

    def query(self, vector, k: int = 5) -> List[dict]:
        if self._use_stub:
            # Very naive fallback – just return the first *k* vectors
            # (no similarity ranking). This is enough for local debugging.
            results = []
            for idx, (values, meta) in enumerate(self._vectors.values()):
                if idx >= k:
                    break
                results.append({"metadata": meta, "id": idx})
            return results

        return self.index.query(
            top_k=k, vector=vector, include_metadata=True
        )["matches"]

    def count(self) -> int:
        if self._use_stub:
            return len(self._vectors)
        return self.index.describe_index_stats()["total_vector_count"]
