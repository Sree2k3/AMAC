from typing import List, Dict
from .embedding import EmbeddingService
from .pinecone import PineconeService
from .llm import LLMService

class RAGPipeline:
    def __init__(self, pinecone_cfg: dict, llm_cfg: dict):
        # initialise the three building blocks
        self.embedd   = EmbeddingService()
        self.pinecone = PineconeService(**pinecone_cfg)
        self.llm      = LLMService(**llm_cfg)

    # --------------------------------------------------------------
    #   Ingestion – turn code chunks into vectors and upsert them
    # --------------------------------------------------------------
    def ingest(self, chunks: List[Dict]):
        vectors = []
        for ch in chunks:
            # One‑sentence embedding → 384‑dim vector
            vec = self.embedd.embed([ch["text"]])[0]
            vectors.append({
                "id": ch["id"],
                "values": vec,
                "metadata": ch["metadata"],
            })
        self.pinecone.upsert(vectors)

    # --------------------------------------------------------------
    #   Query + LLM generation
    # --------------------------------------------------------------
    def ask(self, question: str, k: int = 5) -> tuple[str, List[Dict]]:
        # 1️⃣  Embed the user question
        query_vec = self.embedd.embed([question])[0]

        # 2️⃣  Retrieve top‑k chunks from Pinecone (or the in‑memory stub)
        results = self.pinecone.query(query_vec, k)

        # 3️⃣  Build a context string for the prompt
        context = "\n\n".join(
            [
                f"{res['metadata']['file']}:{res['metadata']['startLine']}–{res['metadata']['endLine']}"
                f"\n{res['metadata']['text']}"
                for res in results
            ]
        )

        system_prompt = "You are a helpful assistant that references code snippets."
        final_prompt = f"{system_prompt}\n\n{context}\n\nQuestion: {question}"

        # 4️⃣  Call the LLM (Ollama).  If Ollama is not reachable we raise a clean error.
        try:
            answer = self.llm.chat(final_prompt)
        except Exception as exc:
            raise RuntimeError(
                "LLM invocation failed – make sure Ollama is running (ollama serve). Details: " + str(exc)
            )

        # 5️⃣  Prepare source metadata for the API response
        sources = [
            {
                "file": res["metadata"]["file"],
                "startLine": res["metadata"]["startLine"],
                "endLine": res["metadata"]["endLine"],
            }
            for res in results
        ]

        return answer, sources
