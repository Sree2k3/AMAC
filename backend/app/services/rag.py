from typing import List, Dict
from .embedding import EmbeddingService
from .pinecone  import PineconeService
from .llm       import LLMService

class RAGPipeline:
    def __init__(self, pinecone_cfg: dict, llm_cfg: dict):
        self.embedd   = EmbeddingService()
        self.pinecone = PineconeService(**pinecone_cfg)
        self.llm      = LLMService(**llm_cfg)

    # ------------------------------------------------------------------
    def ingest(self, chunks: List[Dict]):
        """Each `chunk` must contain {"id", "text", "metadata"}."""
        vectors = []
        for ch in chunks:
            vec = self.embedd.embed([ch["text"]])[0]
            vectors.append(
                {"id": ch["id"], "values": vec, "metadata": ch["metadata"]}
            )
        self.pinecone.upsert(vectors)

    # ------------------------------------------------------------------
    def ask(self, question: str, k: int = 5) -> tuple[str, List[dict]]:
        """Run the RAG flow and return (answer, source docs)."""
        query_vec = self.embedd.embed([question])[0]
        results   = self.pinecone.query(query_vec, k)

        # Build a tiny context string
        context = "\\n\\n".join(
            [
                f"{res['metadata']['file']}:{res['metadata']['startLine']}:{res['metadata']['endLine']}\\n{res['metadata']['text']}"
                for res in results
            ]
        )

        system_prompt = "You are a helpful assistant that references code snippets."
        final_prompt  = f"{system_prompt}\\n\\n{context}\\n\\nQuestion: {question}"
        answer = self.llm.chat(final_prompt)

        sources = [
            {
                "file":    res["metadata"]["file"],
                "startLine": res["metadata"]["startLine"],
                "endLine":   res["metadata"]["endLine"],
            }
            for res in results
        ]
        return answer, sources

