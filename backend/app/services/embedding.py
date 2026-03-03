from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    """
    Tiny wrapper around a Sentence-Transformers model.
    The default model is `paraphrase-MiniLM-L3-v2`, a 384-dim
    encoder that downloads in <?5?s on a fresh machine.
    You can override the model name with the environment variable
    EMBEDDING_MODEL (e.g. EMBEDDING_MODEL=all-MiniLM-L6-v2).
    """
    def __init__(self, model_name: str = None):
        # Allow an env-override for power-users
        if model_name is None:
            import os
            model_name = os.getenv("EMBEDDING_MODEL",
                                   "sentence-transformers/paraphrase-MiniLM-L3-v2")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of 384-dim vectors (plain Python list)."""
        # `encode` returns a NumPy array; we convert to list for JSON-friendliness.
        return self.model.encode(texts, show_progress_bar=False).tolist()
