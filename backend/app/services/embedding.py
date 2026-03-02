from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v6"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]):
        """Return a list of 384-dim vectors (np.ndarray) for each string."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

