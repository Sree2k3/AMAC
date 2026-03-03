# backend/app/services/embedding.py
import os
import logging
import time
from typing import List
import numpy as np

# ----------------------------------------------------------------------
class EmbeddingService:
    """
    Loads MiniLM‑L6‑384 from Hugging‑Face (cached locally).
    If the model cannot be loaded (no internet, DNS block, etc.) it falls
    back to a deterministic random vector generator so the rest of the
    pipeline stays functional.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.log = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.dim = 384                                 # MiniLM‑L6‑384 size

        # --------------------------------------------------------------
        # Where should SentenceTransformer look for cached files?
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        load_kwargs = {"cache_folder": cache_dir} if cache_dir else {}

        # --------------------------------------------------------------
        # Try a few times – the first run often hits a transient network glitch.
        for attempt in range(5):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, **load_kwargs)
                self.log.info(f"✅ Loaded embedding model '{model_name}'")
                break
            except Exception as exc:   # noqa: BLE001
                self.log.warning(
                    f"Attempt {attempt+1}/5 – could not load '{model_name}': {exc}"
                )
                if attempt < 4:
                    time.sleep(2 ** attempt)   # 1 s, 2 s, 4 s, 8 s

        # --------------------------------------------------------------
        if self.model is None:
            self.log.error(
                "❌ Failed to load MiniLM after several attempts. "
                "Falling back to deterministic random vectors (dim=384)."
            )
            # deterministic RNG – same text always yields same vector
            self._fallback_rng = np.random.default_rng(0xC0FFEE)

    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Returns a list of 384‑dim vectors (one per input string).
        """
        if self.model is not None:
            # SentenceTransformer returns a NumPy array → convert to list
            return self.model.encode(texts, show_progress_bar=False).tolist()

        # ---------- Deterministic random fallback ----------
        vectors = []
        for txt in texts:
            seed = abs(hash(txt)) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(self.dim).tolist())
        return vectors
