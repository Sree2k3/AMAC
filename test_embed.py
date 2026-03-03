# test_embed.py
from backend.app.services.embedding import EmbeddingService

svc = EmbeddingService()
vec = svc.embed(["Hello world"])[0]

print("Vector length:", len(vec))
print("First 5 values:", vec[:5])
