import pinecone
from typing import Dict, List

class PineconeService:
    def __init__(self, api_key, env, index_name):
        pinecone.init(api_key=api_key, environment=env)
        self.index_name = index_name

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name=index_name, dimension=384, metric="cosine")

        self.index = pinecone.Index(index_name)

    def upsert(self, vectors: List[Dict]):
        """`vectors` is a list of dicts – {id, values, metadata}."""
        self.index.upsert(vectors)

    def query(self, vector, k=5):
        return self.index.query(
            top_k=k, vector=vector, include_metadata=True
        )["matches"]

    def count(self):
        return self.index.describe_index_stats()["total_vector_count"]

