from qdrant_client import QdrantClient

from src.utils.config import get_qdrant_path


class VectorRetriever:
    def __init__(self):
        self.client = QdrantClient(path=get_qdrant_path())
        # Map routing actions to Qdrant collections
        self.collections = {
            "OPENAI_QUERY": "opnai_data",
            "10K_DOCUMENT_QUERY": "10k_data",
        }

    def retrieve(self, query_vector, action, limit=3):
        if action not in self.collections:
            raise ValueError(f"Invalid action '{action}' for retrieval.")
        collection = self.collections[action]
        try:
            results = self.client.query_points(
                collection_name=collection, query=query_vector, limit=limit
            ).points
            # Extract content chunks from points
            return [point.payload["content"] for point in results]
        except Exception as e:
            raise RuntimeError(f"Qdrant retrieval failed: {e}")
