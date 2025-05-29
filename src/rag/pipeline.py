import requests

from src.rag.embeddings import EmbeddingModel
from src.rag.generator import RAGGenerator
from src.rag.retriever import VectorRetriever
from src.rag.router import QueryRouter
from src.utils.config import get_ares_api_key


class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.retriever = VectorRetriever()
        self.router = QueryRouter()
        self.generator = RAGGenerator()
        self.routes = {
            "OPENAI_QUERY": self.retrieve_and_respond,
            "10K_DOCUMENT_QUERY": self.retrieve_and_respond,
            "INTERNET_QUERY": self.get_internet_content,
        }

    def retrieve_and_respond(self, user_query, action):
        try:
            vector = self.embedder.get_embedding(user_query)
            context = self.retriever.retrieve(vector, action)
            if not context:
                return "No relevant content found."
            return self.generator.generate(user_query, context)
        except Exception as e:
            return f"Error in retrieval/generation: {e}"

    def get_internet_content(self, user_query, action):
        url = "https://api-ares.traversaal.ai/live/predict"
        payload = {"query": [user_query]}
        headers = {"x-api-key": get_ares_api_key(), "content-type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return (
                response.json()
                .get("data", {})
                .get("response_text", "No response received.")
            )
        except Exception as e:
            return f"Internet query failed: {e}"

    def run(self, user_query: str):
        subqueries = self.router.split_subqueries(user_query)
        results = {}
        for subq in subqueries["subQuestions"]:
            route_info = self.router.route_query(subq)
            action = route_info.get("action")
            handler = self.routes.get(action)
            if handler:
                results[subq] = handler(subq, action)
            else:
                results[subq] = f"Unsupported action: {action}"
        return results
