from datetime import datetime

import requests

from src.rag.embeddings import EmbeddingModel
from src.rag.generator import RAGGenerator
from src.rag.retriever import VectorRetriever
from src.rag.router import QueryRouter
from src.rag.semantic_cache import SemanticCache
from src.utils.config import get_ares_api_key
from src.utils.logging import RAGLogger


class RAGPipeline:
    def __init__(self, allow_web_search: bool = True):
        self.logger = RAGLogger()
        self.embedder = EmbeddingModel()
        self.retriever = VectorRetriever()
        self.router = QueryRouter()
        self.generator = RAGGenerator()
        self.cache = SemanticCache(self.embedder)
        self.allow_web_search = allow_web_search
        self.routes = {
            "OPENAI_QUERY": self.retrieve_and_respond,
            "10K_DOCUMENT_QUERY": self.retrieve_and_respond,
            "INTERNET_QUERY": (
                self.get_internet_content
                if allow_web_search
                else self.skip_internet_search
            ),
        }

    def skip_internet_search(self, user_query, action):
        return "[Web Search Disabled] The system was not allowed to perform an internet search."

    def retrieve_and_respond(self, user_query, action):
        # Get context
        vector = self.embedder.get_embedding(user_query)
        context = self.retriever.retrieve(vector, action)
        if not context:
            return "No relevant content found."

        # Check cache
        cached_answer = self.cache.get(user_query, context)
        if cached_answer:
            return f"[Semantic Cache HIT]\n{cached_answer}"

        # Not cached: run generator and cache result
        answer = self.generator.generate(user_query, context)
        self.cache.add(user_query, answer, context)
        return answer

    def get_internet_content(self, user_query, action):
        # Use empty list as context for internet queries
        context_chunks = []
        cached_answer = self.cache.get(user_query, context_chunks)
        if cached_answer:
            return f"[Semantic Cache HIT]\n{cached_answer}"

        url = "https://api-ares.traversaal.ai/live/predict"
        payload = {"query": [user_query]}
        headers = {"x-api-key": get_ares_api_key(), "content-type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            answer = (
                response.json()
                .get("data", {})
                .get("response_text", "No response received.")
            )
            self.cache.add(user_query, answer, context_chunks)
            return answer
        except Exception as e:
            return f"Internet query failed: {e}"

    def run(self, user_query: str):
        subqueries = self.router.split_subqueries(user_query)
        results = {}
        for subq in subqueries["subQuestions"]:
            route_info = self.router.route_query(subq)
            action = route_info.get("action")
            handler = self.routes.get(action)
            log_entry = {
                "user_query": user_query,
                "subquery": subq,
                "route_action": action,
                "route_reason": route_info.get("reason", ""),
            }
            if handler:
                start = datetime.utcnow()
                answer = handler(subq, action)
                duration = (datetime.utcnow() - start).total_seconds()
                log_entry.update(
                    {
                        "cache_hit": answer.startswith("[Semantic Cache HIT]"),
                        "latency_seconds": duration,
                        "answer_preview": answer[:200],
                    }
                )
                results[subq] = handler(subq, action)
            else:
                results[subq] = f"Unsupported action: {action}"
                log_entry["error"] = "Unsupported action"
            self.logger.log(log_entry)
        return results
