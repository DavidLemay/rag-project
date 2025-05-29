import json
import os

import faiss
import numpy as np


class SemanticCache:
    def __init__(
        self, embedding_model, cache_path="semantic_cache.json", distance_threshold=0.1
    ):
        """
        embedding_model: EmbeddingModel instance from the pipeline
        cache_path: file to save the cache
        distance_threshold: max cosine distance for cache hit (lower = more strict)
        """
        self.embedding_model = embedding_model
        self.cache_path = cache_path
        self.distance_threshold = distance_threshold

        self.questions = []
        self.answers = []
        self.contexts = []  # The context chunks used for this answer
        self.embeddings = None  # np.ndarray [n_questions, dim]
        self.index = None
        self.load_cache()

    def load_cache(self):
        if not os.path.exists(self.cache_path):
            # Nothing to load yet
            self.questions, self.answers, self.contexts = [], [], []
            self.embeddings = None
            self.index = None
            return

        with open(self.cache_path, "r") as f:
            data = json.load(f)
            self.questions = data["questions"]
            self.answers = data["answers"]
            self.contexts = data["contexts"]
            self.embeddings = (
                np.array(data["embeddings"], dtype=np.float32)
                if data["embeddings"]
                else None
            )

        if self.embeddings is not None and len(self.embeddings) > 0:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = None

    def save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(
                {
                    "questions": self.questions,
                    "answers": self.answers,
                    "contexts": self.contexts,
                    "embeddings": (
                        self.embeddings.tolist() if self.embeddings is not None else []
                    ),
                },
                f,
            )

    def _normalize(self, vec):
        # Normalize to unit vector for cosine
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def get(self, question, current_context_chunks):
        """Check if a similar question is in the cache and, if so, return its answer. Else return None."""
        if self.index is None or self.embeddings is None or len(self.questions) == 0:
            return None

        query_emb = self.embedding_model.get_embedding(question)
        query_emb = self._normalize(query_emb.astype(np.float32)).reshape(1, -1)

        # Search for top-1 most similar cached question
        D, I = self.index.search(query_emb, 1)
        top_idx = I[0][0]
        similarity = D[0][0]  # For cosine, FAISS gives dot product (max=1, min=-1)
        # print(f"Similarity for '{question}': {similarity}")
        # print(f"DEBUG: Similarity for '{question}' and '{self.questions[top_idx]}' = {similarity:.4f}")
        # print(f"DEBUG: Context match? {self.contexts[top_idx] == current_context_chunks}")
        if similarity >= 1 - self.distance_threshold:
            # Optionally, also check if context is still valid (chunks used are the same)
            if self.contexts[top_idx] == current_context_chunks:
                return self.answers[top_idx]
            # If context differs, do not use cache

        return None

    def add(self, question, answer, context_chunks):
        """Add a new Q/A/context to the cache."""
        q_emb = self.embedding_model.get_embedding(question)
        q_emb = self._normalize(q_emb.astype(np.float32)).reshape(1, -1)

        # Add to arrays
        self.questions.append(question)
        self.answers.append(answer)
        self.contexts.append(context_chunks)
        if self.embeddings is None:
            self.embeddings = q_emb
            # Init FAISS index
            dim = q_emb.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, q_emb])
            self.index.add(q_emb)
        self.save_cache()

    def invalidate_for_context(self, context_chunks):
        """Remove any cache entries whose context matches the given (e.g., if context has changed)."""
        to_keep = [i for i, ctx in enumerate(self.contexts) if ctx != context_chunks]
        self.questions = [self.questions[i] for i in to_keep]
        self.answers = [self.answers[i] for i in to_keep]
        self.contexts = [self.contexts[i] for i in to_keep]
        self.embeddings = (
            np.array([self.embeddings[i] for i in to_keep], dtype=np.float32)
            if to_keep
            else None
        )
        if self.embeddings is not None and len(self.embeddings) > 0:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = None
        self.save_cache()
