import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].detach().cpu().numpy()
