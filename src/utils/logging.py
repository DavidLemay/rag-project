# src/utils/logger.py
import json
import os
from datetime import datetime


class RAGLogger:
    def __init__(self, log_path="logs/rag_log.jsonl"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

    def log(self, entry: dict):
        entry["timestamp"] = datetime.utcnow().isoformat()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
