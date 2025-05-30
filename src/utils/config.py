import os

from dotenv import load_dotenv

load_dotenv()


def get_qdrant_path():
    # Default for local testing; update as needed
    return os.getenv(
        "QDRANT_PATH", "multi-agent-course/Module_1/Agentic_RAG/qdrant_data"
    )


def get_openai_api_key():
    return os.getenv("OPENAI_API_KEY")


def get_ares_api_key():
    return os.getenv("ARES_API_KEY")
