from openai import OpenAI

from src.utils.config import get_openai_api_key


class RAGGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=get_openai_api_key())

    def generate(self, user_query: str, context: list):
        rag_prompt = f"""
        Based on the given context, answer the user query: {user_query}\nContext:\n{context}
        and employ references to the ID of articles provided [ID], ensuring their relevance to the query.
        The referencing should always be in the format of [1][2]... etc.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": rag_prompt}]
        )
        return response.choices[0].message.content
