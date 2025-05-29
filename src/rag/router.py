import json
import re

from openai import OpenAI

from src.utils.config import get_openai_api_key


class QueryRouter:
    def __init__(self):
        self.client = OpenAI(api_key=get_openai_api_key())

    def route_query(self, user_query: str):
        router_prompt = f"""
        As a professional query router, your objective is to correctly classify user input ...
        ... (same prompt as in your example, shortened here for brevity)
        User: {user_query}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "system", "content": router_prompt}]
            )
            resp_text = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", resp_text, re.DOTALL)
            json_text = json_match.group()
            return json.loads(json_text)
        except Exception as err:
            return {
                "action": "INTERNET_QUERY",
                "reason": f"Router error: {err}",
                "answer": "",
            }

    def split_subqueries(self, user_query: str):
        split_prompt = f"""
        You are a query router. If the input contains multiple distinct questions, break it into sub-questions...
        Query: "{user_query}"
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": split_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
