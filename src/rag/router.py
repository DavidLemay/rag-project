import json
import re

from openai import OpenAI

from src.utils.config import get_openai_api_key


class QueryRouter:
    def __init__(self):
        self.client = OpenAI(api_key=get_openai_api_key())

    def route_query(self, user_query: str):
        router_system_prompt = f"""
        As a professional query router, your objective is to correctly classify user input into one of three categories based on the source most relevant for answering the query:
        1. "OPENAI_QUERY": If the user's query appears to be answerable using information from OpenAI's official documentation, tools, models, APIs, or services (e.g., GPT, ChatGPT, embeddings, moderation API, usage guidelines).
        2. "10K_DOCUMENT_QUERY": If the user's query pertains to a collection of documents from the 10k annual reports, datasets, or other structured documents, typically for research, analysis, or financial content.
        3. "INTERNET_QUERY": If the query is neither related to OpenAI nor the 10k documents specifically, or if the information might require a broader search (e.g., news, trends, tools outside these platforms), route it here.

        Your decision should be made by assessing the domain of the query.

        Always respond in this valid JSON format:
        {{
            "action": "OPENAI_QUERY" or "10K_DOCUMENT_QUERY" or "INTERNET_QUERY",
            "reason": "brief justification",
            "answer": "AT MAX 5 words answer. Leave empty if INTERNET_QUERY"
        }}

        EXAMPLES:

        - User: "How to fine-tune GPT-3?"
        Response:
        {{
            "action": "OPENAI_QUERY",
            "reason": "Fine-tuning is OpenAI-specific",
            "answer": "Use fine-tuning API"
        }}

        - User: "Where can I find the latest financial reports for the last 10 years?"
        Response:
        {{
            "action": "10K_DOCUMENT_QUERY",
            "reason": "Query related to annual reports",
            "answer": "Access through document database"
        }}

        - User: "Top leadership styles in 2024"
        Response:
        {{
            "action": "INTERNET_QUERY",
            "reason": "Needs current leadership trends",
            "answer": ""
        }}

        - User: "What's the difference between ChatGPT and Claude?"
        Response:
        {{
            "action": "INTERNET_QUERY",
            "reason": "Cross-comparison of different providers",
            "answer": ""
        }}

        Strictly follow this format for every query, and never deviate.
        User: {user_query}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": router_system_prompt}],
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
You are a query router. If the input contains multiple distinct questions, break it into sub-questions. Do not invent sub-questions, only split existing ones. Each sub-question should be a complete thought or question.
Respond ONLY in the following JSON format:
{{
    "subQuestions": ["...", "..."]
}}
Return your answer as valid JSON. Do not include anything except the JSON.
Query: "{user_query}"
"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": split_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
