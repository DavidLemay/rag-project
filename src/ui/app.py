# ui/app.py
import streamlit as st

from src.rag.pipeline import RAGPipeline
from src.utils.config import set_ares_api_key, set_openai_key

st.set_page_config(page_title="RAG QA System", layout="wide")

st.title("📚 RAG Question Answering System")
st.markdown(
    "Ask your question and get answers using vector search and optional internet retrieval."
)

# --- Optional API key inputs ---
with st.expander("🔐 API Keys (Optional)", expanded=False):
    openai_key = st.text_input("OpenAI API Key", type="password")
    ares_key = st.text_input("ARES API Key", type="password")
    if openai_key:
        set_openai_key(openai_key)
    if ares_key:
        set_ares_api_key(ares_key)

# --- Input Section ---
query = st.text_area("📝 Enter your question:", height=150)
allow_web_search = st.checkbox("Allow Web Search", value=True)

if st.button("🔍 Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your query..."):
            pipeline = RAGPipeline(allow_web_search=allow_web_search)
            results = pipeline.run(query)

        for subq, answer in results.items():
            st.markdown(f"**🧩 Sub-question:** `{subq}`")
            st.markdown(f"**✅ Answer:**\n{answer}")
            st.markdown("---")
