import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import streamlit as st

from src.rag.pipeline import RAGPipeline

# Path to your log file
LOG_PATH = "logs/rag_log.jsonl"

# Page setup
st.set_page_config(page_title="RAG QA System", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose a view:", ["RAG QA", "Logs Dashboard"])

# RAG QA MODE
if mode == "RAG QA":
    st.title("📚 RAG Question Answering System")
    st.markdown(
        "Ask your question and get answers using vector search and optional internet retrieval."
    )

    query = st.text_area("📝 Enter your question:", height=150)
    allow_web_search = st.checkbox("Allow Web Search", value=True)

    if st.button("🔍 Submit"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing your query..."):
                pipeline = RAGPipeline(allow_web_search=allow_web_search)
                results = pipeline.run(query)

            st.success("Results:")
            for subq, answer in results.items():
                st.markdown(f"**🧩 Sub-question:** `{subq}`")
                st.markdown(f"**✅ Answer:**\n{answer}")
                st.markdown("---")

# LOGS DASHBOARD MODE
elif mode == "Logs Dashboard":
    st.title("📊 RAG Pipeline Logs")

    if not os.path.exists(LOG_PATH):
        st.warning("No logs yet. Submit a query first.")
    else:
        df = pd.read_json(LOG_PATH, lines=True)

        st.subheader("📌 Summary Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", len(df))
        col2.metric(
            "Cache Hit Rate",
            f"{(df['cache_hit'].mean() * 100):.1f}%" if "cache_hit" in df else "N/A",
        )
        col3.metric(
            "Avg Latency (s)",
            f"{df['latency_seconds'].mean():.2f}" if "latency_seconds" in df else "N/A",
        )

        st.subheader("📦 Routing Decision Breakdown")
        if "route_action" in df:
            route_counts = df["route_action"].value_counts()
            st.bar_chart(route_counts)

        st.subheader("📈 Latency Distribution")
        if "latency_seconds" in df:
            st.line_chart(df["latency_seconds"])

        st.subheader("📝 Recent Logs")
        st.dataframe(df.sort_values("timestamp", ascending=False).head(20))

        if st.button("🧹 Clear Logs"):
            open(LOG_PATH, "w").close()
            st.success("Logs cleared.")
