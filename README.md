# 🔍 RAG Question Answering System (Streamlit + Qdrant)

This project implements a **Retrieval-Augmented Generation (RAG)** system using:

- 📦 Prebuilt document embeddings stored in a **Qdrant vector database**
- 🧠 A router agent to decide whether to answer via OpenAI docs, 10-K financial filings, or the internet
- 💬 A Streamlit-based UI with observability dashboards
- ⚡ Semantic caching and query deduplication
- 🌐 Optional web search agent (ARES API)

---

## 📁 Project Structure
```yaml
rag-project/
├── app.py # Streamlit UI (entry point)
├── src/
│ ├── rag/ # RAG pipeline logic (router, retriever, cache, generator)
│ ├── utils/ # Config and logger utils
│ └── ...
├── logs/ # JSONL logs of user queries and decisions
├── multi-agent-course/ # ✅ Cloned external repo with Qdrant vector database
│ └── Module_1/Agentic_RAG/qdrant_data/
├── pyproject.toml
├── README.md
└── .env
```

---

## ⚙️ Local Setup Instructions

> 🐍 Requires Python 3.11+

### 1. Clone this repo and navigate to it:

```bash
git clone <your-repo-url>
cd rag-project
```
### 2. Clone the Qdrant vector database (required):
```bash
git clone https://github.com/hamzafarooq/multi-agent-course.git
```
⚠️ This must be cloned inside the root of the RAG app, so that this folder exists:

multi-agent-course/Module_1/Agentic_RAG/qdrant_data/

This database includes:

- opnai_data: vectorized OpenAI documentation

- 10k_data: vectorized SEC filings (Uber, Lyft)

### 3. Install dependencies using Poetry
If you don’t have Poetry:

```bash
pip install poetry
```
Then run:

```bash
poetry install
```
### 4. Create a .env file (optional)
```env
OPENAI_API_KEY=your-key-here
ARES_API_KEY=your-key-here
QDRANT_PATH=multi-agent-course/Module_1/Agentic_RAG/qdrant_data
``` 

### 🚀 Run the App
```bash
poetry run streamlit run app.py
```
This will launch a local Streamlit UI at http://localhost:8501

### 🧠 UI Features
#### 📝 Ask any question via the text box

#### 🔁 Enable or disable web search (uses ARES API)

#### 💾 Semantic caching to avoid redundant queries

#### 📚 Automatic routing to OpenAI docs, 10-Ks, or web

#### 📊 Log dashboard with:

##### Query count

##### Cache hit rate

##### Routing decision breakdown

#### Latency stats

📓 Example Queries
"How do I fine-tune a model with OpenAI?"

"Show me the latest Lyft 10-K results."

"What are the top leadership trends in 2024?"

✨ Future Ideas
Add document upload support

Integrate user feedback (👍/👎)

### Deploy to Streamlit Cloud, Heroku, or Azure