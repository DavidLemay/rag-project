from src.rag.pipeline import RAGPipeline

if __name__ == "__main__":
    pipeline = RAGPipeline()
    user_query = input("Enter your question: ")
    results = pipeline.run(user_query)
    for subq, answer in results.items():
        print(f"Q: {subq}\nA: {answer}\n")
