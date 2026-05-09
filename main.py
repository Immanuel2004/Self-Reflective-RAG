import os
from graph_builder.graph_builder import build_bible_graph

def main():
    if not os.path.exists("faiss_bible_index"):
        print("Error: Bible index not found. Please run 'uv run python data_ingestion/ingest_bible.py' first.")
        return

    app = build_bible_graph()
    
    inputs = {
        "question": "What does the Bible say about the fruit of the Spirit?",
        "retries": 0,
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": []
    }

    print("\n--- INITIATING SELF-REFLECTIVE RAG ---")
    final_state = {}
    for output in app.stream(inputs):
        for node, values in output.items():
            final_state.update(values)

    print("\n" + "="*50)
    print("FINAL RESPONSE:")
    print(final_state.get("answer"))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()