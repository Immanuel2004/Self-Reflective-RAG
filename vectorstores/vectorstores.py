import os

# Disable SDPA requirements for transformers compatibility
os.environ["TRANSFORMERS_USE_PYTORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Use your existing index path
    db_path = "faiss_bible_index" 
    
    if os.path.exists(db_path):
        vector_store = FAISS.load_local(
            db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_kwargs={"k": 5})
    else:
        raise FileNotFoundError("FAISS index not found. Run ingestion first.")