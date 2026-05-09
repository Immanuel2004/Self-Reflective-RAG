import os
import re
import pypdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BIBLE_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", 
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", 
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", 
    "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", 
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", 
    "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", 
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians", 
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", 
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"
]

def run_ingestion(pdf_path="data/ESV_Clean.pdf"):
    print(f"--- STARTING BIBLE INGESTION: {pdf_path} ---")
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    try:
        reader = pypdf.PdfReader(pdf_path)
        documents = []
        current_book = "Preface"

        print(f"Reading {len(reader.pages)} pages...")

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text: continue
            
            header_chunk = text[:500] 
            for book in BIBLE_BOOKS:
                if re.search(rf"\b{book}\b", header_chunk):
                    current_book = book
                    break
            
            documents.append(Document(
                page_content=text,
                metadata={"reference": current_book, "page": i+1}
            ))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        final_docs = text_splitter.split_documents(documents)

        print(f"Creating embeddings for {len(final_docs)} chunks...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vector_store = FAISS.from_documents(final_docs, embeddings)
        
        vector_store.save_local("faiss_bible_index")
        print("SUCCESS: 'ESV_Clean' indexed in 'faiss_bible_index' folder.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_ingestion()