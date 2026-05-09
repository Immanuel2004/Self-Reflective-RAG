import os
import json
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# Disable SDPA requirements for transformers compatibility before imports
os.environ["TRANSFORMERS_USE_PYTORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from langchain_tavily import TavilySearch
except Exception:
    # Fallback stub when `langchain_tavily` isn't installed.
    # Keeps the app runnable; returns no web results.
    class TavilySearch:
        def __init__(self, max_results: int = 3):
            self.max_results = max_results

        def run(self, query: str):
            return []

from config.llm import get_llm, get_structured_llm,get_json_grader
from vectorstores.vectorstores import get_retriever

# Initialize
_llm = None
_retriever = None
_web_search_tool = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever

def _get_web_search_tool():
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = TavilySearch(max_results=3)
    return _web_search_tool

# --- Pydantic Schemas ---
class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(description="True if the verse is relevant to the question")

class SupportDecision(BaseModel):
    score: str = Field(description="Score: 'fully_supported' or 'no_support'")

class UtilityDecision(BaseModel):
    useful: bool = Field(description="Is the answer helpful? True/False")

NT_BOOKS = {
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians",
    "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus",
    "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John",
    "2 John", "3 John", "Jude", "Revelation"
}

# --- Nodes ---

def _build_retrieval_queries(question):
    queries = [question]
    lowered = question.lower()

    if any(keyword in lowered for keyword in ["how should we", "how do we", "what should we", "treat our enemies", "love your enemies", "forgive", "enemy"]):
        queries.append(f"New Testament teaching about {question}")
        queries.append(f"Jesus teaching about {question}")
        queries.append(f"Apostle teaching about {question}")

    return queries

def _is_new_testament_book(book_name):
    return book_name in NT_BOOKS

def _is_ethics_question(question):
    lowered = question.lower()
    return any(keyword in lowered for keyword in [
        "how should we", "how do we", "what should we", "treat our enemies",
        "love your enemies", "forgive", "enemy", "persecute", "pray for"
    ])

def retrieve(state):
    print("--- NODE: RETRIEVE ---")
    q = state.get("retrieval_query") or state['question']
    docs = []
    seen_references = set()

    for query in _build_retrieval_queries(q):
        for doc in _get_retriever().invoke(query):
            reference = doc.metadata.get("reference")
            if reference and reference not in seen_references:
                seen_references.add(reference)
                docs.append(doc)

    if _is_ethics_question(q):
        docs.sort(key=lambda doc: (0 if _is_new_testament_book(doc.metadata.get("book", "")) else 1, doc.metadata.get("book", ""), doc.metadata.get("chapter", 0)))

    return {"docs": docs}

def is_relevant(state):
    print("--- NODE: CHECK RELEVANCE (JSON MODE) ---")
    relevant_docs = []
    grader = get_json_grader() 
    ethics_question = _is_ethics_question(state["question"])

    for doc in state.get('docs', []):
        if ethics_question and _is_new_testament_book(doc.metadata.get("book", "")):
            print(f"✅ Relevant (NT priority): {doc.metadata.get('reference')}")
            relevant_docs.append(doc)
            continue

        prompt = f"""You are a Bible scholar. Determine if the following verse is relevant to the question.
        Question: {state['question']}
        Verse: {doc.page_content}
        
        Return ONLY a JSON object with a 'relevant' boolean key.
        Example: {{"relevant": true}}"""
        
        try:
            res = grader.invoke(prompt)
            data = json.loads(res.content)
            if data.get("relevant"):
                print(f"✅ Relevant: {doc.metadata.get('reference')}")
                relevant_docs.append(doc)
        except Exception as e:
            print(f"⚠️ Parsing error: {e}")
            
    return {"relevant_docs": relevant_docs}

def generate_from_context(state):
    print("--- NODE: GENERATE ---")
    references = []
    nt_passages = []
    ot_passages = []

    for doc in state['relevant_docs']:
        ref = doc.metadata.get('reference')
        book = doc.metadata.get('book', '')
        passage = f"[{ref}] {doc.page_content}"
        references.append({
            "reference": ref,
            "book": book,
            "testament": "New Testament" if _is_new_testament_book(book) else "Old Testament",
            "text": doc.page_content,
        })
        if _is_new_testament_book(book):
            nt_passages.append(passage)
        else:
            ot_passages.append(passage)

    nt_context = "\n\n".join(nt_passages)
    ot_context = "\n\n".join(ot_passages)
    context = "\n\n".join(nt_passages + ot_passages)
    prompt = f"""You are a Biblical scholar and theologian. Your task is to answer the user's question using ONLY the provided Bible passages below.

CRITICAL: Your answer must directly address what the question is asking. Stay focused on the specific question topic.

If New Testament passages are present, prioritize them in the answer and cite them in your reasoning.

QUESTION: {state['question']}

INSTRUCTIONS:
1. Answer the specific question asked - do not discuss related but different topics
2. Provide a clear, concise answer based ONLY on the Biblical context provided
3. Do NOT include citation format [Book Chapter:Verse] in your answer text - passages are referenced in Scripture References section
4. Synthesize multiple passages into a cohesive response that directly answers the question
5. Write in a respectful, scholarly tone
6. When New Testament passages are present, name them specifically in the reasoning instead of speaking generically about the New Testament
7. Do not say the New Testament is unavailable if NT passages are included below
8. If the passages don't answer the question, say so clearly

NEW TESTAMENT PASSAGES:
{nt_context}

OLD TESTAMENT PASSAGES:
{ot_context}

BIBLE PASSAGES:
{context}

ANSWER:"""
    
    out = _get_llm().invoke(prompt)
    return {"answer": out.content, "context": context, "references": references}
    
    out = _get_llm().invoke(prompt)
    return {"answer": out.content, "context": context}

def web_search_node(state):
    print("--- NODE: WEB SEARCH ---")
    results = _get_web_search_tool().run(state['question'])
    return {"relevant_docs": [Document(page_content=str(results), metadata={"reference": "Web Search"})]}

def is_sup(state):
    print("--- NODE: CHECK SUPPORT ---")
    grader = get_json_grader()
    prompt = f"""Compare the answer to the context. 
    Context: {state['context']}
    Answer: {state['answer']}
    Return ONLY JSON: {{"score": "fully_supported"}} or {{"score": "no_support"}}"""
    
    try:
        res = grader.invoke(prompt)
        data = json.loads(res.content)
        return {"issup": data.get("score", "no_support")}
    except:
        return {"issup": "no_support"}

def revise_answer(state):
    tries = state.get("retries", 0) + 1
    print(f"--- NODE: REVISE (Attempt {tries}) ---")
    prompt = f"""You are a Biblical scholar. Please improve the answer to better address the user's question based on the provided Bible passages.

ORIGINAL QUESTION: {state['question']}

CRITICAL: The revised answer must directly answer the question asked. Do not discuss related topics or go off-topic.

CONTEXT FROM BIBLE:
{state['context']}

CURRENT ANSWER: {state['answer']}

REVISION INSTRUCTIONS:
1. FOCUS: Ensure every part of the answer directly addresses the specific question
2. If the answer discusses related but different topics, refocus it on the core question
3. Use only the Bible passages provided to support your points
4. Remove any citations in [Book Chapter:Verse] format from the answer text
5. Make the answer concise and clear
6. Maintain scholarly tone
7. If the passages don't directly answer the question, acknowledge this

REVISED ANSWER:"""
    out = _get_llm().invoke(prompt)
    return {"answer": out.content, "retries": tries}
    out = _get_llm().invoke(prompt)
    return {"answer": out.content, "retries": tries}

def is_use(state):
    print("--- NODE: CHECK UTILITY (JSON MODE) ---")
    grader = get_json_grader() # Remove the argument inside the parentheses
    
    prompt = f"""You are a Bible study assistant. Does the following answer provide a helpful and 
    accurate response to the user's question?
    
    Question: {state['question']}
    Answer: {state['answer']}
    
    Return ONLY a JSON object with a 'useful' boolean key.
    Example: {{"useful": true}}"""
    
    try:
        res = grader.invoke(prompt)
        data = json.loads(res.content)
        return {"isuse": "useful" if data.get("useful") else "not_useful"}
    except Exception as e:
        print(f"⚠️ Utility parsing error: {e}")
        return {"isuse": "useful"} # Default to useful if parsing fails

def rewrite_question(state):
    print("--- NODE: REWRITE ---")
    tries = state.get("rewrite_tries", 0) + 1
    return {"retrieval_query": f"Scripture regarding {state['question']}", "rewrite_tries": tries}

def no_answer_found(state):
    return {"answer": "I could not find a supported answer in the indexed Bible or web search."}

def decide_retrieval(state):
    """Optional router if you use it in the graph start"""
    print("--- NODE: DECIDE RETRIEVAL ---")
    return {"need_retrieval": True}

def generate_direct(state):
    print("--- NODE: GENERATE DIRECT ---")
    out = _get_llm().invoke(state['question'])
    return {"answer": out.content}