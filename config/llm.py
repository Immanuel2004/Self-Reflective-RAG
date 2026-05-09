import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Primary model for high-quality generation (Llama 3.3 70B)."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def get_json_grader():
    """Returns an LLM forced to output JSON without using tool-calling."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}, 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


def get_structured_llm(*args, **kwargs):
    return get_json_grader()

get_grader_llm = get_structured_llm