# from typing import List, TypedDict, Literal
# from langchain_core.documents import Document

# class State(TypedDict):
#     question: str
#     need_retrieval: bool
#     docs: List[Document]
#     relevant_docs: List[Document] 
#     context: str 
#     answer: str
#     retrieval_query: str
#     rewrite_tries: int
#     issup: Literal["fully_supported", "partially_supported", "no_support"]
#     evidence: List[str]
#     retries: int
#     isuse: Literal["useful", "not_useful"]
#     use_reason: str

from typing import List, TypedDict, Literal
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document] 
    context: str 
    answer: str
    retrieval_query: str
    rewrite_tries: int
    issup: str  # fully_supported, no_support
    retries: int
    isuse: str  # useful, not_useful