from langgraph.graph import StateGraph, START, END
from state.rag_state import State
from nodes import nodes

def route_after_relevance(state):
    return "generate_from_context" if state.get("relevant_docs") else "web_search"

def route_after_issup(state):
    if state.get("issup") == "fully_supported" or state.get("retries", 0) >= 2:
        return "is_use"
    return "revise_answer"

def route_after_isuse(state):
    if state.get("isuse") == "useful" or state.get("rewrite_tries", 0) >= 1:
        return END
    return "rewrite_question"

def build_bible_graph():
    workflow = StateGraph(State)

    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("is_relevant", nodes.is_relevant)
    workflow.add_node("web_search", nodes.web_search_node)
    workflow.add_node("generate_from_context", nodes.generate_from_context)
    workflow.add_node("is_sup", nodes.is_sup)
    workflow.add_node("revise_answer", nodes.revise_answer)
    workflow.add_node("is_use", nodes.is_use)
    workflow.add_node("rewrite_question", nodes.rewrite_question)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "is_relevant")
    workflow.add_conditional_edges("is_relevant", route_after_relevance)
    workflow.add_edge("web_search", "generate_from_context")
    workflow.add_edge("generate_from_context", "is_sup")
    workflow.add_conditional_edges("is_sup", route_after_issup)
    workflow.add_edge("revise_answer", "is_sup")
    workflow.add_conditional_edges("is_use", route_after_isuse)
    workflow.add_edge("rewrite_question", "retrieve")

    return workflow.compile()