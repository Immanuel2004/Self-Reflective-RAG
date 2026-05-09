import os
import re
import time

# Disable SDPA requirements for transformers compatibility
os.environ["TRANSFORMERS_USE_PYTORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from graph_builder.graph_builder import build_bible_graph

st.set_page_config(page_title="Self-Reflective RAG - Biblical QA", layout="wide")

st.title("Biblical Q&A Assistant")
st.write("Powered by Self-Reflective RAG — Ask questions about the Bible and receive answers with source citations")

st.markdown("---")
with st.sidebar:
    st.header("Settings")
    answer_style = st.selectbox(
        "Answer Style",
        ["Concise", "Detailed", "Academic"],
        help="Choose how detailed the answer should be"
    )
    include_context = st.checkbox("Show context references", value=True)
    max_citations = st.slider("Max citations to show", 1, 10, 3)

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Your question", placeholder="e.g., What does the Bible say about love?", label_visibility="collapsed")
with col2:
    ask_button = st.button("Search", use_container_width=True, type="primary")

st.markdown("---")

with st.expander("Example questions", expanded=False):
    examples = [
        "What are the fruits of the spirit?",
        "What does Jesus teach about forgiveness?",
        "What is the greatest commandment?",
        "What is love according to the Bible?",
        "How should we treat our enemies?"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            question = ex
            ask_button = True

if ask_button and question.strip():
    start_time = time.time()
    
    progress_placeholder = st.empty()
    result_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.info("Searching for relevant passages and generating answer...")
        progress_bar = st.progress(0)
    
    try:
        workflow = build_bible_graph()
        inputs = {
            "question": question,
            "retries": 0,
            "rewrite_tries": 0,
            "docs": [],
            "relevant_docs": []
        }

        final_results = {}
        node_count = 0
        
        for output in workflow.stream(inputs):
            for node, values in output.items():
                final_results.update(values)
                node_count += 1
                progress_bar.progress(min(0.9, node_count * 0.15))

        progress_bar.progress(1.0)
        elapsed = time.time() - start_time
        
        progress_placeholder.empty()
        
        answer = final_results.get("answer") or "(no answer returned)"
        context = final_results.get("context") or ""
        references = final_results.get("references") or []

        citations = []
        unique_sources = set()
        # Extract citations from structured references when available
        if references:
            for item in references:
                ref = item.get("reference", "Unknown reference")
                snippet = item.get("text", "")
                book = item.get("book", "Unknown")
                citations.append((ref, snippet, item.get("testament", "")))
                unique_sources.add(book)
        elif context:
            pattern = re.compile(r"\[([^\]]+)\]\s*([^\[]+)", re.M)
            for m in pattern.finditer(context):
                ref = m.group(1).strip()
                snippet = m.group(2).strip()
                citations.append((ref, snippet, ""))
                book = ref.split()[0] if ref else "Unknown"
                unique_sources.add(book)

        with result_placeholder.container():
            # Answer section
            st.markdown("### Answer")
            st.write(answer)
            st.markdown("---")
            
            col_cite, col_meta = st.columns([3, 1])
            
            with col_cite:
                st.markdown("### Scripture References")
                if citations:
                    limited_citations = citations[:max_citations]
                    st.write(f"Found **{len(citations)}** matching passage{'s' if len(citations) != 1 else ''}" + 
                             (f" (showing {len(limited_citations)})" if len(citations) > max_citations else ""))
                    
                    for i, citation in enumerate(limited_citations, 1):
                        ref, snippet, testament = citation
                        label = f"[{i}] {ref}"
                        if testament:
                            label = f"{label} — {testament}"
                        with st.expander(label):
                            st.write(snippet)
                else:
                    st.info("No citations found in the context.")
            
            with col_meta:
                with st.expander("Query Details"):
                    st.metric("Books Referenced", len(unique_sources))
                    
                    if unique_sources:
                        st.markdown("**Sources:**")
                        for source in sorted(unique_sources):
                            st.write(source)
            
            st.markdown("---")
            st.caption("Each scripture reference is extracted and verified from the ingested Bible passages.")

    except Exception as e:
        st.error(f"Error during query processing: {str(e)}")
        st.caption("Please check that the FAISS index is built and Ollama/LLM service is running.")
        progress_placeholder.empty()
