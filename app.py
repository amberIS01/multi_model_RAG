import streamlit as st
import os
import json
from vector_store import VectorStore
from llm_qa import LLMQA, SimpleQA
import config

st.set_page_config(
    page_title="Multi-Modal RAG",
    page_icon="📚",
    layout="wide"
)

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if not st.session_state.loaded:
    faiss_file = os.path.join(config.VECTOR_STORE_PATH, "index.faiss")
    if os.path.exists(faiss_file) or os.path.exists(f"{config.VECTOR_STORE_PATH}.faiss"):
        with st.spinner("Loading pre-processed data..."):
            try:
                vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
                vector_store.load(config.VECTOR_STORE_PATH)
                st.session_state.vector_store = vector_store
                
                try:
                    qa_system = LLMQA(model_name=config.LLM_MODEL)
                    st.session_state.qa_system = qa_system
                except:
                    st.warning("Using simple QA (LLM model failed to load)")
                    st.session_state.qa_system = SimpleQA()
                
                st.session_state.loaded = True
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.session_state.loaded = False

st.title("Multi-Modal RAG System")
st.markdown("*Intelligent document Q&A powered by AI*")

with st.sidebar:
    st.header("System Status")

    if st.session_state.loaded:
        st.success("System Ready")

        st.markdown("---")
        st.subheader("Try These Questions")
        sample_qs = [
            "What is the economic outlook?",
            "What are the key recommendations?",
            "Summarize the fiscal policy",
        ]
        for q in sample_qs:
            st.caption(f"• {q}")

        if st.session_state.vector_store:
            total = len(st.session_state.vector_store.chunks)
            text_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'text')
            table_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'table')
            image_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'image')

            st.markdown("---")
            st.subheader("Document Statistics")
            st.metric("Total Chunks", total)
            col1, col2, col3 = st.columns(3)
            col1.metric("Text", text_count)
            col2.metric("Tables", table_count)
            col3.metric("Images", image_count)

        st.markdown("---")
        st.subheader("Session Info")
        st.caption(f"Queries: {st.session_state.query_count}")

        st.markdown("---")
        st.subheader("Model Info")
        st.caption(f"Embedding: {config.EMBEDDING_MODEL.split('/')[-1]}")
        st.caption(f"LLM: {config.LLM_MODEL.split('/')[-1]}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.query_count = 0
                st.rerun()
        with col2:
            if st.session_state.chat_history:
                chat_json = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    "Export",
                    chat_json,
                    "chat_history.json",
                    "application/json"
                )

    else:
        st.error("Data Not Loaded")
        st.markdown("---")
        st.subheader("Setup Required")
        st.markdown("""
Please run the following commands:

**Step 1: Process Document**
```bash
python process_document.py
```

**Step 2: Create Embeddings**
```bash
python create_embeddings.py
```

**Step 3: Restart App**
```bash
streamlit run app.py
```
""")
        

# Main chat interface
if st.session_state.loaded:
    st.markdown("---")

    if not st.session_state.chat_history:
        st.info("Welcome! Ask any question about the document and I'll find the relevant information.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message:
                with st.expander(f"View {len(message['citations'])} Citations"):
                    for i, cite in enumerate(message["citations"], 1):
                        score_pct = (1 - cite['relevance_score']) * 100
                        st.markdown(
                            f"**[{i}] {cite['source']}**  \n"
                            f"Type: `{cite['type']}` | Relevance: {score_pct:.1f}%"
                        )

    query = st.chat_input("Type your question here (e.g., What is the economic outlook?)")
    
    if query:
        st.session_state.query_count += 1
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document and generating response..."):
                search_results = st.session_state.vector_store.search(query, k=5)
                
                result = st.session_state.qa_system.generate_answer_with_citations(
                    query, search_results
                )
                
                st.markdown(result['answer'])
                
                with st.expander("View Citations"):
                    for cite in result['citations']:
                        st.markdown(
                            f"**{cite['source']}** | "
                            f"Type: {cite['type']} | "
                            f"Relevance: {cite['relevance_score']:.3f}"
                        )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "citations": result['citations']
                })

else:
    st.info("Please follow the setup steps in the sidebar")

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>"
    f"Multi-Modal RAG System v{config.VERSION}</div>",
    unsafe_allow_html=True
)