import streamlit as st
import os
from vector_store import VectorStore
from llm_qa import LLMQA, SimpleQA
import config

st.set_page_config(
    page_title="RAG multi-model"
)

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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

st.title(" Multi-Modal RAG ")
st.markdown("Ask questions about the Qatar IMF Report")

with st.sidebar:
    st.header("file Status")
    
    if st.session_state.loaded:
        st.success(" Ready to use ")

        if st.session_state.vector_store:
            total = len(st.session_state.vector_store.chunks)
            text_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'text')
            table_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'table')
            image_count = sum(1 for c in st.session_state.vector_store.chunks if c['type'] == 'image')
        
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.error(" Data Not Found!")
        st.markdown("---")
        st.subheader(" Setup Required")
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
 
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message:
                with st.expander("📎 View Citations"):
                    for cite in message["citations"]:
                        st.markdown(
                            f"**{cite['source']}** | "
                            f"Type: {cite['type']} | "
                            f"Relevance: {cite['relevance_score']:.3f}"
                        )

    query = st.chat_input("Ask a question about the document...")
    
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
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
    st.info(" Follow steps")
    
    st.markdown("""
    Step 1: python process_document.py
    ### Step 2:python create_embeddings.py
    ### Step 3:streamlit run app.py
    """)