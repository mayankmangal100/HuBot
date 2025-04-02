"""Streamlit app for RAG chatbot interface."""

import streamlit as st
import os
from src.utils import Config
from src.rag_system import RAGSystem

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        config = Config.load_config("config/config.json")
        st.session_state.rag = RAGSystem(config=config)
        # Try to load existing data
        data_loaded = st.session_state.rag.load_existing_data()
        if data_loaded:
            st.session_state.data_loaded = True
        else:
            st.session_state.data_loaded = False

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Processing")
        
        document_path = "Documents/EveriseHandbook.pdf"
        
        # Show appropriate UI based on data state
        if st.session_state.data_loaded:
            st.success("Using existing vector store")
            if st.button("Reprocess Document"):
                with st.spinner("Reprocessing document..."):
                    st.session_state.rag.ingest(document_path)
                st.success("Document reprocessed successfully!")
        else:
            if os.path.exists(document_path):
                st.warning("No existing data found. Please process the document.")
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        st.session_state.rag.ingest(document_path)
                        st.session_state.data_loaded = True
                    st.success("Document processed successfully!")
            else:
                st.error("Document not found!")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View Sources"):
                    st.markdown(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add assistant message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Stream the response
            stream = True
            if stream:
                text = ""
                for token in st.session_state.rag.answer_question(prompt, stream=True):
                    text += token
                    message_placeholder.write(text)
                answer = text
                
                # Update conversation history after streaming completes
                # Get the last streamed response from LLM interface
                last_response = st.session_state.rag.llm.get_last_streamed_response()
                
                # Update conversation history with the complete response
                st.session_state.rag.query_rewriter.add_exchange(
                    query=prompt,
                    rewritten_query=st.session_state.rag.last_retrieval_query,
                    answer=last_response,
                    query_embedding=st.session_state.rag.last_query_embedding,
                    context_docs=st.session_state.rag.last_context_docs
                )
            else:
                answer = st.session_state.rag.answer_question(prompt, stream=False)
                message_placeholder.write(answer)
            
            # Add sources
            sources = "\n".join([
                f"Source: {doc['metadata']['source']} (Chunk {doc['metadata']['chunk_id']}, Score: {score:.3f})"
                for doc, score in st.session_state.rag.last_context_docs
            ])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            
            with st.expander("View Sources"):
                st.markdown(sources)

if __name__ == "__main__":
    main()
