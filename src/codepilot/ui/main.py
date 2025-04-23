"""
Streamlit-based UI for CodePilot.
This provides a web interface alternative to the command-line interface.
"""
import os
import sys
import streamlit as st
from typing import List, Dict, Any

if __name__ == "__main__":
    import pathlib
    src_dir = str(pathlib.Path(__file__).resolve().parents[3])
    if (src_dir not in sys.path):
        sys.path.insert(0, src_dir)
    
    from codepilot.processors import AstParser, Chunker, MetadataExtractor
    from codepilot.vector_db import FaissVectorStore, EmbeddingGenerator
    from codepilot.llm import OllamaClient
    from codepilot.engine import Retriever, ResponseGenerator
    from codepilot.config import Config
    from codepilot.logging.logger import get_logger
    from codepilot.cli import CodePilot
else:
    # NOTE: Relative imports when imported as part of the package
    from ..processors import AstParser, Chunker, MetadataExtractor
    from ..vector_db import FaissVectorStore, EmbeddingGenerator
    from ..llm import OllamaClient
    from ..engine import Retriever, ResponseGenerator
    from ..config import Config
    from ..logging.logger import get_logger
    from ..cli import CodePilot

logger = get_logger(__name__)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "code_pilot" not in st.session_state:
        st.session_state.code_pilot = CodePilot()
        
    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False
        
    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []

def load_index():
    """Load the FAISS index if available."""
    if st.session_state.code_pilot.load_index():
        st.session_state.index_loaded = True
        logger.info("Successfully loaded existing index")
        return True
    else:
        logger.warning("No existing index found")
        return False

def index_codebase(directory_path: str):
    """Index a Python codebase."""
    if not os.path.isdir(directory_path):
        st.error(f"Invalid directory: {directory_path}")
        logger.error(f"Invalid directory: {directory_path}")
        return False
    
    with st.spinner("Indexing codebase... This may take a while."):
        st.session_state.code_pilot.index_codebase(directory_path)
    
    st.session_state.index_loaded = True
    return True

def chat_interface():
    """Display the chat interface."""
    # Configure the page
    st.set_page_config(
        page_title="CodePilot",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Index management
        st.header("Index Management")
        
        if not st.session_state.index_loaded:
            load_index()
        
        if st.session_state.index_loaded:
            st.success("Index loaded successfully! Ready to answer queries.")
        else:
            st.warning("No index loaded. Please index a codebase first.")
            
            directory_path = st.text_input("Enter the path to the Python codebase to index:")
            if st.button("Index Codebase"):
                if directory_path:
                    if index_codebase(directory_path):
                        st.success("Indexing complete! Ready to answer queries.")
                else:
                    st.error("Please enter a valid directory path.")
        
        # Model configuration
        st.header("Model Configuration")
        
        # Display current model
        st.info(f"Using model: {Config.MODEL_NAME}")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.slider("Temperature", min_value=0.0, max_value=1.0, value=Config.TEMPERATURE, step=0.1, key="temperature")
            st.slider("Top K", min_value=1, max_value=10, value=Config.TOP_K, step=1, key="top_k")
            st.slider("Max Tokens", min_value=512, max_value=4096, value=Config.MAX_TOKENS, step=512, key="max_tokens")
        
        # NOTE: Safely save Config (after sliders have initialized session_state)
        Config.TOP_K = st.session_state.top_k
        Config.TEMPERATURE = st.session_state.temperature
        Config.MAX_TOKENS = st.session_state.max_tokens

        # About section
        st.header("About")
        st.markdown("""
        **CodePilot** is a RAG system for Python codebases using code-specific embeddings and FAISS.
        
        It helps you understand and navigate Python codebases by providing context-aware responses.
        """)
    
    # Main content area
    st.title("💻 CodePilot Chat")
    
    # Display welcome message if no messages
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Hello! Ask me anything about your Python codebase."})
    
    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your codebase..."):
        if not st.session_state.index_loaded:
            st.error("Please load or create an index first.")
            st.stop()
        
        # Add user message to chat history and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        # "Lazy" loading of the response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response, retrieved_docs = st.session_state.code_pilot.query(prompt)
                st.session_state.last_retrieved_docs = retrieved_docs
            
            message_placeholder.markdown(response)
            
            # Retrieved code snippets
            if retrieved_docs:
                with st.expander("📚 View retrieved code snippets", expanded=False):
                    for i, doc in enumerate(retrieved_docs, 1):
                        metadata = doc.get("metadata", {})
                        
                        # Format metadata for display
                        meta_text = f"**Source {i}:** {metadata.get('file_path', 'Unknown')}\n"
                        
                        if "type" in metadata:
                            doc_type = metadata["type"]
                            if doc_type == "class":
                                meta_text += f"**Type:** Class `{metadata.get('name', 'Unknown')}`\n"
                            elif doc_type == "function":
                                meta_text += f"**Type:** Function `{metadata.get('name', 'Unknown')}`\n"
                                meta_text += f"**Arguments:** {', '.join(metadata.get('arguments', []))}\n"
                            else:
                                meta_text += f"**Type:** {doc_type}\n"
                        
                        if "docstring" in metadata and metadata["docstring"]:
                            meta_text += f"**Docstring:** {metadata['docstring']}\n"
                        
                        # Metadata
                        st.markdown(meta_text)
                        
                        # Code content (later could be scaled to be language-aware not just pytho)
                        if doc.get("content"):
                            st.code(doc["content"], language="python")
                        
                        if i < len(retrieved_docs):
                            st.markdown("---")
            
            st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    chat_interface()

if __name__ == "__main__":
    main()