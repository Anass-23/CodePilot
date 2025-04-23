import os
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple

from .processors import AstParser, Chunker, MetadataExtractor
from .vector_db import FaissVectorStore, EmbeddingGenerator
from .llm import OllamaClient
from .engine import Retriever, ResponseGenerator
from .config import Config
from .logging.logger import get_logger


class CodePilot:
    """Main RAG system class combining all components.
    
    This class serves as the main entry point for the CodePilot application,
    orchestrating all components together including code parsing, chunking, 
    vector embedding, retrieval, and response generation.
    """
    
    def __init__(self):
        """Initialize all components of the CodePilot system."""
        self.logger = get_logger(__name__)
        # Initialize components
        self.ast_parser = AstParser()
        self.chunker = Chunker()
        self.metadata_extractor = MetadataExtractor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FaissVectorStore()
        self.llm_client = OllamaClient()
        self.retriever = Retriever(self.vector_store, self.embedding_generator)
        self.response_generator = ResponseGenerator(self.llm_client)
    
    def index_codebase(self, directory_path: str) -> None:
        """Index a Python codebase.
        
        Parameters
        ----------
        directory_path : str
            Path to the directory containing the Python codebase to index
        """
        self.logger.info(f"Indexing codebase at {directory_path}")
        
        # Parse all Python files
        self.logger.info("Parsing Python files...")
        documents = self.ast_parser.parse_directory(directory_path)
        
        # Enrich with metadata
        self.logger.info("Extracting metadata...")
        documents = self.metadata_extractor.enrich_metadata(documents)
        
        # Chunk documents
        self.logger.info("Chunking documents...")
        chunked_documents = self.chunker.chunk_documents(documents)
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        texts = [doc["content"] for doc in chunked_documents]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to vector store
        self.logger.info("Adding to vector store...")
        self.vector_store.add_documents(chunked_documents, embeddings)
        
        # Save vector store
        self.logger.info("Saving vector store...")
        self.vector_store.save()
        
        self.logger.info(f"Indexing complete! {len(chunked_documents)} chunks indexed.")
    
    def load_index(self) -> bool:
        """Load existing index.
        
        Returns
        -------
        bool
            True if the index was loaded successfully, False otherwise
        """
        return self.vector_store.load()

    def query(self, query_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Query the RAG system.
        
        Parameters
        ----------
        query_text : str
            The query text to search for
            
        Returns
        -------
        Tuple[str, List[Dict[str, Any]]]
            A tuple containing (response, retrieved_docs)
        """
        # Retrieve relevant documents
        self.logger.info("Retrieving relevant code...")
        retrieved_docs = self.retriever.retrieve(query_text)
        
        # Generate response
        self.logger.info("Generating response...")
        response = self.response_generator.generate_response(query_text, retrieved_docs)
        
        return response, retrieved_docs


def main():
    """Main entry point for the CLI application.
    
    Handles command line arguments, initializes the CodePilot system,
    and provides both script and interactive modes.
    """
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(description="RAG System for Python Codebases")
    parser.add_argument("--index", type=str, help="Directory path to index")
    parser.add_argument("--query", type=str, help="Query the system")
    
    args = parser.parse_args()
    
    rag_system = CodePilot()
    
    if args.index:
        rag_system.index_codebase(args.index)
    elif args.query:
        # Try to load existing index
        if rag_system.load_index():
            response, _ = rag_system.query(args.query)
            logger.info("Response generated successfully")
            print("\nResponse:")
            print(response)
        else:
            logger.error("No index found. Please index a codebase first with --index")
            print("Error: No index found. Please index a codebase first with --index")
    else:
        # Interactive mode
        if not rag_system.load_index():
            logger.warning("No existing index found. Prompting user to index a codebase.")
            print("No existing index found. Please index a codebase first.")
            directory = input("Enter the path to the Python codebase to index: ")
            if os.path.isdir(directory):
                rag_system.index_codebase(directory)
            else:
                logger.error(f"Invalid directory: {directory}")
                print(f"Invalid directory: {directory}")
                return
        
        logger.info("Starting interactive mode")
        print("\nRAG System for Python Codebases")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your query about the codebase: ")
            
            if query.lower() in ["exit", "quit"]:
                logger.info("User exited interactive mode")
                break
            
            response, _ = rag_system.query(query)
            logger.info("Response generated successfully")
            print("\nResponse:")
            print(response)


if __name__ == "__main__":
    main()