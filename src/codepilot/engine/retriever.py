import numpy as np
from typing import List, Dict, Any
from ..vector_db.faiss_store import FaissVectorStore
from ..vector_db.embeddings import EmbeddingGenerator
from ..config import Config
from ..logging.logger import get_logger


class Retriever:
    """Retrieves relevant documents for a query.
    
    This class handles the retrieval of relevant documents from a vector store
    based on semantic similarity to the input query.
    """
    
    def __init__(self, vector_store: FaissVectorStore, embedding_generator: EmbeddingGenerator):
        """Initialize the retriever with the necessary components.
        
        Parameters
        ----------
        vector_store : FaissVectorStore
            The vector database to search through
        embedding_generator : EmbeddingGenerator
            Component for generating embeddings from text
        """
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query.
        
        Parameters
        ----------
        query : str
            The search query text
        top_k : int, optional
            Number of top results to retrieve
            
        Returns
        -------
        List[Dict[str, Any]]
            List of relevant documents with their metadata and relevance scores
        """
        if top_k is None:
            top_k = Config.TOP_K
        
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embeddings([query])
        
        # Search the vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        return self._enrich_results(results)
    
    def _enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with original content.
        
        Parameters
        ----------
        results : List[Dict[str, Any]]
            The raw search results from the vector store
            
        Returns
        -------
        List[Dict[str, Any]]
            Enhanced results with additional content and relevance scores
        """
        enriched_results = []
        
        for result in results:
            metadata = result["metadata"]
            distance = result["distance"]
            
            # Get content from the metadata
            content = self._get_content_from_metadata(metadata)
            
            # Calculate a relevance score (inverse of distance)
            relevance_score = 1.0 / (1.0 + distance)
            
            enriched_results.append({
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "relevance_score": relevance_score
            })
        
        # Sort by relevance score (highest first)
        enriched_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return enriched_results
    
    def _get_content_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Retrieve content based on metadata.
        
        Parameters
        ----------
        metadata : Dict[str, Any]]
            Metadata about the document to retrieve
            
        Returns
        -------
        str
            The content associated with the metadata
        """
        # Check if content is directly available in metadata
        if "content" in metadata:
            return metadata["content"]
        
        file_path = metadata.get("file_path", "unknown")
        content_type = metadata.get("type", "unknown")
        
        # If we have line ranges, use them for more accurate content extraction
        if "line_range" in metadata:
            try:
                start, end = metadata["line_range"]
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                return ''.join(lines[start-1:end])
            except Exception as e:
                self.logger.error(f"Error loading content from {file_path}: {e}")
        
        # Fallback for different content types
        if content_type == "class":
            name = metadata.get("name", "UnknownClass")
            return f"class {name}:\n    # Content not available - check file {file_path}\n    pass"
        
        elif content_type == "function":
            name = metadata.get("name", "unknown_function")
            args = ", ".join(metadata.get("arguments", []))
            return f"def {name}({args}):\n    # Content not available - check file {file_path}\n    pass"
        
        else:
            # For other types of content
            return f"# Content from {file_path} (unavailable in search results)"
