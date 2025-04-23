import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from ..config import Config
from ..logging.logger import get_logger


class FaissVectorStore:
    """Vector store implementation using FAISS.
    
    This class provides a vector database implementation using Facebook AI Similarity Search (FAISS)
    for efficient similarity search and clustering of dense vectors.
    """
    
    def __init__(self, dimension=768):
        """Initialize a FAISS vector store.
        
        Parameters
        ----------
        dimension : int, optional
            Dimensionality of the embedding vectors, defaults to 768
        """
        self.logger = get_logger(__name__)
        self.index = None
        self.metadata = []
        self.dimension = dimension
        
        data_dir = os.path.join(os.path.expanduser("~"), ".codepilot", "data")
        self.index_path = os.path.join(data_dir, Config.INDEX_PATH)
        self.metadata_path = os.path.join(data_dir, Config.METADATA_PATH)
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.load()
                self.logger.info(f"Successfully loaded existing index from {self.index_path}")
            except (FileNotFoundError, IOError) as e:
                self.logger.warning(f"Found index files but failed to load them: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()
            self.logger.info(f"No existing index found at {self.index_path}. Created a new empty index.")
    
    def _create_empty_index(self):
        """Initialize an empty FAISS index with the specified dimension.
        
        This method creates a new L2-distance (Euclidean) FAISS index.
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.logger.info(f"Created new empty FAISS index with dimension {self.dimension}")
        
        try:
            self.save()
            self.logger.info(f"Saved empty index to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Failed to save empty index: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the vector store.
        
        Parameters
        ----------
        documents : List[Dict[str, Any]]
            List of document dictionaries containing content and metadata
        embeddings : np.ndarray
            Matrix of document embeddings with shape (n_documents, dimension)
        """
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self._create_empty_index()
        
        # Store metadata
        for doc in documents:
            self.metadata.append(doc["metadata"])
        
        # Add vectors to the index
        self.index.add(embeddings)
        
        self.logger.info(f"Added {len(documents)} documents to vector store. Total: {self.index.ntotal}")
        
        # Save index to disk after adding documents
        try:
            self.save()
            self.logger.info(f"Saved index to {self.index_path} after adding documents")
        except Exception as e:
            self.logger.error(f"Failed to save index after adding documents: {e}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents based on query embedding.
        
        Parameters
        ----------
        query_embedding : np.ndarray
            Embedding vector of the query
        top_k : int, optional
            Number of top results to retrieve
            
        Returns
        -------
        List[Dict[str, Any]]
            List of search results with metadata and distance scores
        """
        if top_k is None:
            top_k = Config.TOP_K
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue  # Skip invalid indices
            
            results.append({
                "metadata": self.metadata[idx],
                "distance": float(distances[0][i])
            })
        
        return results
    
    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> None:
        """Save the index and metadata to disk.
        
        Parameters
        ----------
        index_path : str, optional
            Path to save the FAISS index
        metadata_path : str, optional
            Path to save the metadata JSON file
        """
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path
        
        if self.index is None:
            raise ValueError("No index to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        
        self.logger.info(f"Saved vector store with {self.index.ntotal} documents")
    
    def load(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> bool:
        """Load the index and metadata from disk.
        
        Parameters
        ----------
        index_path : str, optional
            Path to load the FAISS index from
        metadata_path : str, optional
            Path to load the metadata JSON file from
            
        Returns
        -------
        bool
            True if loading was successful, False otherwise
        """
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.logger.info(f"Loaded vector store with {self.index.ntotal} documents")
            return True
        except (FileNotFoundError, IOError) as e:
            self.logger.error(f"Failed to load vector store: {e}")
            return False