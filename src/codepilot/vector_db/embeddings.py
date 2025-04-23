import numpy as np
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
from ..config import Config


class EmbeddingGenerator:
    """Generate embeddings for documents using a Hugging Face code embedding model.
    
    This class provides functionality to convert text into vector embeddings
    using pre-trained language models from Hugging Face.
    """
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding generator with a specific model.
        
        Parameters
        ----------
        model_name : str, optional
            Name of the Hugging Face model to use for embeddings
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        # Force CPU usage
        self.device = "cpu"
        self.model.to(self.device)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings to generate embeddings for
            
        Returns
        -------
        np.ndarray
            Array of embedding vectors, one for each input text
        """
        embeddings = []
        
        # Process in small batches
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            if i > 0 and i % (batch_size * 5) == 0:
                print(f"Generated embeddings for {i}/{len(texts)} texts")
            
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self._get_embeddings_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts using the HuggingFace model.
        
        Parameters
        ----------
        texts : List[str]
            Batch of text strings to process together
            
        Returns
        -------
        List[List[float]]
            List of embedding vectors as lists of floats
        """
        try:
            # Tokenize inputs
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                # Use CLS token embedding (first token) for sentence representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * Config.VECTOR_DIMENSION] * len(texts)