import json
import requests
from typing import Dict, Any, List, Optional
from ..config import Config
from ..logging.logger import get_logger


class OllamaClient:
    """Client for interacting with Ollama API.
    
    This class provides methods to connect to a local or remote Ollama server
    and generate text completions, chat responses, and embeddings.
    """
    
    def __init__(self, base_url: str = None, model_name: str = None):
        """Initialize the Ollama client.
        
        Parameters
        ----------
        base_url : str, optional
            Base URL of the Ollama API server
        model_name : str, optional
            Name of the model to use for generation
        """
        self.logger = get_logger(__name__)
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model_name = model_name or Config.MODEL_NAME
        
        # NOTE: Check if Ollama is running
        self.check_ollama_availability()
    
    def check_ollama_availability(self):
        """Check if Ollama is available and the model is loaded.
        
        Validates that the Ollama server is running and the requested model
        is available, with fallback to similar models if needed.
        """
        try:
            # Check server is running
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if self.model_name not in model_names:
                self.logger.warning(f"Model '{self.model_name}' not found in Ollama. Available models: {', '.join(model_names)}")
                # If the exact model isn't found, check if we can find a match with the base name
                model_base_name = self.model_name.split(':')[0] if ':' in self.model_name else self.model_name
                available_model = next((m for m in model_names if model_base_name in m), None)
                if available_model:
                    self.logger.info(f"Using available model '{available_model}' instead of '{self.model_name}'")
                    self.model_name = available_model
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running.")
        except Exception as e:
            self.logger.error(f"Error checking Ollama availability: {str(e)}")
    
    def generate(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Generate text completion from Ollama.
        
        Parameters
        ----------
        prompt : str
            The input prompt to generate text from
        temperature : float, optional
            Sampling temperature for generation
        max_tokens : int, optional
            Maximum number of tokens to generate
            
        Returns
        -------
        str
            The generated text response
        """
        url = f"{self.base_url}/api/generate"
        
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature if temperature is not None else Config.TEMPERATURE,
            "num_predict": max_tokens if max_tokens is not None else Config.MAX_TOKENS,
            "stream": False
        }
        
        try:
            self.logger.debug(f"Sending request to {url} with model {self.model_name}")
            response = requests.post(url, json=params)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Error: Failed to generate response from Ollama: {str(e)}"
    
    def generate_chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = None, 
        max_tokens: int = None
    ) -> str:
        """Generate chat completion from Ollama.
        
        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries with role and content keys
        temperature : float, optional
            Sampling temperature for generation
        max_tokens : int, optional
            Maximum number of tokens to generate
            
        Returns
        -------
        str
            The generated chat response text
        """
        url = f"{self.base_url}/api/chat"
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else Config.TEMPERATURE,
            "num_predict": max_tokens if max_tokens is not None else Config.MAX_TOKENS,
            "stream": False
        }
        
        try:
            self.logger.debug(f"Sending chat request to {url} with model {self.model_name}")
            response = requests.post(url, json=params)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            return f"Error: Failed to generate chat response from Ollama: {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text from Ollama.
        
        Parameters
        ----------
        text : str
            The text to generate embeddings for
            
        Returns
        -------
        List[float]
            Vector embedding of the input text
        """
        url = f"{self.base_url}/api/embeddings"
        
        params = {
            "model": Config.EMBEDDING_MODEL,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=params)
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return []
