from typing import List, Dict, Any
from ..llm.ollama_client import OllamaClient
from ..llm.prompt_templates import PromptTemplates


class ResponseGenerator:
    """Generates responses to queries using retrieved documents and LLM.
    
    This class handles generating responses using a large language model (LLM)
    with Retrieval Augmented Generation (RAG) by combining user queries with
    relevant context retrieved from the vector database.
    """
    
    def __init__(self, llm_client: OllamaClient):
        """Initialize the response generator.
        
        Parameters
        ----------
        llm_client : OllamaClient
            Client for interacting with the Ollama LLM
        """
        self.llm_client = llm_client
        self.prompt_templates = PromptTemplates()
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate a response based on the query and retrieved documents.
        
        Parameters
        ----------
        query : str
            The user's query text
        retrieved_docs : List[Dict[str, Any]]
            List of relevant documents with their content and metadata
            
        Returns
        -------
        str
            The generated response text
        """
        if not retrieved_docs:
            return self._generate_no_context_response(query)
        
        # Create RAG prompt with retrieved contexts (& metadata)
        prompt = self.prompt_templates.create_rag_prompt(query, retrieved_docs)
        
        # Generate response using the LLM
        response = self.llm_client.generate(prompt)
        
        return response
    
    def _generate_no_context_response(self, query: str) -> str:
        """Generate a response when no relevant context is found.
        
        Parameters
        ----------
        query : str
            The user's query text
            
        Returns
        -------
        str
            The generated response text
        """
        prompt = self.prompt_templates.create_no_context_prompt(query)
        return self.llm_client.generate(prompt)
    
    def analyze_code(self, code: str) -> str:
        """Generate code analysis response.
        
        Parameters
        ----------
        code : str
            The code to analyze
            
        Returns
        -------
        str
            Analysis of the provided code
        """
        prompt = self.prompt_templates.create_code_analysis_prompt(code)
        return self.llm_client.generate(prompt)