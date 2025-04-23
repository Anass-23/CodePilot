from typing import List, Dict, Any


class PromptTemplates:
    """XML-structured prompt templates for the LLM.
    
    This class provides methods to generate structured prompts in XML format
    for different types of interactions with the language model.
    """
    
    @staticmethod
    def create_rag_prompt(query: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        """Create a RAG prompt with XML structure.
        
        Parameters
        ----------
        query : str
            The user's query text
        retrieved_contexts : List[Dict[str, Any]]
            List of retrieved document contexts with content and metadata
            
        Returns
        -------
        str
            Formatted RAG prompt with XML structure
        """
        formatted_contexts = []
        
        for i, ctx in enumerate(retrieved_contexts, 1):
            metadata = ctx.get("metadata", {})
            context_str = f"<context id='{i}'>\n"
            
            # Add metadata
            context_str += "<metadata>\n"
            file_path = metadata.get("file_path", "unknown")
            context_str += f"File: {file_path}\n"
            
            if "type" in metadata:
                context_type = metadata["type"]
                context_str += f"Type: {context_type}\n"
                
                if context_type == "class":
                    context_str += f"Class: {metadata.get('name', 'unknown')}\n"
                elif context_type == "function":
                    context_str += f"Function: {metadata.get('name', 'unknown')}\n"
            
            context_str += "</metadata>\n\n"
            
            # Add code content
            if metadata.get("type") in ["class", "function"]:
                context_str += "<code>\n"
                context_str += ctx.get("content", "")
                context_str += "\n</code>"
            else:
                context_str += ctx.get("content", "")
            
            context_str += "\n</context>"
            formatted_contexts.append(context_str)
        
        joined_contexts = "\n\n".join(formatted_contexts)
        prompt = f"""<instruction>
You are an expert Python developer assistant. Your task is to answer questions about a Python codebase.
You will be given:
1. Retrieved code contexts from the codebase
2. A question about the code

Provide a helpful, accurate, and concise response. When referring to code, use code blocks.
Focus on explaining the relevant parts of the code that answer the question.
</instruction>

<contexts>
{joined_contexts}
</contexts>

<question>
{query}
</question>

<answer>
"""
        return prompt
    
    @staticmethod
    def create_no_context_prompt(query: str) -> str:
        """Create a prompt for when no relevant context is found.
        
        Parameters
        ----------
        query : str
            The user's query text
            
        Returns
        -------
        str
            Formatted prompt for queries without context
        """
        prompt = f"""<instruction>
You are an expert Python developer assistant. Your task is to answer questions about Python programming.
You've been asked a question, but no specific code context was found in the codebase.
Provide a helpful, general response based on your knowledge of Python.
</instruction>

<question>
{query}
</question>

<answer>
"""
        return prompt
    
    @staticmethod
    def create_code_analysis_prompt(code: str) -> str:
        """Create a prompt for analyzing code.
        
        Parameters
        ----------
        code : str
            The code to analyze
            
        Returns
        -------
        str
            Formatted prompt for code analysis
        """
        prompt = f"""<instruction>
You are an expert Python developer assistant. Your task is to analyze the given Python code.
Provide insights about:
1. Purpose of the code
2. Key functions and classes
3. Potential bugs or improvements
</instruction>

<code>
{code}
</code>

<analysis>
"""
        return prompt