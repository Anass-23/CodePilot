import ast
from typing import Dict, Any, List


class MetadataExtractor:
    """Extract additional metadata from Python code.
    
    This class provides functionality to analyze code content and extract
    meaningful metadata such as imports, function calls, and complexity metrics.
    """
    
    def enrich_metadata(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add additional metadata to documents based on content analysis.
        
        Parameters
        ----------
        documents : List[Dict[str, Any]]
            List of document dictionaries containing content and metadata
            
        Returns
        -------
        List[Dict[str, Any]]
            Documents with enriched metadata
        """
        enriched_documents = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            if metadata["type"] in ["class", "function"]:
                # Extract additional metadata for code
                enriched_metadata = self._extract_code_metadata(content, metadata)
                doc["metadata"].update(enriched_metadata)
            
            enriched_documents.append(doc)
        
        return enriched_documents
    
    def _extract_code_metadata(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract advanced metadata from code snippets.
        
        Parameters
        ----------
        content : str
            The code content to analyze
        metadata : Dict[str, Any]
            The existing metadata for the code
            
        Returns
        -------
        Dict[str, Any]]
            Additional metadata extracted from the code
        """
        additional_metadata = {}
        
        try:
            tree = ast.parse(content)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        imports.append(f"{module}.{name.name}")
            
            if imports:
                additional_metadata["imports"] = imports
            
            # Extract function calls
            function_calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                    function_calls.append(node.func.id)
            
            if function_calls:
                additional_metadata["function_calls"] = function_calls
            
            # Complexity analysis - count branches
            branches = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    branches += 1
            
            additional_metadata["complexity"] = {
                "branches": branches
            }
            
        except SyntaxError:
            # If parsing fails, we skip the additional metadata
            pass
        
        return additional_metadata