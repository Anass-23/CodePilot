from typing import Dict, List, Any
import re
from ..config import Config


class Chunker:
    """Split documents into chunks for embedding.
    
    This class handles splitting different types of content into appropriate 
    sized chunks for vector embedding, with special handling for code vs text.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize the Chunker with configurable chunk size and overlap.
        
        Parameters
        ----------
        chunk_size : int, optional
            Maximum size of each chunk in characters
        chunk_overlap : int, optional
            Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks, preserving metadata.
        
        Parameters
        ----------
        documents : List[Dict[str, Any]]
            List of documents to be chunked, each with content and metadata
            
        Returns
        -------
        List[Dict[str, Any]]
            List of chunked documents with updated metadata
        """
        chunked_documents = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Special handling based on content type
            if metadata["type"] in ["class", "function"]:
                # For code, we try to keep "logical blocks" together
                chunks = self._chunk_code(content, metadata)
            else:
                # For other content, use standard text chunking
                chunks = self._chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                
                chunked_documents.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
        
        return chunked_documents
    
    def _chunk_code(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Chunk code by logical structure (simplified approach)
        
        Parameters
        ----------
        content : str
            The code content to chunk
        metadata : Dict[str, Any]
            Metadata about the code content
            
        Returns
        -------
        List[str]
            List of code chunks
        """
        # If it's already small enough, return as is
        if len(content) <= self.chunk_size:
            return [content]
        
        # Identify code type from metadata
        content_type = metadata.get("type", "")
        
        # For functions/classes, we will try basic approach to keep them whole
        if content_type == "function":
            # Just return the whole function if possible
            if len(content) <= self.chunk_size * 2:  # Allow larger chunks for functions
                return [content]
        
        # For classes or larger functions, split by "logical blocs"
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        indent_level = None
        
        for line in lines:
            line_with_newline = line + '\n'
            line_size = len(line_with_newline)
            
            # Check for logical boundaries based on indentation
            stripped = line.lstrip()
            if stripped and indent_level is None:
                # Initialize indent level with first non-empty line
                indent_level = len(line) - len(stripped)
            
            # Start new chunk at logical boundaries (methods, decorators, etc.)
            if (line.startswith('@') or  # Decorator
                (stripped.startswith('def ') and not line.startswith(' ' * (indent_level + 4) if indent_level is not None else False)) or  # Method/function
                (stripped.startswith('class ') and not line.startswith(' ' * (indent_level + 4) if indent_level is not None else False))):  # Nested class
                
                # If we have an existing chunk that's not empty, save it
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            # If this line would make the chunk too large, start a new one
            if current_size + line_size > self.chunk_size * 1.5 and current_chunk:  # Allow some flexibility
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add the final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # If we didn't split successfully, fall back to simple size-based chunking
        if not chunks:
            return self._simple_chunk_by_size(content)
        
        return chunks

    def _simple_chunk_by_size(self, content: str) -> List[str]:
        """Very simple chunking approach based purely on size.
        
        Parameters
        ----------
        content : str
            The content to chunk
            
        Returns
        -------
        List[str]
            List of content chunks divided by size
        """
        # If it's already small enough, return as is
        if len(content) <= self.chunk_size:
            return [content]
        
        # Simple chunking by characters
        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunks.append(content[i:i + self.chunk_size])
        
        return chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """Simple text chunking without complex overlap logic.
        
        Parameters
        ----------
        text : str
            The text content to chunk
            
        Returns
        -------
        List[str]
            List of text chunks
        """
        # If it's already small enough, return as iss
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para) + 2  # +2 for the newlines
            
            # If adding this paragraph would exceed chunk size, save current chunk and start new one
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # If a single paragraph is too large, split it
            if para_size > self.chunk_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                for sentence in sentences:
                    sentence_size = len(sentence) + 1  # +1 for the space
                    
                    if current_size + sentence_size > self.chunk_size and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    # If a single sentence is still too large, use the simple approach
                    if sentence_size > self.chunk_size:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        
                        # Add sentence chunks directly to our chunks list
                        sentence_chunks = self._simple_chunk_by_size(sentence)
                        chunks.extend(sentence_chunks)
                        
                        current_chunk = []
                        current_size = 0
                        continue
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add the final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
