import ast
import os
from typing import Dict, List, Tuple, Any
from ..logging.logger import get_logger

logger = get_logger(__name__)


class NodeVisitor(ast.NodeVisitor):
    """AST node visitor that adds parent references to all nodes.

    This visitor traverses the AST and adds a parent reference
    to each node, enabling bottom-up traversal.
    """
    
    def visit(self, node):
        """Visit a node and set its parent for all its children.
        
        Parameters
        ----------
        node : ast.AST
            The node to visit
        """
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self.visit(child)


class AstParser:
    """Parser for Python code using the Abstract Syntax Tree.
    
    This class provides functionality to parse Python code files and directories,
    extracting structured information about classes and functions.
    """
    
    def __init__(self):
        """Initialize the AST parser."""
        self.files_parsed = 0
        logger.debug("Initialized AstParser")
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Parse all Python files in a directory and its subdirectories.

        Parameters
        ----------
        directory_path : str
            Path to the directory to parse

        Returns
        -------
        List[Dict[str, Any]]
            List of document chunks with metadata extracted from Python files
        """
        documents = []
        skipped_dirs = 0
        
        logger.info(f"Starting to parse Python files in directory: {directory_path}")
        
        for root, _, files in os.walk(directory_path):
            # NOTE: Skip virtual environment directories (still could be improved to avoid unreleated directories)
            if "env" in root.split(os.path.sep) or "venv" in root.split(os.path.sep):
                skipped_dirs += 1
                logger.debug(f"Skipping virtual environment directory: {root}")
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        file_documents = self.parse_file(file_path)
                        documents.extend(file_documents)
                        self.files_parsed += 1
                        logger.debug(f"Successfully parsed {file_path}, extracted {len(file_documents)} document chunks")
                    except Exception as e:
                        logger.error(f"Error parsing {file_path}: {str(e)}", exc_info=True)
        
        logger.info(f"Successfully parsed {self.files_parsed} Python files, skipped {skipped_dirs} directories")
        return documents
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a single Python file into document chunks with metadata.
        
        Parameters
        ----------
        file_path : str
            Path to the Python file to parse
            
        Returns
        -------
        List[Dict[str, Any]]
            List of document chunks with metadata extracted from the file
        """
        logger.debug(f"Parsing file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        try:
            tree = ast.parse(content)
            # Add parent references to nodes
            visitor = NodeVisitor()
            tree.parent = None  # Root node has no parent
            visitor.visit(tree)
            logger.debug(f"Successfully created AST for {file_path}")
        except SyntaxError as e:
            # Fall back to raw text if AST parsing fails
            logger.warning(f"Syntax error in {file_path}: {str(e)}. Falling back to raw text.")
            return [{"content": content, "metadata": {"file_path": file_path, "type": "raw_text"}}]
        
        documents = []
        class_count = 0
        function_count = 0
        
        # Process each node in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                documents.append(self._process_class(node, file_path, content))
                class_count += 1
            elif isinstance(node, ast.FunctionDef):
                # Only include top-level functions
                try:
                    if hasattr(node, 'parent') and isinstance(node.parent, ast.Module):
                        documents.append(self._process_function(node, file_path, content))
                        function_count += 1
                except AttributeError:
                    # Skip if there's an issue with the parent attribute
                    logger.debug(f"Skipping function {getattr(node, 'name', 'unknown')} due to missing parent attribute")
                    continue
        
        logger.debug(f"Extracted {class_count} classes and {function_count} top-level functions from {file_path}")
        return documents
    
    def _process_class(self, node: ast.ClassDef, file_path: str, content: str) -> Dict[str, Any]:
        """Process a class definition node.
        
        Parameters
        ----------
        node : ast.ClassDef
            The class definition node from the AST
        file_path : str
            Path to the file containing the class
        content : str
            Source code of the file
            
        Returns
        -------
        Dict[str, Any]
            Document chunk with metadata about the class
        """
        # Get the source code
        start_line = node.lineno
        end_line = max(child.end_lineno for child in ast.walk(node) if hasattr(child, 'end_lineno')) \
                  if hasattr(node, 'end_lineno') else start_line
        
        class_source = '\n'.join(content.split('\n')[start_line-1:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Extract methods
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(child.name)
        
        logger.debug(f"Processed class {node.name} from {file_path} (lines {start_line}-{end_line}) with {len(methods)} methods")
        
        return {
            "content": class_source,
            "metadata": {
                "file_path": file_path,
                "type": "class",
                "name": node.name,
                "docstring": docstring,
                "methods": methods,
                "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                "line_range": (start_line, end_line)
            }
        }
    
    def _process_function(self, node: ast.FunctionDef, file_path: str, content: str) -> Dict[str, Any]:
        """Process a function definition node.
        
        Parameters
        ----------
        node : ast.FunctionDef
            The function definition node from the AST
        file_path : str
            Path to the file containing the function
        content : str
            Source code of the file
            
        Returns
        -------
        Dict[str, Any]
            Document chunk with metadata about the function
        """
        # Get the source code
        start_line = node.lineno
        end_line = max(child.end_lineno for child in ast.walk(node) if hasattr(child, 'end_lineno')) \
                  if hasattr(node, 'end_lineno') else start_line
        
        function_source = '\n'.join(content.split('\n')[start_line-1:end_line])
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Extract arguments
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        logger.debug(f"Processed function {node.name} from {file_path} (lines {start_line}-{end_line}) with {len(args)} arguments")
        
        return {
            "content": function_source,
            "metadata": {
                "file_path": file_path,
                "type": "function",
                "name": node.name,
                "docstring": docstring,
                "arguments": args,
                "line_range": (start_line, end_line)
            }
        }