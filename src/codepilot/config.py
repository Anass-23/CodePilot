class Config:
    """Configuration settings for the RAG system."""
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Vector store
    EMBEDDING_MODEL = "microsoft/codebert-base"  # Code-specific embedding model
    VECTOR_DIMENSION = 768
    INDEX_PATH = "faiss_index"
    METADATA_PATH = "metadata.json"
    
    # LLM
    OLLAMA_BASE_URL = "http://localhost:11434"
    MODEL_NAME = "codellama:7b"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.1
    
    # Query engine
    TOP_K = 5
    
    # Logging configuration
    LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_TO_CONSOLE = True
    LOG_TO_FILE = False
    LOG_FILE_PATH = None
    LOG_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s:%(className)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    LOG_AS_JSON = False
    LOG_JSON_FIELDS = None
    INCLUDE_CLASS_NAME = True