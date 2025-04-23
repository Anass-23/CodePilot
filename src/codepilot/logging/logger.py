import logging
import os
import inspect
import sys
import datetime
import json
from typing import Dict, Any, Optional, Union, List
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from ..config import Config


class CustomFormatter(logging.Formatter):
    """Custom formatter that allows flexible formatting with various components.
    
    Parameters
    ----------
    fmt : str, optional
        Format string for log records
    datefmt : str, optional
        Format string for dates
    style : str, optional
        Style of the format string (%, {, or $)
    validate : bool, optional
        Whether to validate the format string
    include_class_name : bool, optional
        Whether to include the class name in logs
        
    Notes
    -----
    Supported components in the format string:
    - timestamp: Current time in the specified format
    - level: Log level (DEBUG, INFO, etc.)
    - name: Logger name
    - module: Module where the log was created
    - filename: File where the log was created
    - lineno: Line number where the log was created
    - funcName: Function where the log was created
    - className: Class where the log was created (if applicable)
    - process: Process ID
    - thread: Thread ID
    - message: The log message
    """

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, include_class_name=True):
        super().__init__(fmt, datefmt, style, validate)
        self.include_class_name = include_class_name

    def formatTime(self, record, datefmt=None):
        """Override to provide more detailed timestamp formatting.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to format
        datefmt : str, optional
            Format string for the timestamp
            
        Returns
        -------
        str
            Formatted timestamp
        """
        if datefmt:
            return datetime.datetime.fromtimestamp(record.created).strftime(datefmt)
        else:
            return datetime.datetime.fromtimestamp(record.created).isoformat()

    def format(self, record):
        """Add class name to the record if available and requested.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to format
            
        Returns
        -------
        str
            Formatted log record
        """
        if self.include_class_name and not hasattr(record, 'className'):
            frame = inspect.currentframe()
            try:
                frame = frame.f_back
                while frame:
                    if 'self' in frame.f_locals:
                        record.className = frame.f_locals['self'].__class__.__name__
                        break
                    frame = frame.f_back
                else:
                    record.className = 'None'
            except Exception:
                record.className = 'Unknown'
            finally:
                del frame

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after gathering all the log record attributes.
    
    Parameters
    ----------
    fields : List[str], optional
        List of specific fields to include in the JSON output
    time_format : str, optional
        Format string for timestamps
    """
    def __init__(self, fields: Optional[List[str]] = None, time_format: str = '%Y-%m-%d %H:%M:%S'):
        self.fields = fields
        self.time_format = time_format
        super().__init__()

    def format(self, record):
        """Format the record as a JSON string.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to format
            
        Returns
        -------
        str
            JSON-formatted log record
        """
        log_data = {}
        
        for attr in record.__dict__:
            if not attr.startswith('_') and (not self.fields or attr in self.fields):
                if attr == 'created':
                    log_data['timestamp'] = datetime.datetime.fromtimestamp(
                        record.created).strftime(self.time_format)
                elif attr == 'exc_info' and record.exc_info:
                    log_data['exception'] = self.formatException(record.exc_info)
                else:
                    log_data[attr] = getattr(record, attr)

        if hasattr(record, 'className'):
            log_data['className'] = record.className
            
        return json.dumps(log_data)


class CodePilotLogger:
    """Main logger class for CodePilot that provides a customizable logging system.
    
    This class allows control over log format, destinations, and log levels
    throughout the CodePilot application.
    
    Parameters
    ----------
    name : str, optional
        The logger name, typically __name__
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_str : str, optional
        Custom format string for logs
    date_format : str, optional
        Custom date format string
    log_to_console : bool, optional
        Whether to log to console
    log_to_file : bool, optional
        Whether to log to a file
    log_file : str, optional
        Path to log file (uses default if None)
    log_as_json : bool, optional
        Whether to log in JSON format
    json_fields : List[str], optional
        List of fields to include in JSON log
    max_file_size : int, optional
        Maximum log file size in bytes before rotating
    backup_count : int, optional
        Number of backup log files to keep
    include_class_name : bool, optional
        Whether to include class name in logs
    propagate : bool, optional
        Whether to propagate logs to parent loggers
    """
    
    DEFAULT_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s:%(className)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    def __init__(
        self,
        name: str = None,
        level: str = "INFO",
        format_str: str = None,
        date_format: str = None,
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_file: str = None,
        log_as_json: bool = False,
        json_fields: List[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        include_class_name: bool = True,
        propagate: bool = False
    ):
        self.name = name or "codepilot"
        self.level = self._get_log_level(level)
        self.format_str = format_str or self.DEFAULT_FORMAT
        self.date_format = date_format or self.DEFAULT_DATE_FORMAT
        self.include_class_name = include_class_name
        
        # Get or create the logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.propagate = propagate
        
        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Configure handlers
        if log_to_console:
            self._add_console_handler(log_as_json, json_fields)
            
        if log_to_file:
            if log_file is None:
                # Default log file path
                log_dir = os.path.join(os.path.expanduser("~"), ".codepilot", "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{self.name}.log")
            self._add_file_handler(log_file, max_file_size, backup_count, log_as_json, json_fields)
    
    def _get_log_level(self, level: str) -> int:
        """Convert a string log level to its numeric value.
        
        Parameters
        ----------
        level : str
            String representation of log level
            
        Returns
        -------
        int
            Numeric log level
        """
        return self.LOG_LEVELS.get(level.upper(), logging.INFO)
    
    def _create_formatter(self, log_as_json: bool, json_fields: List[str] = None):
        """Create a formatter based on the configuration.
        
        Parameters
        ----------
        log_as_json : bool
            Whether to use JSON formatting
        json_fields : List[str], optional
            Fields to include in JSON output
            
        Returns
        -------
        logging.Formatter
            The configured formatter
        """
        if log_as_json:
            return JSONFormatter(fields=json_fields, time_format=self.date_format)
        else:
            return CustomFormatter(
                fmt=self.format_str,
                datefmt=self.date_format,
                include_class_name=self.include_class_name
            )
    
    def _add_console_handler(self, log_as_json: bool, json_fields: List[str] = None):
        """Add a console handler to the logger.
        
        Parameters
        ----------
        log_as_json : bool
            Whether to use JSON formatting
        json_fields : List[str], optional
            Fields to include in JSON output
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self._create_formatter(log_as_json, json_fields))
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(
        self,
        log_file: str,
        max_file_size: int,
        backup_count: int,
        log_as_json: bool,
        json_fields: List[str] = None
    ):
        """Add a file handler to the logger.
        
        Parameters
        ----------
        log_file : str
            Path to the log file
        max_file_size : int
            Maximum file size before rotation in bytes
        backup_count : int
            Number of backup files to keep
        log_as_json : bool
            Whether to use JSON formatting
        json_fields : List[str], optional
            Fields to include in JSON output
        """
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self._create_formatter(log_as_json, json_fields))
        self.logger.addHandler(file_handler)
    
    def add_daily_file_handler(
        self,
        log_file: str,
        backup_count: int = 30,
        log_as_json: bool = False,
        json_fields: List[str] = None
    ):
        """Add a time-rotating file handler that rotates at midnight every day.
        
        Parameters
        ----------
        log_file : str
            Path to the log file
        backup_count : int, optional
            Number of backup files to keep
        log_as_json : bool, optional
            Whether to use JSON formatting
        json_fields : List[str], optional
            Fields to include in JSON output
        """
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timed rotating file handler
        handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=backup_count
        )
        handler.setLevel(self.level)
        handler.setFormatter(self._create_formatter(log_as_json, json_fields))
        handler.suffix = "%Y-%m-%d"
        self.logger.addHandler(handler)
    
    def debug(self, msg, *args, **kwargs):
        """Log a debug message.
        
        Parameters
        ----------
        msg : str
            The message to log
        *args : Any
            Arguments to pass to the logger
        **kwargs : Any
            Keyword arguments to pass to the logger
        """
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """Log an info message.
        
        Parameters
        ----------
        msg : str
            The message to log
        *args : Any
            Arguments to pass to the logger
        **kwargs : Any
            Keyword arguments to pass to the logger
        """
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Log a warning message.
        
        Parameters
        ----------
        msg : str
            The message to log
        *args : Any
            Arguments to pass to the logger
        **kwargs : Any
            Keyword arguments to pass to the logger
        """
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Log an error message.
        
        Parameters
        ----------
        msg : str
            The message to log
        *args : Any
            Arguments to pass to the logger
        **kwargs : Any
            Keyword arguments to pass to the logger
        """
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """Log a critical message.
        
        Parameters
        ----------
        msg : str
            The message to log
        *args : Any
            Arguments to pass to the logger
        **kwargs : Any
            Keyword arguments to pass to the logger
        """
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, exc_info=True, **kwargs):
        """Log an exception with traceback.
        
        Parameters
        ----------
        msg : str
            The message to log
        *args : Any
            Arguments to pass to the logger
        exc_info : bool, optional
            Whether to include exception info
        **kwargs : Any
            Keyword arguments to pass to the logger
        """
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)
    
    def set_level(self, level: Union[str, int]):
        """Set the logging level.
        
        Parameters
        ----------
        level : Union[str, int]
            The new logging level, either as a string or an integer constant
        """
        if isinstance(level, str):
            level = self._get_log_level(level)
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


# Create a default logger instance
default_logger = CodePilotLogger(
    name="codepilot",
    level=getattr(Config, "LOG_LEVEL", "INFO"),
    log_to_console=getattr(Config, "LOG_TO_CONSOLE", True),
    log_to_file=getattr(Config, "LOG_TO_FILE", False),
    include_class_name=True
)


def get_logger(name: str) -> CodePilotLogger:
    """Get a logger instance with the specified name.
    
    Parameters
    ----------
    name : str
        The name for the logger, typically __name__
        
    Returns
    -------
    CodePilotLogger
        A configured logger instance
    """
    level = getattr(Config, "LOG_LEVEL", "INFO")
    log_to_console = getattr(Config, "LOG_TO_CONSOLE", True)
    log_to_file = getattr(Config, "LOG_TO_FILE", False)

    return CodePilotLogger(
        name=name,
        level=level,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )