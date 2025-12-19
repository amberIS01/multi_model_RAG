"""
Logging configuration module for the Multi-Modal RAG system.
"""
import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get or create a logger instance."""
    if name is None:
        name = 'rag_system'
    return setup_logger(name)
