"""
Configuration module for Multi-Modal RAG QA System.
Contains all paths, model names, and system constants.
"""
import os
from typing import List

VERSION: str = "1.0.0"
APP_NAME: str = "Multi-Modal RAG QA System"

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR: str = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, 'processed')
VECTOR_STORE_DIR: str = os.path.join(DATA_DIR, 'vector_store')
IMAGES_DIR: str = os.path.join(DATA_DIR, 'images')

PDF_PATH: str = os.path.join(RAW_DATA_DIR, 'qatar_test_doc.pdf')
CHUNKS_PATH: str = os.path.join(PROCESSED_DATA_DIR, 'extracted_chunks.json')
VECTOR_STORE_PATH: str = os.path.join(VECTOR_STORE_DIR, 'faiss_index')

EMBEDDING_MODEL: str = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL: str = 'google/flan-t5-base'

MAX_CHUNK_SIZE: int = 500
MIN_CHUNK_SIZE: int = 50
DEFAULT_SEARCH_RESULTS: int = 5
MAX_SEARCH_RESULTS: int = 10
MAX_CHAT_HISTORY: int = 50
MAX_OUTPUT_TOKENS: int = 512
MAX_QUERY_LENGTH: int = 500
MAX_CONTEXT_CHUNKS: int = 3

def create_directories() -> None:
    """Create all required directories for the application."""
    directories: List[str] = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        VECTOR_STORE_DIR,
        IMAGES_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("All directories created successfully")

if __name__ == "__main__":
    create_directories()
    print(f"\nDirectory structure:")
    print(f"  Raw data: {RAW_DATA_DIR}")
    print(f"  Processed data: {PROCESSED_DATA_DIR}")
    print(f"  Vector store: {VECTOR_STORE_DIR}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"\nPDF should be placed at: {PDF_PATH}")