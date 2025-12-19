"""
Utility functions for the Multi-Modal RAG system.
"""
from typing import List, Dict, Any
import os


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def count_chunks_by_type(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count chunks grouped by type."""
    counts = {'text': 0, 'table': 0, 'image': 0}
    for chunk in chunks:
        chunk_type = chunk.get('type', 'text')
        if chunk_type in counts:
            counts[chunk_type] += 1
    return counts


def file_exists(filepath: str) -> bool:
    """Check if a file exists at the given path."""
    return os.path.exists(filepath) and os.path.isfile(filepath)
