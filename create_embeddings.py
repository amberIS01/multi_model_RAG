import json
import os
from typing import List, Dict, Any
from vector_store import VectorStore
import config

def main() -> None:
    print("=" * 50)
    print("STEP 2: Creating Embeddings")
    print("=" * 50)

    if not os.path.exists(config.CHUNKS_PATH):
        print(f"Error: Processed data not found at {config.CHUNKS_PATH}")
        print("Please run process_document.py first")
        return

    print(f"\nLoading extracted chunks from {config.CHUNKS_PATH}...")
    with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(chunks)} chunks successfully")

    text_count = sum(1 for c in chunks if c['type'] == 'text')
    table_count = sum(1 for c in chunks if c['type'] == 'table')
    image_count = sum(1 for c in chunks if c['type'] == 'image')

    print(f"\nChunk breakdown:")
    print(f"  - Text chunks: {text_count}")
    print(f"  - Tables: {table_count}")
    print(f"  - Images: {image_count}")

    print(f"\nCreating embeddings using {config.EMBEDDING_MODEL}...")

    vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
    vector_store.create_embeddings(chunks)

    print(f"\nSaving vector store to {config.VECTOR_STORE_PATH}...")
    vector_store.save(config.VECTOR_STORE_PATH)

    print("\n" + "=" * 50)
    print("EMBEDDING CREATION COMPLETE")
    print(f"Total vectors created: {len(chunks)}")
    print("=" * 50)

if __name__ == "__main__":
    main()