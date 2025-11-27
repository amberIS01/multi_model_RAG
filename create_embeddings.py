import json
import os
from vector_store import VectorStore
import config

def main():
    print("STEP 2: Creating Embeddings")
    print()
    print()
    
    if not os.path.exists(config.CHUNKS_PATH):
        print(f"\nerror -> Processed data not found")
        return
    
    print(f"\nprocessed data")

    print(f"\nLoading extracted chunks...")
    with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} chunks")
    
    text_count = sum(1 for c in chunks if c['type'] == 'text')
    table_count = sum(1 for c in chunks if c['type'] == 'table')
    image_count = sum(1 for c in chunks if c['type'] == 'image')
    
    print(f"  - Text chunks: {text_count}")
    print(f"  - Tables: {table_count}")
    print(f"  - Images: {image_count}")
    
    print(f"\nCreating embeddings...")
    print()
    print()
    
    vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
    vector_store.create_embeddings(chunks)
    
    vector_store.save(config.VECTOR_STORE_PATH)
    
    print("COMPLETE")
    print(f"\nTotal vectors: {len(chunks)}")

if __name__ == "__main__":
    main()