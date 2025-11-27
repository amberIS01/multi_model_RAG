import json
import os
from document_processor import DocumentProcessor
import config

def main():
    print("="*70)
    print("STEP 1: Document Processing")
    print("="*70)
    
    config.create_directories()
    
    if not os.path.exists(config.PDF_PATH):
        print(f"\nERROR: PDF not found ")
        return
    
    print(f"\n Found PDF")
    processor = DocumentProcessor(config.PDF_PATH)
    
    chunks = processor.process_document()
    processor.close()
    
    print(f"\n Extracted {len(chunks)} chunks")
    
    text_count = sum(1 for c in chunks if c['type'] == 'text')
    table_count = sum(1 for c in chunks if c['type'] == 'table')
    image_count = sum(1 for c in chunks if c['type'] == 'image')
    
    print(f"  - Text chunks: {text_count}")
    print(f"  - Tables: {table_count}")
    print(f"  - Images (OCR): {image_count}")
    
    print(f"\nSaving data ")
    with open(config.CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()