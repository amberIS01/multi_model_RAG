from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
        print(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore: Optional[FAISS] = None
        self.chunks: List[Dict[str, Any]] = []

        print("Embedding model loaded successfully")
        
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            print("No chunks provided for embedding")
            return

        self.chunks = chunks
        documents: List[Document] = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'page': chunk['page'],
                    'type': chunk['type'],
                    'source': chunk['source'],
                    'chunk_id': i
                }
            )
            documents.append(doc)
        
        print("Building FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"FAISS index with {len(documents)} vectors")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            print("Vectorstore not created")
            return []
        if not query or not query.strip():
            print("Empty query provided")
            return []
        if k < 1:
            k = 1

        results = self.vectorstore.similarity_search_with_score(query.strip(), k=k)

        formatted_results: List[Dict[str, Any]] = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append({
                'chunk': {
                    'content': doc.page_content,
                    'page': doc.metadata['page'],
                    'type': doc.metadata['type'],
                    'source': doc.metadata['source']
                },
                'score': float(score),
                'rank': i + 1
            })
        
        return formatted_results

    def save(self, filepath: str = 'vector_store') -> None:
        if self.vectorstore is None:
            print("No vectorstore to save")
            return
        self.vectorstore.save_local(filepath)

        with open(f"{filepath}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load(self, filepath: str = 'vector_store') -> None:
        self.vectorstore = FAISS.load_local(
            filepath,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        with open(f"{filepath}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)

        print(f"Loaded vector store with {len(self.chunks)} chunks")

if __name__ == "__main__":
    test_chunks = [
        {'content': 'Qatar has strong economic growth', 'page': 1, 'type': 'text', 'source': 'Page 1'},
        {'content': 'Banking sector remains healthy', 'page': 2, 'type': 'text', 'source': 'Page 2'},
        {'content': 'IMF recommendations for fiscal policy', 'page': 3, 'type': 'text', 'source': 'Page 3'}
    ]
    
    print("Testing LangChain Vector Store...")
    store = VectorStore()
    store.create_embeddings(test_chunks)
    
    results = store.search("What is Qatar's economic situation?", k=2)
    print(f"\nSearch Results:")
    for result in results:
        print(f"Rank {result['rank']}: {result['chunk']['content'][:50]}... (Score: {result['score']:.3f})")