# Multi-Modal RAG QA System - Technical Report

**Project:** Document Intelligence System for Qatar IMF Article IV Consultation Report
**Author:** Sahil
**Date:** November 2024

---

## 1. Introduction

This report presents a Multi-Modal Retrieval-Augmented Generation (RAG) system designed to answer questions about the Qatar IMF Article IV Consultation Report (2024). The system processes text, tables, and images from a 73-page PDF document, creating an intelligent QA interface with citation-backed answers.

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Modal RAG Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│  Document Ingestion  →  Embedding Creation  →  QA Interface     │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ PDF Parser   │    │ Sentence-BERT    │    │ Streamlit UI │  │
│  │ (PyMuPDF)    │───▶│ Embeddings       │───▶│ + Flan-T5    │  │
│  │ + Tesseract  │    │ (all-MiniLM-L6)  │    │ LLM          │  │
│  │ OCR          │    │ + FAISS Index    │    │              │  │
│  └──────────────┘    └──────────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| PDF Processing | PyMuPDF (fitz) | Text & structure extraction |
| OCR | Tesseract + Pillow | Image text extraction |
| Embeddings | all-MiniLM-L6-v2 | Semantic vector encoding (384-dim) |
| Vector Store | FAISS | Similarity search (L2 distance) |
| LLM | Google Flan-T5-base | Context-aware answer generation |
| Interface | Streamlit | Interactive chat UI |

## 3. Multi-Modal Document Processing

### 3.1 Text Extraction
The `DocumentProcessor` class extracts text content page-by-page using PyMuPDF. Each page becomes a separate chunk with metadata (page number, source reference).

### 3.2 Table Detection
Tables are identified by analyzing text block structures. Blocks with 3+ lines containing structured data are classified as tables and extracted with their positional metadata.

### 3.3 Image OCR Pipeline
Images are extracted from the PDF, saved locally, and processed through Tesseract OCR:
1. Extract image bytes from PDF using xref
2. Convert to PIL Image format
3. Apply OCR to extract text content
4. Store OCR text as image-type chunk

### 3.4 Extraction Results

| Content Type | Chunks Extracted |
|--------------|------------------|
| Text | 78 |
| Tables | 619 |
| Images (OCR) | 13 |
| **Total** | **710** |

## 4. Retrieval System

### 4.1 Embedding Strategy
All chunks (text, tables, images) are embedded using the Sentence-Transformers model `all-MiniLM-L6-v2`:
- **Dimension:** 384
- **Normalization:** L2 normalized embeddings
- **Advantage:** Fast inference, good semantic understanding

### 4.2 Vector Index
FAISS (Facebook AI Similarity Search) provides efficient nearest-neighbor search:
- **Index Type:** Flat L2 (exact search)
- **Index Size:** ~1.1 MB
- **Query Time:** <100ms for k=5 results

### 4.3 Search Process
```
User Query → Embed Query → FAISS Similarity Search → Top-k Results
```

Each result includes:
- Content snippet
- Page number
- Content type (text/table/image)
- Relevance score

## 5. Answer Generation

### 5.1 LLM Integration (Primary)
The Flan-T5-base model generates answers using retrieved context:
1. Top-3 chunks are concatenated into context
2. Prompt template guides the model to answer from context
3. Model generates concise, context-grounded answers

**Prompt Template:**
```
Based on the following context, answer the question.
If the answer is not in the context, say "I cannot find
this information in the document."

Context: {context}
Question: {question}
Answer:
```

### 5.2 Fallback System (SimpleQA)
If the LLM fails to load, a SimpleQA fallback returns top-k relevant snippets directly without LLM inference.

### 5.3 Citation System
Every answer includes citations with:
- Source reference (page, content type)
- Relevance score
- Rank in retrieval results

## 6. User Interface

The Streamlit interface provides:
- **Chat Input:** Natural language questions
- **Response Display:** Answers with expandable citations
- **Session Management:** Persistent chat history
- **Status Indicators:** System readiness and chunk statistics
- **Clear History:** Reset conversation button

## 7. Evaluation & Performance

### 7.1 Qualitative Assessment
| Aspect | Performance |
|--------|-------------|
| Text retrieval | Accurate for factual queries |
| Table data | Successfully retrieves numerical data |
| OCR content | Dependent on image quality |
| Answer coherence | Good for specific questions |

### 7.2 System Performance
| Metric | Value |
|--------|-------|
| Embedding creation | ~2 min (710 chunks) |
| Query latency | <2s (search + generation) |
| Index load time | ~3s |
| Memory usage | ~500MB (with LLM) |

## 8. Limitations & Future Work

### Current Limitations
- Table detection is heuristic-based (not using ML)
- Single document support
- No cross-modal reranking
- Limited to Flan-T5-base (248M params)

### Potential Improvements
1. **Vision-text embeddings** (CLIP) for better cross-modal retrieval
2. **Hybrid search** combining dense + sparse retrieval (BM25 + embeddings)
3. **Larger LLMs** (Flan-T5-large/XL) for better generation
4. **Evaluation dashboard** with retrieval metrics (MRR, Recall@k)
5. **Document summarization** for briefing generation

## 9. Conclusion

This Multi-Modal RAG system successfully:
- Processes diverse content types from complex PDF documents
- Creates semantic embeddings for efficient retrieval
- Generates citation-backed answers via LLM
- Provides an intuitive chat interface for document QA

The modular architecture allows easy extension and improvement of individual components.

---

## Quick Start

```bash
# 1. Process document
python process_document.py

# 2. Create embeddings
python create_embeddings.py

# 3. Run interface
streamlit run app.py
```

---

**Technology Stack:** Python, PyMuPDF, Tesseract OCR, Sentence-Transformers, FAISS, HuggingFace Transformers, Streamlit
