## Multi-Repository Code Intelligence Assistant

An AI-powered code assistant that enables natural language querying across multiple software repositories.
The system combines semantic search, BM25 keyword retrieval, and cross-encoder reranking to locate relevant code and generate accurate answers grounded in source files.

## Features

- Query multiple repositories using natural language
- Supports C, C++, SystemC/TLM, Python, JavaScript, TypeScript, Markdown, and Jupyter notebooks
- Smart language-aware code chunking
- Semantic retrieval using Sentence Transformers and ChromaDB
- BM25 keyword search for exact identifier matching
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for improved retrieval accuracy
- Interactive Gradio web interface
- Repository-aware indexing and retrieval

## Tech Stack

- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Rank-BM25
- Cross Encoder (ms-marco-MiniLM-L-6-v2)
- Google Gemini
- Gradio
- Python

## Retrieval Pipeline

### Stage 1 — Repository Ingestion

The system clones one or more Git repositories and converts source files into LangChain Documents

### Stage 2 — Smart Chunking

Language-specific chunking strategies are used:

| Language                | Strategy                    |
| ----------------------- | --------------------------- |
| Python                  | Python-aware splitter       |
| JavaScript / TypeScript | JS-aware splitter           |
| C / C++ / SystemC / TLM | C++-aware splitter          |
| Markdown                | Header-aware splitter       |
| Others                  | Recursive fallback splitter |

### Stage 3 — Hybrid Retrieval

The system performs:

1. Semantic Search using embeddings stored in ChromaDB
2. BM25 Keyword Search for exact identifier matching

### Stage 4 — Cross-Encoder Reranking

Retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This significantly improves retrieval quality by jointly evaluating the query and candidate chunk.

## Demo

Access the live demo here: https://huggingface.co/spaces/Priyabrata111/multi_repo_GPT_v2

<img width="1911" height="617" alt="image" src="https://github.com/user-attachments/assets/6ad54b95-35cd-44e9-ae74-8d5b473dd886" />

<img width="1792" height="857" alt="image" src="https://github.com/user-attachments/assets/cf840ff5-67d8-4d33-abac-c6f625cc24ae" />

## Author

Priyabrata Mondal

GitHub: https://github.com/Priyabrata111
