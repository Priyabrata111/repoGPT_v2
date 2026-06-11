import os
import re
import pickle

import gradio as gr

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------
# Embeddings
# -------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# Chroma
# -------------------

vectorstore = Chroma(
    collection_name="multi_repo_v1",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
#-----------------------
# verify db
#-----------------------
import os

db_path = "./chroma_db"

print(os.path.exists(db_path))
print(os.listdir(db_path))
# -------------------
# BM25
# -------------------

with open("bm25_bundle.pkl", "rb") as f:
    data = pickle.load(f)

bm25 = data["bm25"]
smart_chunks = data["chunks"]
# -------------------
# Reranker
# -------------------

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "./reranker_model"
)

# -------------------
# Gemini
# -------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key= os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful coding assistant answering questions about a codebase.

Use ONLY the context below.

If the answer is not present in the context, say:
"I don't see that in the code."

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------
# Utilities
# -------------------

def tokenize(text):
    return [t.lower() for t in re.findall(r"\w+", text)]


# def format_context(docs):
#     return "\n\n---\n\n".join(
#         f"# File: {d.metadata['source']}\n{d.page_content}"
#         for d in docs
#     )

def format_context(docs):

    return "\n\n".join(
        f"""
Repository: {d.metadata.get('repo')}

File: {d.metadata.get('source')}

Code:
{d.page_content}
"""
        for d in docs
    )

# -------------------
# Hybrid Search
# -------------------

def hybrid_search(query, k=12):

    semantic_docs = vectorstore.similarity_search(
        query,
        k=k
    )

    bm25_scores = bm25.get_scores(
        tokenize(query)
    )

    top_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    keyword_docs = [
        smart_chunks[i]
        for i in top_idx
    ]

    return semantic_docs + keyword_docs

# -------------------
# Reranking
# -------------------

def search_with_rerank(query, k=4):

    candidates = hybrid_search(query)

    pairs = [
        (query, doc.page_content)
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:k]]



# -------------------
# QA
# -------------------

def ask(question):

    docs = search_with_rerank(question)

    print("\n===== RETRIEVED DOCS =====")

    for i, d in enumerate(docs, 1):
        print(
            f"{i}. "
            f"{d.metadata.get('repo')} | "
            f"{d.metadata.get('source')}"
        )

    print("==========================\n")

    docs = search_with_rerank(question)

    print("===== CONTEXT =====")
    print(format_context(docs)[:5000])
    print("===================")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "context": format_context(docs),
            "question": question
        }
    )
#--------------------
# Debugging 
#--------------------

import torch
import transformers
import sentence_transformers
import numpy as np

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("numpy:", np.__version__)
# -------------------
# Gradio
# -------------------

def gradio_respond(message, history):
    return ask(message)

demo = gr.ChatInterface(
    fn=gradio_respond,
    title="Multi-Repository Code Intelligence Assistant",
    description="Query multiple repositories using semantic search, BM25 retrieval, and cross-encoder reranking.",
    examples=[
        "How is the JWT token validated?",
        "How is b_transport implemented?",
        "Which file contains the implementation of nb_transport_fw()",
        "How are JWT tokens generated and validated in ecommerce app?",
        "Where is the game-over condition implemented in the Simon Game?"
    ]
)

demo.launch()