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

# -------------------
# BM25
# -------------------

with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

# -------------------
# Reranker
# -------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# -------------------
# Gemini
# -------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
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


def format_context(docs):
    return "\n\n---\n\n".join(
        f"# File: {d.metadata['source']}\n{d.page_content}"
        for d in docs
    )


# -------------------
# Hybrid Search
# -------------------

def hybrid_search(query, k=12):

    semantic = vectorstore.similarity_search(
        query,
        k=k
    )

    return semantic


# -------------------
# Reranking
# -------------------

def search_with_rerank(query, k=4):

    candidates = hybrid_search(query, k=12)

    pairs = [
        (query, d.page_content)
        for d in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, _ in ranked[:k]]


# -------------------
# QA
# -------------------

def ask(question):

    docs = search_with_rerank(question)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "context": format_context(docs),
            "question": question
        }
    )


# -------------------
# Gradio
# -------------------

def gradio_respond(message, history):
    return ask(message)

demo = gr.ChatInterface(
    fn=gradio_respond,
    type="messages",
    title="Multi-Repository Code Intelligence Assistant",
    description="Query multiple repositories using semantic search, BM25 retrieval, and cross-encoder reranking.",
    examples=[
        "How Initiator makes a connection to target in TLM 2.0?",
        "How is b_transport implemented?",
        "Where is bcrypt used?",
        "How are JWT tokens generated and validated in ecommerce app?",
        "How does the simon game generate the next sequence?"
    ]
)

demo.launch()