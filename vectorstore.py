"""
Vector store module for the Berkshire transcript RAG system.

Loads per-year JSON transcript files, embeds section titles (keys) using
Google Generative AI (Gemini) embeddings, and stores them in a LangChain
InMemoryVectorStore.
"""

import glob
import json
import os
import time

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Gemini free-tier: 100 requests/min, 1000 requests/day.
# We use small batches with generous delays to stay within limits.
_BATCH_SIZE = 50
_BATCH_DELAY_SECONDS = 60
_MAX_RETRIES = 5
_RETRY_DELAY_SECONDS = 60
_STORE_PATH = "vectorstore_cache.json"


def _load_documents(documents_dir: str) -> list[Document]:
    """Load all Q&A documents from transcript JSON files."""
    json_files = sorted(glob.glob(os.path.join(documents_dir, "transcripts_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No transcript JSON files found in {documents_dir}")

    documents: list[Document] = []

    for filepath in json_files:
        filename = os.path.basename(filepath)
        year = filename.replace("transcripts_", "").replace(".json", "")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for title, answers in data.items():
            doc = Document(
                page_content=title,
                metadata={
                    "year": year,
                    "buffett_answer": answers[0],
                    "munger_answer": answers[1],
                },
            )
            documents.append(doc)

    return documents


def _add_batch_with_retry(vectorstore: InMemoryVectorStore, batch: list[Document]) -> None:
    """Add a batch of documents with retry on rate-limit errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < _MAX_RETRIES:
                wait = _RETRY_DELAY_SECONDS * attempt
                print(f"  Rate limited. Retrying in {wait}s (attempt {attempt}/{_MAX_RETRIES})...")
                time.sleep(wait)
            else:
                # Save progress before crashing
                vectorstore.dump(_STORE_PATH)
                print(f"  Saved partial progress ({len(vectorstore.store)} docs) to {_STORE_PATH}")
                raise


def load_or_build_vectorstore(
    documents_dir: str = "documents",
) -> InMemoryVectorStore:
    """
    Load the vector store from cache, or build it incrementally.

    Saves progress after each batch so work is never lost.
    If a partial cache exists, resumes from where it left off.

    Args:
        documents_dir: Path to directory containing transcripts_YYYY.json files.

    Returns:
        An InMemoryVectorStore populated with all Q&A documents.
    """
    all_docs = _load_documents(documents_dir)
    print(f"Found {len(all_docs)} Q&A documents on disk")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Load existing partial/complete cache if available
    if os.path.exists(_STORE_PATH):
        vectorstore = InMemoryVectorStore.load(_STORE_PATH, embeddings)
        cached_count = len(vectorstore.store)
        print(f"Loaded {cached_count} documents from cache")

        if cached_count >= len(all_docs):
            print("Cache is complete — ready to query!")
            return vectorstore

        # Resume from where we left off
        remaining_docs = all_docs[cached_count:]
        print(f"Resuming: {len(remaining_docs)} documents remaining...")
    else:
        vectorstore = InMemoryVectorStore(embeddings)
        remaining_docs = all_docs
        print("No cache found. Building from scratch...")

    # Embed remaining documents in batches, saving after each
    total_remaining = len(remaining_docs)
    total_batches = (total_remaining + _BATCH_SIZE - 1) // _BATCH_SIZE
    for i in range(0, total_remaining, _BATCH_SIZE):
        batch = remaining_docs[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        done = len(vectorstore.store)
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} docs, {done}/{len(all_docs)} total)...")

        _add_batch_with_retry(vectorstore, batch)

        # Save after every successful batch
        vectorstore.dump(_STORE_PATH)
        print(f"  Saved checkpoint ({len(vectorstore.store)}/{len(all_docs)} docs)")

        if i + _BATCH_SIZE < total_remaining:
            print(f"  Waiting {_BATCH_DELAY_SECONDS}s to respect rate limits...")
            time.sleep(_BATCH_DELAY_SECONDS)

    print(f"Vector store complete! {len(vectorstore.store)} documents ready.")
    return vectorstore


def query_vectorstore(
    store: InMemoryVectorStore, query: str, k: int = 5
) -> list[Document]:
    """
    Query the vector store for similar documents.

    Args:
        store: The vector store to query.
        query: Natural language query string.
        k: Number of results to return.

    Returns:
        List of matching Documents with metadata containing answers.
    """
    return store.similarity_search(query, k=k)
