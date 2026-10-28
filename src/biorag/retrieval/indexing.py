from __future__ import annotations

from biorag.config import Settings
from biorag.data.documents import chunk_document, read_documents, write_jsonl
from biorag.retrieval.dense import Embedder, VectorStore
from biorag.retrieval.retriever import bm25_path, chunks_path, qdrant_path
from biorag.retrieval.sparse import BM25Index


def build_index(settings: Settings) -> int:
    """Chunk the corpus and build both the BM25 and the vector index. Returns the chunk count."""
    chunks = [
        chunk
        for doc in read_documents(settings.corpus_path)
        for chunk in chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
    ]
    index_dir = settings.index_dir
    write_jsonl(chunks_path(index_dir), chunks)

    ids = [c.chunk_id for c in chunks]
    texts = [c.render() for c in chunks]
    BM25Index(ids, texts).save(bm25_path(index_dir))

    embedder = Embedder(settings.embedding_model)
    store = VectorStore(qdrant_path(index_dir), settings.collection)
    store.recreate(embedder.dim)
    store.upsert(ids, embedder.embed_passages(texts))
    return len(chunks)
