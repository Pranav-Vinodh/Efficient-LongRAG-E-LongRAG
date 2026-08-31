import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

class LongRAGVectorStore:
    """
    FAISS dense vector index for LongRAG.
    Indexes long-context chunks (~2k-4k tokens) using L2-normalized cosine similarity (IndexFlatIP).
    """
    _instance = None

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        print(f"[VectorStore] Initializing SentenceTransformer ('{model_name}')...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension() if hasattr(self.encoder, 'get_sentence_embedding_dimension') else self.encoder.get_embedding_dimension()
        self.index = None
        self.chunks = []
        self._try_load_existing_index()

    def _try_load_existing_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                print(f"[VectorStore] Successfully loaded FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"[VectorStore] Could not load existing index ({e}), will build fresh.")

    def build_index(self, chunks, force_rebuild=False):
        self.chunks = chunks
        if self.index is not None and self.index.ntotal == len(chunks) and not force_rebuild:
            print(f"[VectorStore] Reusing existing FAISS index ({self.index.ntotal} vectors).")
            return self.index

        texts = [c["text"] for c in chunks]
        print(f"[VectorStore] Encoding {len(texts)} long chunks with normalized embeddings...")
        
        start_t = time.time()
        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        encode_time = time.time() - start_t
        print(f"[VectorStore] Chunk encoding completed in {encode_time:.2f}s.")

        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)
        
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        print(f"[VectorStore] Built and saved FAISS IndexFlatIP ({self.index.ntotal} vectors) to {FAISS_INDEX_PATH}")
        return self.index

    def encode_text(self, text):
        """Encodes a single text string into a normalized 1D float32 numpy vector."""
        emb = self.encoder.encode([text], normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)

    def encode_batch(self, texts):
        """Encodes a list of text strings into normalized 2D float32 numpy array."""
        emb = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)

    def search(self, query_or_embedding, top_k=3):
        """
        Searches the FAISS vector database.
        Accepts either a string query OR a pre-computed embedding vector (e.g. from HyDE).
        """
        if self.index is None:
            raise ValueError("[VectorStore] Index is not built or loaded.")

        start_t = time.time()
        if isinstance(query_or_embedding, str):
            q_emb_np = self.encode_text(query_or_embedding)
        else:
            q_emb_np = np.array(query_or_embedding, dtype=np.float32)
            if len(q_emb_np.shape) == 1:
                q_emb_np = np.expand_dims(q_emb_np, axis=0)

        # Search nearest neighbors
        k_to_search = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb_np, k_to_search)
        latency_ms = (time.time() - start_t) * 1000.0

        results = []
        for rank in range(k_to_search):
            idx = indices[0][rank]
            score = float(scores[0][rank])
            if idx != -1 and idx < len(self.chunks):
                chunk_data = self.chunks[idx]
                results.append({
                    "rank": rank + 1,
                    "chunk_id": chunk_data["chunk_id"],
                    "similarity_score": score,
                    "approx_token_count": chunk_data.get("approx_token_count", 0),
                    "doc_titles": chunk_data.get("doc_titles", []),
                    "text": chunk_data["text"],
                    "text_snippet": chunk_data["text"][:250] + "..."
                })

        return results, latency_ms
