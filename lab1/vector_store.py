import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = "/home/pranav/.gemini/antigravity/scratch/e-longrag/data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(DATA_DIR, "long_chunks.json")

class LongRAGVectorStore:
    def __init__(self, model_name=MODEL_NAME):
        print(f"[VectorStore] Loading embedding model '{model_name}'...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"[VectorStore] Encoding {len(texts)} long chunks (~4k tokens each)...")
        
        start_t = time.time()
        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        encode_duration = time.time() - start_t
        print(f"[VectorStore] Encoding completed in {encode_duration:.2f} seconds.")

        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Inner Product index on L2-normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)
        print(f"[VectorStore] Added {self.index.ntotal} vectors to FAISS IndexFlatIP.")

        faiss.write_index(self.index, INDEX_FILE)
        return self.index

    def search(self, query, top_k=3):
        if self.index is None:
            raise ValueError("FAISS index has not been built or loaded.")

        start_t = time.time()
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        q_emb_np = np.array(q_emb, dtype=np.float32)

        scores, indices = self.index.search(q_emb_np, top_k)
        latency_ms = (time.time() - start_t) * 1000.0

        results = []
        for rank in range(top_k):
            idx = indices[0][rank]
            score = float(scores[0][rank])
            if idx != -1 and idx < len(self.chunks):
                chunk_data = self.chunks[idx]
                results.append({
                    "rank": rank + 1,
                    "chunk_id": chunk_data["chunk_id"],
                    "similarity_score": score,
                    "approx_token_count": chunk_data["approx_token_count"],
                    "doc_titles": chunk_data["doc_titles"],
                    "text_snippet": chunk_data["text"][:300] + "..."
                })

        return results, latency_ms

if __name__ == "__main__":
    from dataset_loader import load_or_fetch_sample_data
    from chunker import create_long_chunks
    
    ds = load_or_fetch_sample_data(100)
    chunks = create_long_chunks(ds)
    vs = LongRAGVectorStore()
    vs.build_index(chunks)
    res, lat = vs.search(ds[0]["question"], top_k=3)
    print(f"Query: {ds[0]['question']}")
    print(f"Latency: {lat:.2f} ms")
    print(f"Top result score: {res[0]['similarity_score']:.4f}")
