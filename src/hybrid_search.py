"""
BM25 Inverted Index and Hybrid Search Engine (Dense + Sparse RRF).
Pure Python / NumPy implementation of Okapi BM25 with Robertson-Spärck Jones IDF.
Requires zero external C++ / Java dependencies, running at sub-millisecond CPU latency.
"""

import math
import re
import time
from collections import Counter
from typing import List, Dict, Any, Tuple


class BM25Index:
    """
    Inverted index implementation of the Okapi BM25 ranking function:
    Score(D, Q) = sum_{q in Q} IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
    with RSJ-smoothed IDF:
    IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_ids = []
        self.doc_data = []
        self.term_freqs = []  # List of Counter(tokens) per doc
        self.doc_freqs = Counter()  # Term -> number of documents containing term
        self.idf_cache = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, robust lowercased regex word tokenizer."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def fit(self, documents: List[Dict[str, Any]], text_key: str = "text", id_key: str = "chunk_id"):
        """Indexes a list of document dicts."""
        self.corpus_size = len(documents)
        self.doc_ids = [doc.get(id_key, f"doc_{i}") for i, doc in enumerate(documents)]
        self.doc_data = documents
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = Counter()
        self.idf_cache = {}

        total_tokens = 0
        for doc in documents:
            text = doc.get(text_key, "")
            tokens = self.tokenize(text)
            n_tokens = len(tokens)
            self.doc_lens.append(n_tokens)
            total_tokens += n_tokens

            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_tokens / max(1, self.corpus_size)

        # Precompute IDFs
        for term, df in self.doc_freqs.items():
            # Robertson-Spärck Jones IDF with floor of 0.0
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_cache[term] = max(0.001, idf)

    def get_idf(self, term: str) -> float:
        return self.idf_cache.get(term, 0.0)

    def search(self, query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], float]:
        """
        Scores all indexed documents against query terms and returns top-K results.
        """
        start_t = time.time()
        q_tokens = self.tokenize(query)
        if not q_tokens or self.corpus_size == 0:
            return [], 0.0

        scores = [0.0] * self.corpus_size
        k1 = self.k1
        b = self.b
        avgdl = max(1.0, self.avg_doc_len)

        for q in q_tokens:
            idf = self.get_idf(q)
            if idf <= 0.0:
                continue

            for idx in range(self.corpus_size):
                tf = self.term_freqs[idx].get(q, 0)
                if tf > 0:
                    doc_len = self.doc_lens[idx]
                    denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
                    scores[idx] += idf * (tf * (k1 + 1.0)) / denom

        # Sort descending
        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:min(top_k, self.corpus_size)]

        latency_ms = (time.time() - start_t) * 1000.0

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.doc_data[idx]
            results.append({
                "rank": rank + 1,
                "chunk_id": self.doc_ids[idx],
                "bm25_score": float(scores[idx]),
                "similarity_score": float(scores[idx]),  # standard interface compatibility
                "approx_token_count": self.doc_lens[idx],
                "doc_titles": doc.get("doc_titles", []),
                "text": doc.get("text", ""),
                "text_snippet": doc.get("text", "")[:250] + "..."
            })

        return results, latency_ms


class HybridRetriever:
    """
    Hybrid Retrieval combining Dense Bi-Encoder Vector Search and Sparse Lexical BM25
    via Reciprocal Rank Fusion (RRF) and normalized score interpolation.
    """
    def __init__(self, vector_store, chunks: List[Dict[str, Any]], rrf_c: int = 60):
        self.vector_store = vector_store
        self.chunks = chunks
        self.rrf_c = rrf_c
        self.bm25 = BM25Index(k1=1.5, b=0.75)
        self.bm25.fit(chunks, text_key="text", id_key="chunk_id")

    def search_hybrid(
        self,
        query: str,
        query_vector=None,
        top_k: int = 3,
        weight_dense: float = 0.5,
        use_rrf: bool = True
    ) -> Dict[str, Any]:
        """
        Executes both dense and BM25 search, then fuses the rankings.
        """
        start_t = time.time()

        # 1. Dense Search
        search_target = query_vector if query_vector is not None else query
        dense_results, dense_lat = self.vector_store.search(search_target, top_k=len(self.chunks))

        # 2. BM25 Search
        bm25_results, bm25_lat = self.bm25.search(query, top_k=len(self.chunks))

        # Create lookups by chunk_id
        dense_rank_map = {r["chunk_id"]: (r["rank"], r["similarity_score"]) for r in dense_results}
        bm25_rank_map = {r["chunk_id"]: (r["rank"], r["bm25_score"]) for r in bm25_results}
        chunk_obj_map = {c["chunk_id"]: c for c in self.chunks}

        # Min-max normalization for score interpolation
        max_dense = max([s for _, s in dense_rank_map.values()], default=1.0)
        min_dense = min([s for _, s in dense_rank_map.values()], default=0.0)
        range_dense = max(1e-5, max_dense - min_dense)

        max_bm25 = max([s for _, s in bm25_rank_map.values()], default=1.0)
        min_bm25 = min([s for _, s in bm25_rank_map.values()], default=0.0)
        range_bm25 = max(1e-5, max_bm25 - min_bm25)

        # 3. Compute Fusion Scores
        all_chunk_ids = list(chunk_obj_map.keys())
        fused_items = []

        for cid in all_chunk_ids:
            dense_rank, dense_raw = dense_rank_map.get(cid, (len(all_chunk_ids), 0.0))
            bm25_rank, bm25_raw = bm25_rank_map.get(cid, (len(all_chunk_ids), 0.0))

            dense_norm = (dense_raw - min_dense) / range_dense
            bm25_norm = (bm25_raw - min_bm25) / range_bm25

            # Reciprocal Rank Fusion formula:
            # RRF(d) = 1/(c + rank_dense) + 1/(c + rank_bm25)
            rrf_score = (1.0 / (self.rrf_c + dense_rank)) + (1.0 / (self.rrf_c + bm25_rank))
            
            # Weighted linear score
            linear_score = weight_dense * dense_norm + (1.0 - weight_dense) * bm25_norm

            final_score = rrf_score if use_rrf else linear_score

            chunk = chunk_obj_map[cid]
            fused_items.append({
                "chunk_id": cid,
                "score": final_score,
                "rrf_score": rrf_score,
                "linear_score": linear_score,
                "dense_rank": dense_rank,
                "dense_score": dense_raw,
                "bm25_rank": bm25_rank,
                "bm25_score": bm25_raw,
                "approx_token_count": chunk.get("approx_token_count", len(chunk.get("text", "")) // 4),
                "doc_titles": chunk.get("doc_titles", []),
                "text": chunk.get("text", ""),
                "text_snippet": chunk.get("text", "")[:250] + "..."
            })

        fused_items.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(fused_items):
            item["rank"] = i + 1
            item["similarity_score"] = item["score"]

        top_results = fused_items[:min(top_k, len(fused_items))]
        total_lat = (time.time() - start_t) * 1000.0

        return {
            "query": query,
            "fusion_mode": "RRF (Reciprocal Rank Fusion)" if use_rrf else "Weighted Normalized Linear",
            "weight_dense": weight_dense,
            "top_results": top_results,
            "dense_results": dense_results[:top_k],
            "bm25_results": bm25_results[:top_k],
            "latency_ms": {
                "dense_search": dense_lat,
                "bm25_search": bm25_lat,
                "fusion": total_lat - (dense_lat + bm25_lat),
                "total": total_lat
            }
        }
