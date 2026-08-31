import time
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from .config import RERANKER_MODEL_NAME

class MiniLMReranker:
    """
    [Lab 2 Innovation: Paragraph-Level Cross-Encoder Reranking Layer]
    Jointly encodes query and paragraph pairs via full cross-attention transformer layers.
    Computes fine-grained semantic relevance scores: R_{CE}(q, p_{i,j}) in [0, 1].
    """
    def __init__(self, model_name=RERANKER_MODEL_NAME):
        print(f"[Reranker] Loading CrossEncoder model '{model_name}'...")
        # Check if CUDA GPU is available, otherwise uses CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CrossEncoder(model_name, max_length=512, device=device)
        except Exception as e:
            print(f"[Reranker] Warning: GPU init failed ({e}), falling back to CPU.")
            self.model = CrossEncoder(model_name, max_length=512, device="cpu")
            
        print(f"[Reranker] CrossEncoder initialized on device: {self.model.model.device}")

    def score_paragraphs(self, query, paragraphs):
        """
        Takes a query and a list of paragraph dicts (from intra-chunk slicing).
        Computes joint cross-encoder scores for all (query, paragraph) pairs.
        Returns paragraphs sorted by relevance score in descending order.
        """
        if not paragraphs:
            return [], 0.0

        start_t = time.time()
        pairs = [[query, p["text"]] for p in paragraphs]
        
        # CrossEncoder.predict returns raw logits or normalized scores
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Apply Sigmoid activation if scores are unbounded logits
        scores_arr = np.array(raw_scores, dtype=np.float32)
        if np.any(scores_arr < 0.0) or np.any(scores_arr > 1.0):
            # Sigmoid: 1 / (1 + exp(-x))
            probs = 1.0 / (1.0 + np.exp(-scores_arr))
        else:
            probs = scores_arr

        latency_ms = (time.time() - start_t) * 1000.0

        scored_paragraphs = []
        for i, p in enumerate(paragraphs):
            p_copy = dict(p)
            p_copy["relevance_score"] = float(probs[i])
            scored_paragraphs.append(p_copy)

        # Sort descending by relevance score
        scored_paragraphs.sort(key=lambda x: x["relevance_score"], reverse=True)

        for rank, p in enumerate(scored_paragraphs):
            p["rerank_position"] = rank + 1

        return scored_paragraphs, latency_ms

if __name__ == "__main__":
    reranker = MiniLMReranker()
    q = "Who received the Turing Award for Lamport Timestamps?"
    sample_paras = [
        {"paragraph_id": "p0", "text": "Leslie Lamport introduced Lamport Timestamps in 1978 and won the 2013 Turing Award."},
        {"paragraph_id": "p1", "text": "Hardware cooling systems require liquid nitrogen circulation in supercomputers."}
    ]
    res, lat = reranker.score_paragraphs(q, sample_paras)
    print(f"Reranking latency: {lat:.2f} ms")
    for p in res:
        print(f"[{p['paragraph_id']}] Score: {p['relevance_score']:.4f} | Text: {p['text']}")
