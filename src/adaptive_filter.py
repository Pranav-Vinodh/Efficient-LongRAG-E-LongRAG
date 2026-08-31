import numpy as np
from .config import BASE_ADAPTIVE_THRESHOLD, DEDUP_SIMILARITY_THRESHOLD, ALPHA_VARIANCE

class AdaptiveContextFilter:
    """
    [Lab 4 Innovation: Query-Adaptive Dynamic Context Compression & Deduplication Engine]
    Dynamically adjusts context window size based on cross-encoder relevance distribution,
    removes redundant duplicate passages via cosine similarity, and builds high-density compact prompt.
    """
    def __init__(self, 
                 base_threshold=BASE_ADAPTIVE_THRESHOLD, 
                 dedup_threshold=DEDUP_SIMILARITY_THRESHOLD,
                 alpha=ALPHA_VARIANCE):
        self.base_threshold = base_threshold
        self.dedup_threshold = dedup_threshold
        self.alpha = alpha

    def compute_dynamic_threshold(self, scores):
        """
        Formulates adaptive threshold tau(q) = max(tau_min, mu_R - alpha * sigma_R)
        driven by cross-encoder score statistics for the specific query.
        """
        if not scores:
            return self.base_threshold

        scores_arr = np.array(scores, dtype=np.float32)
        mu = float(np.mean(scores_arr))
        sigma = float(np.std(scores_arr))

        # Dynamic threshold calculation
        adaptive_tau = max(self.base_threshold, mu - self.alpha * sigma)
        
        # High confidence cap: if top paragraph is extremely high (>0.85), tighten threshold
        if np.max(scores_arr) > 0.85:
            adaptive_tau = max(adaptive_tau, 0.50)

        return float(adaptive_tau)

    def filter_and_deduplicate(self, scored_paragraphs, vector_store):
        """
        Applies dynamic thresholding tau(q) and semantic cosine deduplication delta.
        Returns:
            - retained_paragraphs: list of high-signal paragraph objects
            - pruned_paragraphs: list of filtered distractor paragraph objects
            - compact_context_text: joined compact prompt text
            - stats: dictionary of compression and efficiency metrics
        """
        if not scored_paragraphs:
            return [], [], "", {"compression_ratio": 0.0, "retained_tokens": 0, "pruned_tokens": 0}

        scores = [p["relevance_score"] for p in scored_paragraphs]
        tau = self.compute_dynamic_threshold(scores)

        retained = []
        pruned = []
        retained_embeddings = []

        # Process in descending order of relevance score
        for p in scored_paragraphs:
            score = p["relevance_score"]
            
            # Rule 1: Relevance Score Check (R_{CE} >= tau(q))
            if score < tau:
                p_copy = dict(p)
                p_copy["prune_reason"] = f"Low Relevance (Score {score:.4f} < Threshold {tau:.4f})"
                pruned.append(p_copy)
                continue

            # Rule 2: Semantic Deduplication Check
            p_emb = vector_store.encode_text(p["text"]) # shape (1, d)
            
            is_duplicate = False
            max_sim = 0.0
            for r_emb in retained_embeddings:
                sim = float(np.dot(p_emb, r_emb.T)[0][0])
                if sim > max_sim:
                    max_sim = sim
                if sim >= self.dedup_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                p_copy = dict(p)
                p_copy["prune_reason"] = f"Semantic Duplicate (Cosine Sim {max_sim:.4f} >= {self.dedup_threshold:.2f})"
                pruned.append(p_copy)
            else:
                retained.append(p)
                retained_embeddings.append(p_emb)

        # Fallback guarantee: if all were pruned, keep at least the single best paragraph
        if not retained and scored_paragraphs:
            retained.append(scored_paragraphs[0])
            if pruned and pruned[0]["paragraph_id"] == scored_paragraphs[0]["paragraph_id"]:
                pruned.pop(0)

        # Construct compact context
        context_parts = []
        for i, p in enumerate(retained):
            title_header = f"[{p.get('doc_titles', ['Section'])[0] if p.get('doc_titles') else f'Passage {i+1}'}]"
            context_parts.append(f"{title_header}\n{p['text']}")

        compact_context_text = "\n\n".join(context_parts)

        # Compute compression statistics
        retained_tokens = sum(p.get("approx_token_count", len(p["text"]) // 4) for p in retained)
        pruned_tokens = sum(p.get("approx_token_count", len(p["text"]) // 4) for p in pruned)
        total_tokens = retained_tokens + pruned_tokens

        compression_ratio = 1.0 - (retained_tokens / max(1, total_tokens))
        
        # Estimated Time-To-First-Token (TTFT) speedup ratio
        ttft_speedup_factor = total_tokens / max(1, retained_tokens)

        stats = {
            "dynamic_threshold": tau,
            "total_candidate_paragraphs": len(scored_paragraphs),
            "retained_count": len(retained),
            "pruned_count": len(pruned),
            "original_tokens": total_tokens,
            "retained_tokens": retained_tokens,
            "pruned_tokens": pruned_tokens,
            "compression_ratio_pct": round(compression_ratio * 100.0, 2),
            "ttft_speedup_factor": round(ttft_speedup_factor, 2)
        }

        return retained, pruned, compact_context_text, stats
