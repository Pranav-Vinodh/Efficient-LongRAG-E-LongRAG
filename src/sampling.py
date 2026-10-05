import time
import numpy as np

class RetrievalSamplingStrategies:
    """
    Retrieval Sampling and Selection Strategies for E-LongRAG:
    1. Deterministic Top-K Sampling
    2. Nucleus (Top-p) Sampling over Cross-Encoder Softmax Distribution
    3. Boltzmann (Temperature-scaled) Stochastic Sampling
    4. Maximal Marginal Relevance (MMR) Balancing Relevance and Pairwise Diversity
    """

    @staticmethod
    def _softmax(scores, temperature=1.0):
        """
        Numerically stable temperature-scaled softmax.
        """
        temp = max(1e-5, float(temperature))
        scores_arr = np.array(scores, dtype=np.float64) / temp
        # Subtract max for numerical stability
        shifted = scores_arr - np.max(scores_arr)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    @classmethod
    def top_k_sampling(cls, scored_paragraphs, k=2):
        """
        Strategy 1: Deterministic Top-K.
        Selects the top-K highest scoring passages by cross-encoder relevance score.
        """
        if not scored_paragraphs:
            return []
        sorted_paras = sorted(scored_paragraphs, key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        selected = sorted_paras[:max(1, k)]
        # Tag selection method
        for rank, p in enumerate(selected, 1):
            p["selection_strategy"] = "Top-K"
            p["selection_rank"] = rank
        return selected

    @classmethod
    def nucleus_top_p_sampling(cls, scored_paragraphs, p=0.85, temperature=1.0):
        """
        Strategy 2: Nucleus (Top-p) Sampling.
        Applies temperature-scaled softmax over cross-encoder scores and accumulates
        probability mass until cumulative threshold p is reached.
        """
        if not scored_paragraphs:
            return []

        sorted_paras = sorted(scored_paragraphs, key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        scores = [p.get("relevance_score", 0.0) for p in sorted_paras]
        probs = cls._softmax(scores, temperature=temperature)

        cum_probs = np.cumsum(probs)
        selected = []

        for i, para in enumerate(sorted_paras):
            para_copy = dict(para)
            para_copy["selection_strategy"] = "Nucleus (Top-p)"
            para_copy["selection_rank"] = i + 1
            para_copy["softmax_prob"] = float(probs[i])
            para_copy["cumulative_prob"] = float(cum_probs[i])
            selected.append(para_copy)
            # Stop once cumulative mass threshold p is reached
            if cum_probs[i] >= p:
                break

        # Fallback guarantee: keep at least one
        if not selected and sorted_paras:
            p0 = dict(sorted_paras[0])
            p0["selection_strategy"] = "Nucleus (Top-p)"
            p0["softmax_prob"] = float(probs[0])
            selected.append(p0)

        return selected

    @classmethod
    def boltzmann_sampling(cls, scored_paragraphs, k=2, temperature=0.5, seed=None):
        """
        Strategy 3: Boltzmann (Softmax Temperature) Stochastic Sampling.
        Samples k distinct passages without replacement from the Boltzmann probability distribution:
            P(p_i) = exp(s_i / tau) / sum_j exp(s_j / tau)
        """
        if not scored_paragraphs:
            return []

        rng = np.random.RandomState(seed)
        n = len(scored_paragraphs)
        sample_size = min(max(1, k), n)

        scores = [p.get("relevance_score", 0.0) for p in scored_paragraphs]
        probs = cls._softmax(scores, temperature=temperature)

        # Sample without replacement
        chosen_indices = rng.choice(n, size=sample_size, replace=False, p=probs)
        
        selected = []
        for idx in chosen_indices:
            p_copy = dict(scored_paragraphs[idx])
            p_copy["selection_strategy"] = f"Boltzmann (T={temperature})"
            p_copy["softmax_prob"] = float(probs[idx])
            selected.append(p_copy)

        # Order selected passages by relevance score descending
        selected.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        for rank, p in enumerate(selected, 1):
            p["selection_rank"] = rank

        return selected

    @classmethod
    def maximal_marginal_relevance(cls, scored_paragraphs, vector_store, k=2, lambda_param=0.7):
        """
        Strategy 4: Maximal Marginal Relevance (MMR).
        Balances cross-encoder query relevance against pairwise dense embedding redundancy:
            MMR(p) = lambda * Rel(p, q) - (1 - lambda) * max_{p_j in S} CosSim(p, p_j)
        """
        if not scored_paragraphs:
            return []

        n = len(scored_paragraphs)
        target_k = min(max(1, k), n)

        # Precompute dense embeddings for all candidate paragraphs
        embeddings = []
        for p in scored_paragraphs:
            emb = vector_store.encode_text(p["text"]) # shape (1, d)
            # Normalize embedding
            norm = np.linalg.norm(emb)
            if norm > 1e-9:
                emb = emb / norm
            embeddings.append(emb)

        # Normalize relevance scores to [0, 1] range for fair weighting with cosine similarity
        raw_scores = [p.get("relevance_score", 0.0) for p in scored_paragraphs]
        min_s, max_s = min(raw_scores), max(raw_scores)
        if max_s > min_s:
            norm_rel = [(s - min_s) / (max_s - min_s) for s in raw_scores]
        else:
            norm_rel = [1.0] * n

        selected_indices = []
        unselected_indices = list(range(n))

        # First passage: select the highest relevance score
        first_idx = int(np.argmax(raw_scores))
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)

        # Greedily select remaining passages according to MMR criterion
        while len(selected_indices) < target_k and unselected_indices:
            best_mmr_score = -float("inf")
            best_idx = None

            for u_idx in unselected_indices:
                rel = norm_rel[u_idx]
                u_emb = embeddings[u_idx]

                # Compute maximum cosine similarity with already selected passages
                max_sim = max(float(np.dot(u_emb, embeddings[s_idx].T)[0][0]) for s_idx in selected_indices)
                
                # MMR formula
                mmr_val = (lambda_param * rel) - ((1.0 - lambda_param) * max_sim)

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = u_idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                unselected_indices.remove(best_idx)
            else:
                break

        selected = []
        for rank, s_idx in enumerate(selected_indices, 1):
            p_copy = dict(scored_paragraphs[s_idx])
            p_copy["selection_strategy"] = f"MMR (lambda={lambda_param})"
            p_copy["selection_rank"] = rank
            p_copy["norm_relevance"] = float(norm_rel[s_idx])
            selected.append(p_copy)

        return selected

    @classmethod
    def compute_context_metrics(cls, selected_paragraphs, candidate_paragraphs, vector_store):
        """
        Computes context compression, intra-context diversity, and relevance metrics.
        - Intra-context Diversity: Mean pairwise cosine distance (1 - cos_sim) between selected passages.
        """
        total_cand_tokens = sum(p.get("approx_token_count", len(p["text"]) // 4) for p in candidate_paragraphs)
        selected_tokens = sum(p.get("approx_token_count", len(p["text"]) // 4) for p in selected_paragraphs)

        token_savings_pct = 0.0
        if total_cand_tokens > 0:
            token_savings_pct = round((1.0 - (selected_tokens / total_cand_tokens)) * 100.0, 2)

        # Mean relevance
        mean_relevance = 0.0
        if selected_paragraphs:
            mean_relevance = float(np.mean([p.get("relevance_score", 0.0) for p in selected_paragraphs]))

        # Intra-context diversity
        intra_diversity = 0.0
        if len(selected_paragraphs) > 1:
            embs = []
            for p in selected_paragraphs:
                e = vector_store.encode_text(p["text"])
                norm = np.linalg.norm(e)
                if norm > 1e-9:
                    e = e / norm
                embs.append(e)

            distances = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    cos_sim = float(np.dot(embs[i], embs[j].T)[0][0])
                    cos_dist = max(0.0, 1.0 - cos_sim)
                    distances.append(cos_dist)

            if distances:
                intra_diversity = float(np.mean(distances))

        # Build context text
        context_parts = []
        for i, p in enumerate(selected_paragraphs):
            header = f"[{p.get('doc_titles', ['Section'])[0] if p.get('doc_titles') else f'Passage {i+1}'}]"
            context_parts.append(f"{header}\n{p['text']}")
        context_text = "\n\n".join(context_parts)

        return {
            "num_selected": len(selected_paragraphs),
            "retained_tokens": selected_tokens,
            "candidate_tokens": total_cand_tokens,
            "token_savings_pct": token_savings_pct,
            "mean_relevance": round(mean_relevance, 4),
            "intra_diversity": round(intra_diversity, 4),
            "context_text": context_text
        }
