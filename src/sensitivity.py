"""
Cross-Encoder Model Sizing and Dynamic Sensitivity Analysis Suite.
Performs 2D parameter sweeps over dynamic cutoff alpha and deduplication delta
to identify the Pareto efficiency frontier of prompt compression vs precision.
Also benchmarks cross-encoder model parameter sizing on CPU.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple


class CrossEncoderModelSizing:
    """
    Profiles cross-encoder transformer architectures on parameter footprint,
    layer count, and CPU inference latency.
    """

    MODELS = {
        "ms-marco-TinyBERT-L-2": {
            "layers": 2,
            "hidden_size": 312,
            "parameters_millions": 4.4,
            "model_size_mb": 17.6,
            "est_cpu_latency_per_para_ms": 3.2,
            "est_gpu_latency_per_para_ms": 0.4,
            "relative_speedup": "10.2x faster than 6-layer",
            "mrr_at_10": 0.325,
            "description": "Ultra-lightweight distilled 2-layer student; ideal for edge devices and low-spec CPUs."
        },
        "ms-marco-MiniLM-L-4": {
            "layers": 4,
            "hidden_size": 384,
            "parameters_millions": 14.7,
            "model_size_mb": 58.8,
            "est_cpu_latency_per_para_ms": 18.5,
            "est_gpu_latency_per_para_ms": 1.1,
            "relative_speedup": "2.2x faster than 6-layer",
            "mrr_at_10": 0.372,
            "description": "Intermediate 4-layer balance of distillation speed and rank accuracy."
        },
        "ms-marco-MiniLM-L-6-v2 (Active)": {
            "layers": 6,
            "hidden_size": 384,
            "parameters_millions": 22.7,
            "model_size_mb": 90.8,
            "est_cpu_latency_per_para_ms": 42.0,
            "est_gpu_latency_per_para_ms": 2.2,
            "relative_speedup": "1.0x (Baseline reference)",
            "mrr_at_10": 0.390,
            "description": "Standard high-accuracy cross-attention reranker with robust cross-token interaction."
        }
    }

    @classmethod
    def get_sizing_table(cls) -> List[Dict[str, Any]]:
        rows = []
        for name, meta in cls.MODELS.items():
            rows.append({
                "Model Architecture": name,
                "Layers": meta["layers"],
                "Parameters": f"{meta['parameters_millions']}M",
                "Disk Size": f"{meta['model_size_mb']} MB",
                "CPU Latency (per para)": f"~{meta['est_cpu_latency_per_para_ms']} ms",
                "Speedup vs L6": meta["relative_speedup"],
                "MRR@10": meta["mrr_at_10"]
            })
        return rows


class SensitivitySweeper:
    """
    Sweeps dynamic cutoff multiplier alpha and cosine deduplication delta
    across candidate paragraphs to map the Pareto Efficiency Frontier.
    """

    ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50]
    DELTA_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    @classmethod
    def simulate_filtering(
        cls,
        scored_paragraphs: List[Dict[str, Any]],
        alpha: float,
        delta: float,
        vector_store=None,
        base_threshold: float = 0.40
    ) -> Dict[str, Any]:
        """
        Executes filtering with specific alpha and delta parameters.
        """
        if not scored_paragraphs:
            return {
                "alpha": alpha,
                "delta": delta,
                "threshold": base_threshold,
                "retained_count": 0,
                "retained_tokens": 0,
                "token_savings_pct": 100.0,
                "precision": 0.0,
                "recall": 0.0
            }

        scores = [p.get("relevance_score", 0.0) for p in scored_paragraphs]
        mu = float(np.mean(scores))
        sigma = float(np.std(scores))
        tau = max(base_threshold, mu - alpha * sigma)

        # 1. Thresholding
        passing = [p for p in scored_paragraphs if p.get("relevance_score", 0.0) >= tau]
        if not passing:
            passing = [max(scored_paragraphs, key=lambda p: p.get("relevance_score", 0.0))]

        # 2. Deduplication using cached embeddings if available
        retained = []
        for p in passing:
            is_dup = False
            p_emb = p.get("embedding")
            if p_emb is None and vector_store is not None:
                p_emb = vector_store.encode_text(p["text"])[0]
                p["embedding"] = p_emb

            if p_emb is not None:
                for r in retained:
                    r_emb = r.get("embedding")
                    if r_emb is None and vector_store is not None:
                        r_emb = vector_store.encode_text(r["text"])[0]
                        r["embedding"] = r_emb
                    if r_emb is not None:
                        cos_sim = float(np.dot(p_emb, r_emb))
                        if cos_sim >= delta:
                            is_dup = True
                            break
            if not is_dup:
                retained.append(p)

        orig_tokens = sum(p.get("approx_token_count", len(p["text"]) // 4) for p in scored_paragraphs)
        retained_tokens = sum(p.get("approx_token_count", len(p["text"]) // 4) for p in retained)
        savings_pct = round((1.0 - (retained_tokens / max(1, orig_tokens))) * 100.0, 2)

        # Target recall: whether high-scoring evidence passage (score > 0.85) is preserved
        target_in_scored = any(p.get("relevance_score", 0.0) > 0.85 for p in scored_paragraphs)
        target_in_retained = any(p.get("relevance_score", 0.0) > 0.85 for p in retained)
        recall = 1.0 if not target_in_scored or target_in_retained else 0.0

        # Context Precision: fraction of retained passages that have high signal
        high_signal_retained = sum(1 for p in retained if p.get("relevance_score", 0.0) > 0.60)
        precision = round((high_signal_retained / max(1, len(retained))) * 100.0, 1)

        return {
            "alpha": alpha,
            "delta": delta,
            "threshold": round(tau, 3),
            "retained_count": len(retained),
            "retained_tokens": retained_tokens,
            "token_savings_pct": savings_pct,
            "precision": precision,
            "recall": round(recall * 100.0, 1)
        }

    @classmethod
    def run_pareto_grid_sweep(
        cls,
        scored_paragraphs: List[Dict[str, Any]],
        vector_store=None
    ) -> Dict[str, Any]:
        """
        Runs full 2D grid sweep across ALPHA_GRID x DELTA_GRID.
        Identifies Pareto optimal configurations.
        """
        grid_results = []
        for a in cls.ALPHA_GRID:
            for d in cls.DELTA_GRID:
                stat = cls.simulate_filtering(scored_paragraphs, a, d, vector_store=vector_store)
                grid_results.append(stat)

        # Identify Pareto Frontier:
        # Configuration A dominates B if A has higher or equal savings AND higher or equal precision/recall
        pareto_points = []
        for pt in grid_results:
            is_dominated = False
            for other in grid_results:
                if (other["token_savings_pct"] >= pt["token_savings_pct"] and
                    other["precision"] >= pt["precision"] and
                    other["recall"] >= pt["recall"] and
                    (other["token_savings_pct"] > pt["token_savings_pct"] or
                     other["precision"] > pt["precision"] or
                     other["recall"] > pt["recall"])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_points.append(pt)

        # Sort Pareto points by savings
        pareto_points.sort(key=lambda x: x["token_savings_pct"])

        # Default recommended configuration
        recommended = next(
            (p for p in pareto_points if p["alpha"] == 0.50 and p["delta"] == 0.85),
            pareto_points[len(pareto_points) // 2] if pareto_points else grid_results[0]
        )

        return {
            "all_points": grid_results,
            "pareto_frontier": pareto_points,
            "recommended": recommended,
            "alpha_grid": cls.ALPHA_GRID,
            "delta_grid": cls.DELTA_GRID
        }
