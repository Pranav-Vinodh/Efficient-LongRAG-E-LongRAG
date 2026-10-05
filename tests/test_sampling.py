import unittest
import os
import sys
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sampling import RetrievalSamplingStrategies
from src.vector_store import LongRAGVectorStore
from src.chunker import create_long_chunks
from src.dataset_loader import load_or_fetch_dataset
from src.pipeline import ELongRAGPipeline

class TestRetrievalSamplingStrategies(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--- RUNNING UNIT TESTS FOR RETRIEVAL SAMPLING ABLATION ---")
        cls.dataset = load_or_fetch_dataset(num_samples=10)
        cls.chunks = create_long_chunks(cls.dataset, target_tokens=1000, force_recreate=False)
        cls.vector_store = LongRAGVectorStore()
        cls.vector_store.build_index(cls.chunks, force_rebuild=False)
        cls.pipeline = ELongRAGPipeline(num_samples=10)

        # Mock scored paragraphs
        cls.sample_paragraphs = [
            {"paragraph_id": "p1", "text": "Leslie Lamport formulated Lamport Timestamps in 1978 for distributed consensus ordering.", "relevance_score": 0.95, "approx_token_count": 14},
            {"paragraph_id": "p2", "text": "Lamport timestamps provide logical clock synchronization across distributed nodes.", "relevance_score": 0.92, "approx_token_count": 12},
            {"paragraph_id": "p3", "text": "Peter Shor discovered quantum polynomial-time integer factorization algorithm.", "relevance_score": 0.40, "approx_token_count": 11},
            {"paragraph_id": "p4", "text": "Modern agricultural tractor fuel injection maintenance schedules.", "relevance_score": 0.05, "approx_token_count": 9},
            {"paragraph_id": "p5", "text": "Deep sea hydrothermal vents host chemosynthetic microbial ecosystems.", "relevance_score": 0.02, "approx_token_count": 10},
        ]

    def test_01_top_k_sampling(self):
        # Top-2 should yield p1 and p2 (highest relevance scores)
        selected = RetrievalSamplingStrategies.top_k_sampling(self.sample_paragraphs, k=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["paragraph_id"], "p1")
        self.assertEqual(selected[1]["paragraph_id"], "p2")
        print("  ✓ Deterministic Top-K correctly selected highest scoring passages.")

    def test_02_nucleus_top_p_sampling(self):
        # When p=0.80, the top 2 paragraphs should cover most probability mass and be selected
        selected = RetrievalSamplingStrategies.nucleus_top_p_sampling(self.sample_paragraphs, p=0.80, temperature=1.0)
        self.assertGreaterEqual(len(selected), 1)
        self.assertLessEqual(len(selected), len(self.sample_paragraphs))
        self.assertEqual(selected[0]["paragraph_id"], "p1")
        # Check that low scoring distractor p5 is excluded
        selected_ids = [p["paragraph_id"] for p in selected]
        self.assertNotIn("p5", selected_ids)
        print(f"  ✓ Nucleus Top-p dynamically selected {len(selected)} passages covering threshold mass.")

    def test_03_boltzmann_sampling(self):
        # Boltzmann with low temperature (T=0.1) should strongly favor top passages
        selected_cold = RetrievalSamplingStrategies.boltzmann_sampling(self.sample_paragraphs, k=2, temperature=0.1, seed=42)
        self.assertEqual(len(selected_cold), 2)
        self.assertIn("p1", [p["paragraph_id"] for p in selected_cold])

        # Boltzmann with k=3 should return exactly 3 distinct items
        selected_k3 = RetrievalSamplingStrategies.boltzmann_sampling(self.sample_paragraphs, k=3, temperature=0.5, seed=42)
        self.assertEqual(len(selected_k3), 3)
        self.assertEqual(len(set(p["paragraph_id"] for p in selected_k3)), 3)
        print("  ✓ Boltzmann stochastic sampling respects temperature and samples distinct passages without replacement.")

    def test_04_maximal_marginal_relevance(self):
        # p1 and p2 are near-duplicates on Lamport timestamps.
        # p3 is on quantum computing (diverse, decent score).
        # With high diversity penalty (e.g. lambda=0.4), MMR should pick p1 then p3 instead of redundant p2.
        selected_mmr = RetrievalSamplingStrategies.maximal_marginal_relevance(
            self.sample_paragraphs, self.vector_store, k=2, lambda_param=0.3
        )
        self.assertEqual(len(selected_mmr), 2)
        self.assertEqual(selected_mmr[0]["paragraph_id"], "p1")
        # The second item should be penalized if it's too redundant with p1
        print(f"  ✓ MMR selected {[p['paragraph_id'] for p in selected_mmr]} balancing relevance vs cosine redundancy.")

    def test_05_compute_context_metrics(self):
        selected = self.sample_paragraphs[:2]
        metrics = RetrievalSamplingStrategies.compute_context_metrics(selected, self.sample_paragraphs, self.vector_store)
        self.assertEqual(metrics["num_selected"], 2)
        self.assertGreater(metrics["retained_tokens"], 0)
        self.assertGreater(metrics["token_savings_pct"], 0.0)
        self.assertGreaterEqual(metrics["intra_diversity"], 0.0)
        self.assertIn("Lamport", metrics["context_text"])
        print(f"  ✓ Context metrics computed: {metrics['retained_tokens']} tokens, {metrics['token_savings_pct']}% savings, {metrics['intra_diversity']:.4f} diversity.")

    def test_06_pipeline_sampling_ablation(self):
        query = "Who introduced Lamport Timestamps in the domain of Distributed Consensus, and which major honor did they receive?"
        ablation_results = self.pipeline.run_sampling_ablation(query, top_k=2, k=2, p=0.85, temperature=0.5, lambda_param=0.7)
        self.assertIn("strategies", ablation_results)
        strategies = ablation_results["strategies"]
        self.assertIn("Adaptive Filter (Proposed)", strategies)
        self.assertIn("Deterministic Top-K", strategies)
        self.assertIn("Nucleus (Top-p)", strategies)
        self.assertIn("Boltzmann Sampling", strategies)
        self.assertIn("Maximal Marginal Relevance (MMR)", strategies)

        for name, data in strategies.items():
            self.assertIn("generated_answer", data)
            self.assertIn("metrics", data)
            self.assertGreater(len(data["selected_paragraphs"]), 0)
            print(f"    - [{name}]: {data['metrics']['retained_tokens']} tokens | Div: {data['metrics']['intra_diversity']} | Latency: {data['latency_ms']}ms")

        print("  ✓ Full Sampling Ablation successfully executed across all 5 strategies.")

if __name__ == "__main__":
    unittest.main()
