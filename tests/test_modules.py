import unittest
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_loader import load_or_fetch_dataset
from src.chunker import create_long_chunks, slice_into_paragraphs, batch_slice_chunks
from src.vector_store import LongRAGVectorStore
from src.hyde import HyDEQueryExpander
from src.reranker import MiniLMReranker
from src.adaptive_filter import AdaptiveContextFilter
from src.pipeline import ELongRAGPipeline

class TestELongRAGModules(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n--- RUNNING UNIT TESTS FOR E-LONGRAG SYSTEM ---")
        cls.dataset = load_or_fetch_dataset(num_samples=10)
        cls.chunks = create_long_chunks(cls.dataset, target_tokens=1000, force_recreate=True)
        cls.vector_store = LongRAGVectorStore()
        cls.vector_store.build_index(cls.chunks, force_rebuild=True)
        cls.hyde = HyDEQueryExpander()
        cls.reranker = MiniLMReranker()
        cls.filter = AdaptiveContextFilter()
        cls.pipeline = ELongRAGPipeline(num_samples=10)

    def test_01_dataset_loader(self):
        self.assertGreater(len(self.dataset), 0)
        sample = self.dataset[0]
        self.assertIn("question", sample)
        self.assertIn("paragraphs", sample)
        print("  ✓ DatasetLoader loaded samples successfully.")

    def test_02_chunker_and_slicer(self):
        self.assertGreater(len(self.chunks), 0)
        chunk = self.chunks[0]
        self.assertIn("text", chunk)
        
        # Test Intra-Chunk Paragraph Slicing
        paragraphs = slice_into_paragraphs(chunk)
        self.assertGreater(len(paragraphs), 0)
        self.assertIn("paragraph_id", paragraphs[0])
        print(f"  ✓ Chunker created {len(self.chunks)} chunks; sliced Chunk 0 into {len(paragraphs)} paragraphs.")

    def test_03_vector_store_search(self):
        query = self.dataset[0]["question"]
        results, lat = self.vector_store.search(query, top_k=2)
        self.assertEqual(len(results), min(2, len(self.chunks)))
        self.assertIn("similarity_score", results[0])
        print(f"  ✓ VectorStore search executed in {lat:.2f}ms with Top Score {results[0]['similarity_score']:.4f}.")

    def test_04_hyde_expansion(self):
        query = "Who won the Turing Award for Distributed Systems?"
        hyde_res = self.hyde.get_hyde_embedding(query, self.vector_store)
        self.assertIn("hypothetical_document", hyde_res)
        self.assertIn("hyde_vector", hyde_res)
        self.assertEqual(hyde_res["hyde_vector"].shape[1], self.vector_store.dimension)
        print("  ✓ HyDE generated hypothetical passage and valid dense vector.")

    def test_05_minilm_reranker(self):
        query = "Who developed Lamport Timestamps?"
        paras = [
            {"paragraph_id": "p1", "text": "Leslie Lamport formulated Lamport Timestamps in 1978 for distributed systems."},
            {"paragraph_id": "p2", "text": "Gardening techniques in southern Italy focus on olive tree irrigation systems."}
        ]
        scored, lat = self.reranker.score_paragraphs(query, paras)
        self.assertEqual(len(scored), 2)
        self.assertGreater(scored[0]["relevance_score"], scored[1]["relevance_score"])
        self.assertEqual(scored[0]["paragraph_id"], "p1")
        print(f"  ✓ MiniLM Reranker successfully identified relevant paragraph (Score: {scored[0]['relevance_score']:.4f} > {scored[1]['relevance_score']:.4f}).")

    def test_06_adaptive_filter(self):
        paras = [
            {"paragraph_id": "p1", "text": "Leslie Lamport won the Turing Award in 2013.", "relevance_score": 0.92, "approx_token_count": 10},
            {"paragraph_id": "p2", "text": "Liquid nitrogen cooling systems.", "relevance_score": 0.12, "approx_token_count": 5}
        ]
        retained, pruned, text, stats = self.filter.filter_and_deduplicate(paras, self.vector_store)
        self.assertEqual(len(retained), 1)
        self.assertEqual(len(pruned), 1)
        self.assertGreater(stats["compression_ratio_pct"], 0.0)
        print(f"  ✓ AdaptiveContextFilter filtered distractor: Retained {len(retained)}, Pruned {len(pruned)} ({stats['compression_ratio_pct']}% Token Savings).")

    def test_07_pipeline_end_to_end(self):
        query = self.dataset[0]["question"]
        comp = self.pipeline.compare(query, top_k=2)
        self.assertIn("baseline", comp)
        self.assertIn("elongrag", comp)
        self.assertIn("comparison", comp)
        self.assertGreaterEqual(comp["comparison"]["token_savings_pct"], 0.0)
        print(f"  ✓ Full ELongRAGPipeline compared Baseline ({comp['comparison']['baseline_tokens']} tok) vs E-LongRAG ({comp['comparison']['elongrag_tokens']} tok).")

if __name__ == "__main__":
    unittest.main()
