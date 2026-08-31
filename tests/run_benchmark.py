import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ELongRAGPipeline
from src.evaluator import PipelineEvaluator

def main():
    print("="*70)
    print("      RUNNING E-LONGRAG MIDSEM ABLATION & EVALUATION BENCHMARK     ")
    print("="*70)
    
    # Initialize pipeline with 50 multi-hop benchmark samples
    pipeline = ELongRAGPipeline(num_samples=50, force_rebuild=True)
    
    # Initialize evaluator and run benchmark
    evaluator = PipelineEvaluator(pipeline)
    results = evaluator.run_ablation_benchmark(num_queries=25)
    
    print("\nBenchmark completed successfully! Generated ablation plots in 'plots/'.")

if __name__ == "__main__":
    main()
