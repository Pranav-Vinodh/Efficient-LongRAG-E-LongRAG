import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = "/home/pranav/.gemini/antigravity/scratch/e-longrag/plots"
DATA_DIR = "/home/pranav/.gemini/antigravity/scratch/e-longrag/data"

def run_topk_tuning_experiment():
    from dataset_loader import load_or_fetch_sample_data
    from chunker import create_long_chunks
    from vector_store import LongRAGVectorStore

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load data & create long chunks
    dataset = load_or_fetch_sample_data(100)
    chunks = create_long_chunks(dataset)

    # 2. Build FAISS index
    vs = LongRAGVectorStore()
    vs.build_index(chunks)

    top_k_values = [1, 3, 5, 10]
    latency_results = {k: [] for k in top_k_values}
    similarity_results = {k: [] for k in top_k_values}

    print(f"\n[Experiment] Running Top-K tuning benchmark over {len(dataset)} queries...")

    for sample in dataset:
        query = sample["question"]
        for k in top_k_values:
            results, latency_ms = vs.search(query, top_k=k)
            latency_results[k].append(latency_ms)
            
            avg_sim = np.mean([r["similarity_score"] for r in results]) if results else 0.0
            similarity_results[k].append(avg_sim)

    # Calculate statistics
    mean_latencies = [np.mean(latency_results[k]) for k in top_k_values]
    mean_similarities = [np.mean(similarity_results[k]) for k in top_k_values]

    print("\n" + "="*50)
    print("      LAB 1: TOP-K RETRIEVAL TUNING RESULTS       ")
    print("="*50)
    print(f"{'Top-K Value':<12} | {'Mean Latency (ms)':<20} | {'Mean Similarity Score':<22}")
    print("-" * 58)
    for k, lat, sim in zip(top_k_values, mean_latencies, mean_similarities):
        print(f"K = {k:<9} | {lat:<20.2f} | {sim:<22.4f}")
    print("="*50 + "\n")

    # Plot 1: Retrieval Latency vs Top-K
    plt.figure(figsize=(7, 5))
    bars = plt.bar([str(k) for k in top_k_values], mean_latencies, color="#3498db", width=0.5, edgecolor="black", linewidth=1.2)
    plt.xlabel("Top-K Retrieved Chunks", fontsize=12, fontweight="bold")
    plt.ylabel("Retrieval Latency (ms)", fontsize=12, fontweight="bold")
    plt.title("Lab 1: Retrieval Latency vs Top-K Chunks", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Value annotations on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "retrieval_latency_vs_topk.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[Experiment] Saved Latency Plot to {plot1_path}")

    # Plot 2: Average Similarity Score vs Top-K
    plt.figure(figsize=(7, 5))
    plt.plot([str(k) for k in top_k_values], mean_similarities, marker="o", color="#e74c3c", linewidth=2.5, markersize=8, label="Cosine Similarity")
    plt.xlabel("Top-K Retrieved Chunks", fontsize=12, fontweight="bold")
    plt.ylabel("Average Cosine Similarity Score", fontsize=12, fontweight="bold")
    plt.title("Lab 1: Average Similarity Score vs Top-K", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    for i, txt in enumerate(mean_similarities):
        plt.annotate(f"{txt:.4f}", (str(top_k_values[i]), mean_similarities[i]), textcoords="offset points", xytext=(0,10), ha="center", fontweight="bold")

    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "average_similarity_vs_topk.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[Experiment] Saved Similarity Plot to {plot2_path}")

    return {
        "top_k_values": top_k_values,
        "mean_latencies": mean_latencies,
        "mean_similarities": mean_similarities
    }

if __name__ == "__main__":
    run_topk_tuning_experiment()
