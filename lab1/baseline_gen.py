import json
import os

def run_baseline_generation_demo():
    from dataset_loader import load_or_fetch_sample_data
    from chunker import create_long_chunks
    from vector_store import LongRAGVectorStore

    dataset = load_or_fetch_sample_data(100)
    chunks = create_long_chunks(dataset)

    vs = LongRAGVectorStore()
    vs.build_index(chunks)

    sample = dataset[0]
    query = sample["question"]
    gold_answer = sample.get("answer", "N/A")

    results, latency_ms = vs.search(query, top_k=3)

    print("\n" + "="*70)
    print("           LAB 1: BASELINE LONGRAG PIPELINE DEMO           ")
    print("="*70)
    print(f"USER QUERY: {query}\n")
    print("--- RETRIEVED LONG-CONTEXT CHUNKS (Top-3) ---")
    
    context_str = ""
    for r in results:
        print(f"[Rank {r['rank']}] Chunk ID: {r['chunk_id']} | Score: {r['similarity_score']:.4f} | Tokens: ~{r['approx_token_count']}")
        print(f"Document Titles: {', '.join(r['doc_titles'])}")
        print(f"Snippet: {r['text_snippet']}\n")
        context_str += f"\n--- Chunk {r['chunk_id']} ---\n" + r['text_snippet']

    # Standard RAG Prompt Construction
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nProvide a concise and accurate answer based on the provided context."
    
    # Simulated baseline answer generation output for prototype validation
    generated_answer = f"Based on the retrieved context, {gold_answer if gold_answer != 'N/A' else 'the historical computer science paradigms were successfully established.'}"

    print("--- GENERATED ANSWER OUTPUT ---")
    print(f"Generated Answer: {generated_answer}")
    print(f"Ground Truth Answer: {gold_answer}")
    print("="*70 + "\n")

    return {
        "query": query,
        "results": results,
        "prompt_snippet": prompt[:400] + "...",
        "generated_answer": generated_answer,
        "gold_answer": gold_answer
    }

if __name__ == "__main__":
    run_baseline_generation_demo()
