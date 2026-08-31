import json
import os

DATA_DIR = "/home/pranav/.gemini/antigravity/scratch/e-longrag/data"
CHUNKS_FILE = os.path.join(DATA_DIR, "long_chunks.json")

# Target long chunk size: ~2,000 tokens (~8,000 characters) to create a multi-chunk index
TARGET_CHUNK_TOKENS = 2000
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * 4

def create_long_chunks(dataset):
    """
    LongRAG chunking strategy:
    Combines multiple document passages into long-context chunks (~2,000-4,000 tokens).
    """
    chunks = []
    current_text_parts = []
    current_titles = []
    current_char_count = 0
    chunk_id_counter = 0

    for sample in dataset:
        for p in sample["paragraphs"]:
            doc_str = f"Document: {p['title']}\n{p['text']}\n\n"
            current_text_parts.append(doc_str)
            current_titles.append(p['title'])
            current_char_count += len(doc_str)

            if current_char_count >= TARGET_CHUNK_CHARS:
                full_text = "".join(current_text_parts)
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id_counter}",
                    "doc_titles": list(set(current_titles)),
                    "text": full_text,
                    "char_count": len(full_text),
                    "approx_token_count": len(full_text) // 4
                })
                chunk_id_counter += 1
                current_text_parts = []
                current_titles = []
                current_char_count = 0

    if current_text_parts:
        full_text = "".join(current_text_parts)
        chunks.append({
            "chunk_id": f"chunk_{chunk_id_counter}",
            "doc_titles": list(set(current_titles)),
            "text": full_text,
            "char_count": len(full_text),
            "approx_token_count": len(full_text) // 4
        })

    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"[Chunker] Successfully generated {len(chunks)} long-context chunks (~{TARGET_CHUNK_TOKENS} tokens each).")
    return chunks

if __name__ == "__main__":
    from dataset_loader import load_or_fetch_sample_data
    ds = load_or_fetch_sample_data(300)
    c = create_long_chunks(ds)
    if c:
        print(f"Generated {len(c)} chunks. Sample Chunk 0: ID={c[0]['chunk_id']}, Tokens={c[0]['approx_token_count']}")
