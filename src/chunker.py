import json
import os
import re
from .config import DATA_DIR, LONG_CHUNKS_PATH, TARGET_LONG_CHUNK_TOKENS, TARGET_PARAGRAPH_TOKENS

def create_long_chunks(dataset, target_tokens=TARGET_LONG_CHUNK_TOKENS, force_recreate=False):
    """
    Implements the LongRAG long-context chunking strategy:
    Combines multiple document passages into long chunks (~2,000-4,000 tokens).
    """
    if os.path.exists(LONG_CHUNKS_PATH) and not force_recreate:
        with open(LONG_CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            if chunks:
                return chunks

    target_chars = target_tokens * 4
    chunks = []
    current_text_parts = []
    current_titles = []
    current_char_count = 0
    chunk_id_counter = 0

    for sample in dataset:
        for p in sample.get("paragraphs", []):
            doc_str = f"Document: {p['title']}\n{p['text']}\n\n"
            current_text_parts.append(doc_str)
            current_titles.append(p['title'])
            current_char_count += len(doc_str)

            if current_char_count >= target_chars:
                full_text = "".join(current_text_parts)
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id_counter}",
                    "doc_titles": list(set(current_titles)),
                    "text": full_text,
                    "char_count": len(full_text),
                    "approx_token_count": max(1, len(full_text) // 4)
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
            "approx_token_count": max(1, len(full_text) // 4)
        })

    with open(LONG_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"[Chunker] Successfully generated {len(chunks)} long chunks (~{target_tokens} tokens each).")
    return chunks

def slice_into_paragraphs(long_chunk, target_paragraph_tokens=TARGET_PARAGRAPH_TOKENS):
    """
    [Lab 2 Innovation: Intra-Chunk Paragraph Slicing]
    Decomposes a 2k-4k long-context chunk into fine-grained semantic paragraphs (200-350 tokens).
    Overcomes the 512-token sequence limit of Cross-Encoders and enables selective noise filtering.
    """
    text = long_chunk.get("text", "")
    parent_chunk_id = long_chunk.get("chunk_id", "unknown_chunk")
    parent_titles = long_chunk.get("doc_titles", [])

    # Split by document headers or double newlines
    raw_blocks = re.split(r'\n\s*\n', text)
    paragraphs = []
    curr_block = []
    curr_char_count = 0
    p_counter = 0

    target_paragraph_chars = target_paragraph_tokens * 4

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        curr_block.append(block)
        curr_char_count += len(block)

        if curr_char_count >= target_paragraph_chars:
            p_text = "\n\n".join(curr_block)
            paragraphs.append({
                "paragraph_id": f"{parent_chunk_id}_p{p_counter}",
                "parent_chunk_id": parent_chunk_id,
                "text": p_text,
                "char_count": len(p_text),
                "approx_token_count": max(1, len(p_text) // 4),
                "doc_titles": parent_titles
            })
            p_counter += 1
            curr_block = []
            curr_char_count = 0

    if curr_block:
        p_text = "\n\n".join(curr_block)
        paragraphs.append({
            "paragraph_id": f"{parent_chunk_id}_p{p_counter}",
            "parent_chunk_id": parent_chunk_id,
            "text": p_text,
            "char_count": len(p_text),
            "approx_token_count": max(1, len(p_text) // 4),
            "doc_titles": parent_titles
        })

    return paragraphs

def batch_slice_chunks(long_chunks):
    """
    Takes a list of long chunks and returns a flat list of all parsed paragraph objects.
    """
    all_paragraphs = []
    for chunk in long_chunks:
        paras = slice_into_paragraphs(chunk)
        all_paragraphs.extend(paras)
    return all_paragraphs

if __name__ == "__main__":
    from .dataset_loader import load_or_fetch_dataset
    ds = load_or_fetch_dataset(50)
    c = create_long_chunks(ds, force_recreate=True)
    p = slice_into_paragraphs(c[0])
    print(f"Chunk 0 tokens: {c[0]['approx_token_count']} -> Parsed into {len(p)} sub-paragraphs.")
