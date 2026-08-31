import json
import os
import urllib.request

DATA_DIR = "/home/pranav/.gemini/antigravity/scratch/e-longrag/data"
OUTPUT_FILE = os.path.join(DATA_DIR, "sample_dataset.json")

def load_or_fetch_sample_data(num_samples=300):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            existing = json.load(f)
            if len(existing) >= num_samples:
                print(f"[DatasetLoader] Using existing sample dataset ({len(existing)} items) at {OUTPUT_FILE}")
                return existing[:num_samples]
            
    print(f"[DatasetLoader] Downloading sample HotpotQA dataset for {num_samples} samples...")
    url = "https://raw.githubusercontent.com/hotpotqa/hotpot/master/data/hotpot_dev_distractor_v1.json"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            raw_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"[DatasetLoader] Online download failed ({e}), generating {num_samples} synthetic multi-hop document samples...")
        raw_data = []

    processed_samples = []
    
    if raw_data:
        for idx, item in enumerate(raw_data[:num_samples]):
            question = item["question"]
            answer = item.get("answer", "")
            
            paragraphs = []
            for title, sentences in item["context"]:
                text = " ".join(sentences)
                paragraphs.append({"title": title, "text": text})
                
            processed_samples.append({
                "id": item["_id"],
                "question": question,
                "answer": answer,
                "paragraphs": paragraphs
            })
    else:
        for i in range(num_samples):
            processed_samples.append({
                "id": f"sample_{i}",
                "question": f"What is the historical significance of event #{i} in computer science?",
                "answer": f"Event #{i} established foundational paradigms in distributed systems.",
                "paragraphs": [
                    {
                        "title": f"Document {i}_A",
                        "text": f"Event #{i} was introduced in year {1970 + (i % 50)}. It revolutionized computational theory by introducing fault-tolerant consensus mechanisms. Key researchers included pioneer group {i}."
                    },
                    {
                        "title": f"Document {i}_B",
                        "text": f"Following event #{i}, industrial applications expanded rapidly across global data centers. Performance benchmarks recorded a 10x throughput increase when applying optimization framework {i}."
                    }
                ]
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(processed_samples, f, indent=2)
        
    print(f"[DatasetLoader] Successfully saved {len(processed_samples)} samples to {OUTPUT_FILE}")
    return processed_samples

if __name__ == "__main__":
    data = load_or_fetch_sample_data(300)
    print(f"Loaded {len(data)} items.")
