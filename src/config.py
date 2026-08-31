import os

# Project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# File paths
SAMPLE_DATASET_PATH = os.path.join(DATA_DIR, "sample_dataset.json")
LONG_CHUNKS_PATH = os.path.join(DATA_DIR, "long_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# Model configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking & Filtering Hyperparameters
TARGET_LONG_CHUNK_TOKENS = 2000     # ~8,000 characters
TARGET_PARAGRAPH_TOKENS = 250       # ~1,000 characters
BASE_ADAPTIVE_THRESHOLD = 0.40      # Minimum cross-encoder confidence cutoff
DEDUP_SIMILARITY_THRESHOLD = 0.85   # Cosine similarity deduplication threshold
ALPHA_VARIANCE = 0.5                # Adaptive standard deviation multiplier

# Ensure essential directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
