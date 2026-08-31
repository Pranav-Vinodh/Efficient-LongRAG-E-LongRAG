# ⚡ Efficient LongRAG (E-LongRAG)

> **Adaptive Context Filtering for High-Precision Long-Context LLM Retrieval**  
> *Course: Large Language Models | Final Year Project | Midsem Evaluation*  
> **Authors**: Pranav Vinodh, Durga Sai Pavan Gangabattula, Sameer

---

## 📌 Executive Summary

Retrieval-Augmented Generation (RAG) paradigms face a fundamental trade-off:
* **Short-Chunk RAG ($100$--$500$ tokens)**: Low token cost, but fragments multi-hop reasoning across document boundaries.
* **Baseline LongRAG ($\approx 4,000$ tokens)** (*Jiang et al., arXiv:2410.18050*): Preserves document context, but causes **massive prompt bloat ($5,000+$ tokens)**, **high Time-To-First-Token (TTFT) pre-fill latency ($O(N^2)$ attention costs)**, and **"lost-in-the-middle" distractor hallucination**.

**Efficient LongRAG (E-LongRAG)** bridges this gap by combining coarse long-chunk retrieval with three lightweight post-retrieval neural and algorithmic optimization layers:
1. **HyDE Query Expansion**: Synthesizes a hypothetical answer document $\hat{d} \sim P(\cdot | q)$ to project short queries into dense document space.
2. **Intra-Chunk Paragraph Slicing & MiniLM Reranker**: Slices 4k chunks into $200$--$350$ token units and applies full joint cross-attention scoring ($R_{CE}$) via `cross-encoder/ms-marco-MiniLM-L-6-v2`, bypassing the 512-token BERT length limit.
3. **Query-Adaptive Context Filtering & Deduplication**: Dynamically calculates score cutoff $\tau(q) = \max(0.40, \mu_R - 0.5\sigma_R)$ and prunes redundant duplicate passages ($\delta = 0.85$).

---

## 📊 Empirical Ablation Benchmark Results

Evaluated across 25 multi-hop academic benchmark queries on multi-hop technical reasoning tasks:

| Model Configuration | Avg. Prompt Tokens | Retrieval Latency | Estimated LLM Pre-fill (TTFT) | Total Response Time | Context Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline LongRAG** | $5,134.0\text{ tok}$ | $10.58\text{ ms}$ | $\approx 2,000\text{ ms}$ | $\approx 2,010\text{ ms}$ | $50.0\%$ |
| **2. LongRAG + HyDE** | $5,123.0\text{ tok}$ | $14.65\text{ ms}$ | $\approx 2,000\text{ ms}$ | $\approx 2,015\text{ ms}$ | $65.0\%$ |
| **3. LongRAG + Reranker** | $5,123.0\text{ tok}$ | $453.30\text{ ms}$ | $\approx 2,000\text{ ms}$ | $\approx 2,453\text{ ms}$ | $80.0\%$ |
| **4. Full E-LongRAG** | **$260.8\text{ tok}$** | $472.41\text{ ms}$ | **$\approx 100\text{ ms}$** | **$\approx 570\text{ ms}$ (CPU) / $\approx 125\text{ ms}$ (GPU)** | **$95.0\%$** |

* **Prompt Compression**: **$94.9\%$ prompt token reduction** ($5,134 \rightarrow 261$ tokens).
* **Speedup**: **$19.7\times$ faster TTFT** during LLM attention pre-fill.
* **Precision**: **$+45.0\%$ precision gain** ($50\% \rightarrow 95\%$).

---

## 🏗️ System Architecture Flow

```
                     ┌───────────────────────────────┐
                     │         User Query q          │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 1. HyDE Query Expansion       │
                     │    Generates d_hat ~ P(q)     │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 2. FAISS Dense Vector Search  │
                     │    Retrieves Top-K 4k Chunks  │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 3. Intra-Chunk Slicer         │
                     │    Decomposes into {p_ij}     │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 4. MiniLM Cross-Encoder       │
                     │    Joint Attention Scoring    │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 5. Adaptive Dynamic Filter    │
                     │    Cutoff tau(q) & Dedup delta│
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ 6. Compact Prompt -> LLM      │
                     │    (260 tokens, 95% Precision)│
                     └───────────────────────────────┘
```

---

## 📁 Repository Structure

```
Efficient-LongRAG-E-LongRAG/
├── src/                                  # Core E-LongRAG Python package
│   ├── __init__.py                       # Package initialization
│   ├── config.py                         # Paths, models, and threshold hyperparameters
│   ├── dataset_loader.py                 # Multi-hop benchmark dataset ingestion & caching
│   ├── chunker.py                        # Long chunker & Intra-chunk paragraph slicer
│   ├── vector_store.py                   # FAISS IndexFlatIP + all-MiniLM-L6-v2 embeddings
│   ├── hyde.py                           # [Lab 3] HyDE query expansion module
│   ├── reranker.py                       # [Lab 2] MiniLM Cross-Encoder neural reranker
│   ├── adaptive_filter.py                # [Lab 4] Dynamic score filtering & deduplication
│   ├── pipeline.py                       # Unified E-LongRAG vs. Baseline orchestrator
│   └── evaluator.py                      # Quantitative benchmarking suite
├── tests/
│   ├── test_modules.py                   # Automated unit test suite (7/7 passing)
│   └── run_benchmark.py                  # 4-way ablation benchmark runner
├── reports/                              # Categorized lab reports & deliverables
│   ├── lab3_baseline_prototype/          # Baseline LongRAG report & LaTeX source
│   └── lab4_innovations/                 # Architectural innovations report & LaTeX source
├── plots/                                # Generated ablation benchmark figures (.png)
│   ├── token_compression_ablation.png
│   ├── precision_gain_ablation.png
│   ├── retrieval_latency_vs_topk.png
│   └── average_similarity_vs_topk.png
├── data/                                 # Cached dataset & FAISS binary index
├── .streamlit/                           # Streamlit configuration
│   └── config.toml                       # High-performance server settings
├── app.py                                # [DEMO] Streamlit Interactive Web Application
├── midsem_presentation.pdf               # 15-Slide Beamer Midsem Presentation (PDF)
├── midsem_presentation.tex               # LaTeX Beamer presentation source
├── requirements.txt                      # Python dependencies
├── .gitignore
└── README.md                             # Comprehensive documentation
```

---

## ⚡ Quick Start: Setup & Run on Any Machine

Follow these step-by-step instructions to clone, configure, and run the demo on **Linux, macOS, or Windows**.

### 1. Clone the Repository

```bash
git clone https://github.com/Pranav-Vinodh/Efficient-LongRAG-E-LongRAG.git
cd Efficient-LongRAG-E-LongRAG
```

---

### 2. Environment Setup

Choose **Option A (Conda)** or **Option B (Standard Virtualenv)**:

#### 🔹 Option A: Using Conda (Recommended)

```bash
# Create a fresh Python 3.10 environment
conda create -n elongrag python=3.10 -y

# Activate the environment
conda activate elongrag

# Install dependencies
pip install -r requirements.txt
```

#### 🔹 Option B: Using Python `venv`

```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Launch the Interactive Web Demo

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open your browser and navigate to:
👉 **`http://localhost:8501`**

#### 🎮 Interactive Features in the Demo:
* **Side-by-Side Comparison**: Run real multi-hop queries side-by-side (Baseline LongRAG vs E-LongRAG).
* **Live System Metrics**: Observe Prompt Token Savings ($94.9\%$), TTFT Latency Speedup ($19.7\times$), and Pruned Passages.
* **Raw Corpus Inspector (Tab 2)**: Read the complete uncompressed 4k-token base documents.
* **Noise Filter Inspector (Tab 3)**: Inspect extracted paragraphs with 🟢 `[+] RETAINED` and 🔴 `[-] PRUNED` badges and exact cross-encoder relevance scores.
* **Latency & Systems Callout**: Understand the computational offloading trade-off between lightweight cross-encoders and heavy LLM readers.

---

### 4. Running Unit Tests & Benchmarks

Run the unit test suite (verifies all modules end-to-end):
```bash
python tests/test_modules.py
```
```
Ran 7 tests in 29.610s
OK
```

Run the 4-way ablation benchmark and update plot figures:
```bash
python tests/run_benchmark.py
```

---

## 👥 Team & Individual Contributions

| Team Member | Module & Responsibilities |
| :--- | :--- |
| **Pranav Vinodh** | • **Intra-Chunk Reranker**: Hierarchical paragraph decomposition & MiniLM Cross-Encoder joint scoring (`src/reranker.py`).<br>• **Adaptive Context Filter**: Dynamic threshold algorithm $\tau(q)$ and semantic cosine deduplication (`src/adaptive_filter.py`).<br>• **Interactive Web App**: Complete Streamlit evaluation dashboard (`app.py`). |
| **Durga Sai Pavan Gangabattula** | • **HyDE Query Expansion**: Generative hypothetical document projection & embedding alignment (`src/hyde.py`).<br>• **Dataset Ingestion**: Multi-hop QA sample curation and structured caching (`src/dataset_loader.py`).<br>• **Unit Testing**: Test suite design and verification (`tests/test_modules.py`). |
| **Sameer** | • **Vector Store Engine**: 4k chunking & FAISS dense vector search index (`src/vector_store.py`).<br>• **Ablation Benchmarking**: Performance profiling & empirical ablation experiments (`src/evaluator.py`).<br>• **Literature & Baseline**: LongRAG baseline survey and token overhead analysis. |

---

## 📜 License

This project is developed for educational and research purposes under the **Apache-2.0 License**.
