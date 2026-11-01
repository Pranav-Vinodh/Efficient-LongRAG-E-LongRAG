# ⚡ Efficient LongRAG (E-LongRAG)

> **Adaptive Context Filtering, Neural Paragraph Reranking & Multi-Metric Evaluation for Long-Context LLMs**  
> _Course: Large Language Models (IT 488) | Final Endsemester Project Evaluation_  
> **Authors**: Pranav Vinodh, Durga Sai Pavan Gangabattula, Sameer  
> **Institution**: Department of Information Technology, National Institute of Technology Karnataka (NITK), Surathkal

---

## 📌 Executive Summary

Retrieval-Augmented Generation (RAG) paradigms face a fundamental structural trade-off:

- **Short-Chunk RAG ($100$--$500$ tokens)**: Minimizes token cost, but fragments multi-hop reasoning across arbitrary document cutoffs, reducing multi-hop retrieval recall below $60\%$.
- **Baseline LongRAG ($\approx 4,000$ tokens)** (_Jiang et al., 2024_): Preserves multi-document narrative continuity, but introduces **massive prompt bloat ($5,000+$ tokens)**, **quadratic Time-To-First-Token (TTFT) pre-fill latency ($O(N^2)$ self-attention)**, and **"lost-in-the-middle" attention collapse** (recall drops to $35.2\%$ at context center).

**Efficient LongRAG (E-LongRAG)** resolves this dilemma by coupling coarse $4,000$-token retrieval with lightweight post-retrieval neural and algorithmic optimization layers:

1. **HyDE Query Expansion**: Generates hypothetical answer documents $\hat{d} \sim P(\cdot \mid q)$ to eliminate lexical-semantic mismatches.
2. **Hybrid Dense-Lexical Search**: Fuses FAISS dense cosine similarity with pure-Python Okapi BM25 via Reciprocal Rank Fusion ($\text{RRF}, c=60$).
3. **Intra-Chunk Paragraph Slicing**: Decomposes 4k chunks into $200$--$350$ token sub-paragraphs, bypassing the 512-token sequence limit of BERT-based models.
4. **MiniLM Cross-Attention Reranking**: Evaluates fine-grained joint query-paragraph representations via `cross-encoder/ms-marco-MiniLM-L-6-v2`.
5. **Query-Adaptive Context Filtering & Deduplication**: Statistically computes dynamic cutoff thresholds $\tau(q) = \max(0.40, \mu_R - 0.5\sigma_R)$ and prunes semantic duplicates ($\delta = 0.85$).
6. **Retrieval Sampling Strategies Suite**: Evaluates Top-$K$, Nucleus Top-$p$, Boltzmann, MMR, and Adaptive Filtering with intra-context diversity tracking.
7. **Positional Invariance**: Eliminates lost-in-the-middle attention sinks, maintaining **100.0\% position-invariant recall** across 10k-token multi-document contexts.
8. **Automated RAGAS Quality Evaluation**: Elevates Context Precision from $24.1\%$ to $94.2\%$ ($+290.9\%$ signal gain).
9. **Multi-Reader Engine**: Delivers sub-5ms local zero-GPU CPU extraction while remaining fully compatible with Google Gemini 1.5 Flash and Groq LPUs.

---

## 📑 Project Deliverables

| Deliverable | File Path | Format / Details |
| :--- | :--- | :--- |
| **Final IEEE Endsem Report** | [**`E-LongRAG_Endsem_Report.pdf`**](./E-LongRAG_Endsem_Report.pdf) | 8-Page IEEE Computer Society Report (`reports/endsem_project_report/`) |
| **Final Endsem Presentation Deck** | [**`proj-final-eval.pdf`**](./proj-final-eval.pdf) | 20-Slide Beamer Deck (`proj-final-eval.tex`) |
| **Midsem Presentation Deck** | [**`proj-mid-eval.pdf`**](./proj-mid-eval.pdf) | 18-Slide Midsem Beamer Deck (`proj-mid-eval.tex`) |
| **Interactive Web Dashboard** | [`app.py`](./app.py) | Streamlit 10-Tab Verification Application (`streamlit run app.py`) |
| **Automated Test Suite** | [`tests/`](./tests/) | 21/21 Passing Unit Tests on CPU in 67.0s (`tests/test_endsem_modules.py`) |

---

## 📊 Final Quantitative Benchmark Results

Evaluated across 50 multi-hop academic reasoning queries against Baseline LongRAG:

| Metric Dimension | Baseline LongRAG (Raw 4k Chunks) | Proposed E-LongRAG | Relative Advantage |
| :--- | :---: | :---: | :---: |
| **Avg. Prompt Token Payload** | $5,134.0\text{ tokens}$ | **$260.8\text{ tokens}$** | **$94.9\%$ Prompt Token Reduction** |
| **LLM Pre-fill TTFT Latency** | $\approx 2,000\text{ ms}$ | **$\approx 100\text{ ms}$** | **$19.7\times$ Pre-fill Latency Speedup** |
| **Context Retrieval Precision** | $24.1\%$ | **$94.2\%$** | **$+290.9\%$ Signal-to-Noise Boost** |
| **Lost-in-Middle (50% Center Depth)** | $35.2\%$ (Severe Collapse) | **$100.0\%$ (Flat Invariance)** | **Zero Hallucination / Attention Sink** |
| **Faithfulness Score** | $0.720$ | **$0.980$** | **$+36.1\%$ Factual Grounding** |
| **Exact Match (EM)** | $0.400$ | **$0.900$** | **$+125.0\%$ Strict Answer Accuracy** |
| **Harmonic RAGAS Index** | $0.534$ | **$0.957$** | **$+79.2\%$ Overall Quality Index** |
| **Local CPU Reader Latency** | $\approx 2,010\text{ ms}$ | **$4.12\text{ ms}$** | **Sub-5ms Zero-Cost Inference** |
| **Hardware Portability** | High-VRAM GPU Required | **Consumer Host CPU** | **Zero Paid GPU Compute Required** |

---

## 🏗️ End-to-End System Architecture

```text
                             ┌─────────────────────────────────┐
                             │       User Question q (QA)      │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 1. HyDE Query Expansion         │
                             │    Generates d_hat ~ P(q)       │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 2. Hybrid FAISS + Okapi BM25    │
                             │    Dense Cosine + Sparse RSJ IDF│
                             │    Reciprocal Rank Fusion (c=60)│
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 3. Intra-Chunk Paragraph Slicer │
                             │    Decomposes 4k into {p_k,j}   │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 4. MiniLM Neural Cross-Encoder  │
                             │    All-to-All Cross-Attention   │
                             │    Calibrated Score R_CE(q, p)  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 5. Query-Adaptive Cutoff tau(q) │
                             │    tau(q) = max(0.40, mu - a*sig│
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 6. Cosine Semantic Deduplication│
                             │    Prunes Duplicates (delta=0.85│
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ 7. Multi-Reader Synthesis Engine│
                             │    Local Zero-GPU CPU (4.12 ms) │
                             │    Gemini 1.5 Flash / Groq LPUs │
                             │    (260.8 Tokens, 94.2% Signal) │
                             └─────────────────────────────────┘
```

---

## 🔬 New Endsemester Modules & Innovations

### 1. Hybrid Search (Dense + BM25 + Reciprocal Rank Fusion)
- **Module**: `src/hybrid_search.py`
- Combines FAISS `IndexFlatIP` cosine similarity with pure-Python Okapi BM25 inverted index using Robertson-Spärck Jones IDF:
  $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{BM25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- Executes in $2.31\text{ ms}$ on CPU, guaranteeing keyword recall against technical acronyms and alphanumeric identifiers.

### 2. Retrieval Sampling Strategies Ablation Suite
- **Module**: `src/sampling.py`
- Evaluates 5 mathematical passage selection strategies: Deterministic Top-$K$, Nucleus Top-$p$, Boltzmann Stochastic Sampling, Maximal Marginal Relevance (MMR), and Adaptive Filtering.
- Formulates the Intra-Context Diversity metric:
  $$\text{Div}(\mathcal{S}) = \frac{1}{|\mathcal{S}|(|\mathcal{S}|-1)} \sum_{i \ne j} \left( 1 - \frac{E(p_i) \cdot E(p_j)}{\|E(p_i)\|_2 \|E(p_j)\|_2} \right)$$
- Proves that Deterministic Top-$K$ suffers from low diversity ($0.1925$), while Adaptive Filtering optimizes both diversity ($0.3078$) and token compression ($94.9\%$).

### 3. Positional "Lost-in-the-Middle" Stress Testing
- **Module**: `src/lost_in_the_middle.py`
- Constructs 10,000-token synthetic multi-document contexts and sweeps needle positions across depths ($0\%, 25\%, 50\%, 75\%, 100\%$).
- Empirically proves that while baseline LongRAG plummets to **$35.2\%$ recall** at 50% depth due to transformer attention sink decay, E-LongRAG achieves **$100.0\%$ position-invariant recall**.

### 4. Automated RAGAS Quality Metrics & Exact Match
- **Module**: `src/ragas_eval.py`
- Evaluates synthesized answers across Faithfulness ($0.98$), Answer Relevance ($0.95$), Context Precision ($0.942$), Context Recall ($0.960$), Exact Match ($0.90$), and Token F1 ($0.88$).
- Harmonic RAGAS quality score surges from $0.534$ to **$0.957$** ($+79.2\%$ gain).

### 5. Cross-Encoder Model Sizing & 42-Point Pareto Sensitivity Frontier
- **Module**: `src/sensitivity.py`
- Benchmarks `TinyBERT-L-2` (4.4M), `MiniLM-L-4` (14.8M), and `MiniLM-L-6` (22.7M).
- Sweeps 42 configurations across $\alpha \in [0.0, 1.5] \times \delta \in [0.70, 0.95]$.
- Identifies golden operating point $(\alpha = 0.50, \delta = 0.85)$ yielding $94.6\%$ token reduction with $100.0\%$ factual recall.

### 6. Multi-Reader Synthesis Engine
- **Module**: `src/reader.py`
- Evaluates compressed context across Local Zero-GPU CPU Reader ($4.12\text{ ms}$, $\$0.00$), Google Gemini 1.5 Flash ($482\text{ ms}$), and Groq LLaMA-3.1-8B-Instant ($124\text{ ms}$).

---

## 💻 Interactive 10-Tab Web Dashboard (`app.py`)

Launch the Streamlit web demo:
```bash
conda activate elongrag
streamlit run app.py
```

### Dashboard Tabs Breakdown:
1. **🚀 1. Side-by-Side Comparison**: Live comparison of baseline vs E-LongRAG answers, prompt tokens, and latencies.
2. **📚 2. Raw Corpus Chunks**: Inspection of coarse 4,000-token chunks.
3. **🔍 3. Noise Filter Inspector**: Visual badges for `RETAINED` vs `PRUNED` paragraphs with exact cross-attention scores.
4. **📊 4. Empirical Benchmarks**: Quantitative component ablation tables and compression charts.
5. **🔬 5. Sampling Strategies Ablation**: Real-time sliders ($K, p, T, \lambda$) and intra-context diversity tracking.
6. **🔀 6. Hybrid Search (Dense + BM25 + RRF)**: Reciprocal Rank Fusion testing with bi-encoder parameter ablation.
7. **📍 7. Lost-in-the-Middle Stress Test**: Needle position slider and live chart of baseline collapse vs E-LongRAG invariance.
8. **🎯 8. RAGAS Quality & Exact Match**: Multi-dimensional radar/bar charts for Faithfulness, Relevance, Precision, Recall, EM, Token F1.
9. **⚡ 9. Sensitivity Sweep & Pareto Frontier**: Live 2D grid sweep and Pareto efficiency trade-off curves.
10. **🤖 10. Multi-Reader Benchmarks**: Side-by-side comparison cards for Local CPU, Gemini 1.5 Flash, and Groq LLaMA-3.1.

---

## 🧪 Comprehensive Engineering Verification (21/21 Unit Tests)

Run the full automated test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

```text
Ran 21 tests in 67.007s — OK (100% Pass Rate on CPU)

✓ BM25 index correctly scored and retrieved exact keyword match.
✓ Hybrid search successfully fused Dense & BM25 rankings via RRF (c=60).
✓ Embedding ablation accurately profiled parameter sizes, latency, and recall.
✓ Lost-in-the-Middle proven: Baseline dips at 50% while E-LongRAG is 100% position invariant.
✓ RAGAS metrics evaluated: Faithfulness=1.0, Harmonic Index=0.9814.
✓ Multi-Reader engine generated answers across Local CPU, Gemini, and Groq.
✓ Pareto sweep identified optimal boundary configurations on the sensitivity frontier.
✓ Full ELongRAGPipeline endsem integration verified 100% end-to-end.
✓ Deterministic Top-K, Nucleus Top-p, Boltzmann, and MMR sampling verified.
✓ DatasetLoader, Chunker, FAISS VectorStore, HyDE, and MiniLM Reranker verified.
```

---

## 👥 Team & Individual Contributions Breakdown

The project workload and module ownership across both Midsem and Endsem phases were allocated according to technical specialization:

| Team Member | Midsem Responsibilities | Endsem Responsibilities & Module Ownership |
| :--- | :--- | :--- |
| **Pranav Vinodh**<br>*(Lead System Architect & Core Innovations)* | • Intra-chunk paragraph slicing pipeline.<br>• Dynamic statistical cutoff $\tau(q) = \mu_R - \alpha \sigma_R$.<br>• Semantic cosine deduplication ($\delta = 0.85$).<br>• $O(N^2)$ pre-fill attention latency analysis. | • **Pure-Python Okapi BM25 & RSJ IDF**: Built sparse inverted index engine (`src/hybrid_search.py`).<br>• **Hybrid Dense-Lexical Search with RRF**: Fused FAISS & BM25 rankings ($c=60$) in `src/hybrid_search.py`.<br>• **Retrieval Sampling Suite**: Formulated Top-$K$, Nucleus, Boltzmann, and MMR algorithms (`src/sampling.py`).<br>• **Intra-Context Diversity Metric**: Formulated mathematical $\text{Div}(\mathcal{S})$ metric.<br>• **Lost-in-the-Middle Stress Testing**: Built 10k-token needle tester & invariance proof (`src/lost_in_the_middle.py`).<br>• **Automated RAGAS Quality Evaluator**: Built multi-metric evaluator (EM, F1) in `src/ragas_eval.py`.<br>• **Pareto Sensitivity Sweeper**: Formulated 42-point 2D grid sweep ($\alpha \times \delta$) in `src/sensitivity.py`.<br>• **Multi-Reader Engine**: Interfaced Local CPU, Gemini 1.5, and Groq LPUs (`src/reader.py`). |
| **Durga Sai Pavan Gangabattula**<br>*(Web Dashboard & Visual Telemetry)* | • HyDE generative query expansion (`src/hyde.py`).<br>• Evaluation query dataset identification assistance.<br>• Environment & requirements configuration. | • **Interactive 10-Tab Web Dashboard**: Developed full front-end user application in Streamlit (`app.py`).<br>• **Dark UI Layout & Visual Styling**: Designed dark-themed cards, badges, and layout aesthetics.<br>• **Visual Telemetry Plotting**: Generated static charts and graphs for RAGAS and Pareto trade-off curves.<br>• **Interactive Control Sliders**: Added real-time user widgets for Top-$K$, Top-$p$, Temperature, and MMR $\lambda$. |
| **Sameer**<br>*(Corpus Curation & Documentation)* | • Coarse 4k chunking text aggregation (`src/chunker.py`).<br>• FAISS `IndexFlatIP` vector index storage setup (`src/vector_store.py`).<br>• Baseline pre-fill latency profiling assistance. | • **Academic Dataset Serialization**: Formatted 50 multi-hop reasoning instances into `data/sample_dataset.json`.<br>• **Distractor Haystack Compilation**: Collected background domain logs for 10k-token stress tests.<br>• **Beamer Presentation Formatting**: Assisted in presentation slide styling, margins, and typography review.<br>• **Report Verification**: Assisted in cross-checking citations and IEEE template compliance. |

---

## 📜 License

This project is developed for educational and research purposes under the **Apache-2.0 License**.
