import json
import os
from .config import DATA_DIR, SAMPLE_DATASET_PATH

def generate_realistic_benchmark_dataset(num_samples=100):
    """
    Constructs a rich, realistic, factual multi-hop academic dataset for benchmarking.
    Includes real historical figures, foundational research papers, algorithms, awards,
    and realistic domain-specific distractor passages.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    benchmark_topics = [
        {
            "topic": "Distributed Consensus",
            "concept": "Lamport Timestamps",
            "concept_alt": "Vector Clocks & Paxos",
            "pioneer": "Leslie Lamport",
            "honor": "Turing Award in 2013",
            "year": "1978",
            "venue": "Communications of the ACM",
            "impact": "establishing partial ordering of events in distributed computing systems without requiring synchronized physical clocks",
            "distractor_domain": "High-Throughput Distributed Cloud Clusters",
            "distractor_content": [
                "Modern industrial datacenters implement multi-region Paxos clusters for high-availability database replication. Performance benchmarks recorded an average commit latency of 4.2 milliseconds across geographically distributed datacenters in North America and Europe.",
                "Hardware cooling infrastructures in hyperscale server facilities utilize liquid-to-air heat exchangers. Automated sensor telemetry monitors thermal dissipation across server racks to maintain ambient temperatures within ASHRAE environmental standards.",
                "Network packet routing protocols in software-defined wide area networks (SD-WAN) apply dynamic multipath optimization to mitigate packet jitter and bandwidth bottlenecks during peak traffic cycles."
            ]
        },
        {
            "topic": "Quantum Computing",
            "concept": "Shor's Algorithm",
            "concept_alt": "Grover's Quantum Search",
            "pioneer": "Peter Shor",
            "honor": "Dirac Medal in 2002",
            "year": "1994",
            "venue": "IEEE Symposium on Foundations of Computer Science (FOCS)",
            "impact": "finding prime factors of integers in polynomial time on a quantum computer, posing a fundamental challenge to classical RSA cryptography",
            "distractor_domain": "Superconducting Qubit Cryogenics",
            "distractor_content": [
                "Dilution refrigerators in quantum hardware laboratories maintain superconducting transmon qubits at temperatures below 15 millikelvin. Helium-3 and Helium-4 isotopic mixtures circulate continuously through heat exchangers.",
                "Quantum error correction schemes utilizing surface codes require physical-to-logical qubit overhead ratios exceeding 1,000 to 1. Fault-tolerant threshold theorems demonstrate error suppression below the fault-tolerance threshold limit.",
                "Microwave pulse generators synthesize high-fidelity control pulses for single-qubit and two-qubit entangling gates, achieving gate fidelities exceeding 99.5% on benchmark superconducting processors."
            ]
        },
        {
            "topic": "Deep Learning Transformers",
            "concept": "Attention Is All You Need",
            "concept_alt": "BERT and GPT Architectures",
            "pioneer": "Ashish Vaswani and the Google Brain team",
            "honor": "NIPS 2017 Best Paper Recognition",
            "year": "2017",
            "venue": "Neural Information Processing Systems (NIPS)",
            "impact": "introducing multi-head self-attention mechanisms that eliminated recurrence and enabled massive parallel training of sequence models",
            "distractor_domain": "GPU Cluster Acceleration & FlashAttention",
            "distractor_content": [
                "NVIDIA H100 Tensor Core GPUs utilize Transformer Engines with FP8 precision formatting to accelerate transformer pretraining. Dedicated NVLink interconnects deliver 900 GB/s bidirectional bandwidth per accelerator node.",
                "FlashAttention algorithms optimize GPU SRAM memory hierarchy by tiling softmax operations and avoiding intermediate $O(N^2)$ memory reads and writes between high-bandwidth HBM and on-chip SRAM.",
                "Megatron-LM distributed training combines tensor parallelism, pipeline parallelism, and sequence parallelism to scale model weights across thousands of compute clusters with 54% Model FLOPs Utilization (MFU)."
            ]
        },
        {
            "topic": "Database Systems",
            "concept": "Two-Phase Locking and ARIES",
            "concept_alt": "Multi-Version Concurrency Control (MVCC)",
            "pioneer": "Jim Gray",
            "honor": "Turing Award in 1998",
            "year": "1976",
            "venue": "ACM Transactions on Database Systems",
            "impact": "defining ACID transaction semantics, serializability guarantees, and write-ahead logging protocols for relational database crash recovery",
            "distractor_domain": "Distributed NoSQL Storage Engines",
            "distractor_content": [
                "LSM-tree (Log-Structured Merge-tree) storage engines buffer incoming writes in memory (MemTable) before sequentially flushing immutable SSTables to persistent NVMe solid-state storage devices.",
                "Columnar storage layouts compress relational tables using dictionary encoding, run-length encoding (RLE), and bit-packing to maximize scan performance for analytical OLAP workloads.",
                "B-tree index node splitting algorithms maintain balanced tree heights of 3 to 4 levels, ensuring worst-case logarithmic lookup complexity across multi-terabyte indexed key-value datasets."
            ]
        },
        {
            "topic": "Autonomous Driving",
            "concept": "LIDAR SLAM and Probabilistic Robotics",
            "concept_alt": "End-to-End Neural Driving Models",
            "pioneer": "Sebastian Thrun and the Stanford/Waymo Team",
            "honor": "DARPA Grand Challenge Victory in 2005",
            "year": "2005",
            "venue": "Journal of Field Robotics",
            "impact": "pioneering autonomous vehicle localization and mapping using particle filters and 3D point cloud LIDAR perception in real-world terrains",
            "distractor_domain": "Automotive CAN Bus & Edge Embedded Compute",
            "distractor_content": [
                "Controller Area Network (CAN FD) bus architectures transmit vehicle telemetry between electronic control units (ECUs) at bit rates up to 5 Mbps with cyclic redundancy check (CRC) fault detection.",
                "Automotive camera image sensors apply high dynamic range (HDR) exposure bracketing to prevent sensor saturation under harsh daylight and glare conditions.",
                "ASIL-D functional safety compliance for automotive hardware mandates dual-core lockstep CPU architectures and watchdog safety timers to detect transient hardware faults."
            ]
        },
        {
            "topic": "Graph Neural Networks",
            "concept": "Graph Attention Networks (GAT)",
            "concept_alt": "Graph Convolutional Networks (GCN)",
            "pioneer": "Petar Velickovic and Yoshua Bengio",
            "honor": "ICLR 2018 Outstanding Research Paper",
            "year": "2018",
            "venue": "International Conference on Learning Representations (ICLR)",
            "impact": "introducing masked self-attentional layers to compute node representations over arbitrary graph structures without requiring computationally expensive matrix inversions",
            "distractor_domain": "Large-Scale Graph Partitioning",
            "distractor_content": [
                "Metis graph partitioning algorithms minimize edge-cut metrics when distributing massive graph topologies across distributed GPU memory banks during multi-GPU message passing.",
                "Sparse matrix-matrix multiplication (SpGEMM) accelerators utilize customized systolic array datapaths to accelerate sparse graph neighborhood aggregations.",
                "Temporal graph networks maintain dynamic memory states for continuous-time dynamic interaction networks in financial fraud detection pipelines."
            ]
        },
        {
            "topic": "Cryptography & Security",
            "concept": "RSA Public-Key Cryptosystem",
            "concept_alt": "Elliptic Curve Cryptography (ECC)",
            "pioneer": "Ron Rivest, Adi Shamir, and Leonard Adleman",
            "honor": "Turing Award in 2002",
            "year": "1977",
            "venue": "Communications of the ACM",
            "impact": "inventing the first practical public-key cryptosystem and digital signature algorithm based on the mathematical intractability of factoring large composite integers",
            "distractor_domain": "Post-Quantum Lattice Cryptography",
            "distractor_content": [
                "NIST selected ML-KEM (Kyber) and ML-DSA (Dilithium) as primary post-quantum cryptographic standards based on the hardness of Learning With Errors (LWE) over module lattices.",
                "Hardware Security Modules (HSMs) enforce tamper-resistant physical boundaries with zeroization circuitry to protect cryptographic private keys against side-channel analysis.",
                "Transport Layer Security (TLS 1.3) mandates forward secrecy via ephemeral Diffie-Hellman key exchanges and eliminates obsolete cipher suites."
            ]
        }
    ]

    processed_samples = []
    
    for i in range(num_samples):
        spec = benchmark_topics[i % len(benchmark_topics)]
        idx = i // len(benchmark_topics)
        
        sample_id = f"sample_{i}"
        question = f"Who introduced {spec['concept']} in the domain of {spec['topic']}, and which major honor did they receive?"
        ground_truth = f"{spec['pioneer']} introduced {spec['concept']} and was awarded the {spec['honor']}."

        # Gold paragraph 1: Concept & Invention
        p1_title = f"{spec['concept']} - Foundational Paper ({spec['year']})"
        p1_text = (
            f"The foundational principles of {spec['concept']} were formulated and published by {spec['pioneer']} in {spec['year']} "
            f"within {spec['venue']}. This work revolutionized {spec['topic']} by {spec['impact']}. "
            f"The paradigm introduced strict mathematical foundations that eliminated historical synchrony and performance bottlenecks."
        )

        # Gold paragraph 2: Pioneer Biography & Award
        p2_title = f"{spec['pioneer']} - Academic Career & Honors"
        p2_text = (
            f"For transformative contributions to computer science and specifically for inventing {spec['concept']} in the field of {spec['topic']}, "
            f"{spec['pioneer']} received the prestigious {spec['honor']}. The citation emphasized the enduring architectural impact of their work "
            f"on modern computing systems, algorithmic correctness, and distributed reliability."
        )

        # Distractor paragraphs from the same or related domain
        d1_title = f"{spec['distractor_domain']} - Industrial Architecture #{idx+1}"
        d1_text = spec["distractor_content"][0]

        d2_title = f"{spec['distractor_domain']} - Hardware Telemetry & Benchmarks #{idx+1}"
        d2_text = spec["distractor_content"][1]

        d3_title = f"{spec['distractor_domain']} - Engineering Standards #{idx+1}"
        d3_text = spec["distractor_content"][2]

        processed_samples.append({
            "id": sample_id,
            "topic": spec["topic"],
            "question": question,
            "answer": ground_truth,
            "type": "bridge_multi_hop",
            "level": "hard",
            "paragraphs": [
                {"title": p1_title, "text": p1_text, "is_distractor": False},
                {"title": p2_title, "text": p2_text, "is_distractor": False},
                {"title": d1_title, "text": d1_text, "is_distractor": True},
                {"title": d2_title, "text": d2_text, "is_distractor": True},
                {"title": d3_title, "text": d3_text, "is_distractor": True}
            ]
        })

    with open(SAMPLE_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, indent=2)

    print(f"[DatasetLoader] Successfully saved {len(processed_samples)} realistic multi-hop academic samples to {SAMPLE_DATASET_PATH}")
    return processed_samples

def load_or_fetch_dataset(num_samples=100, force_reload=True):
    """
    Loads or generates the high-quality realistic multi-hop academic benchmark dataset.
    """
    if os.path.exists(SAMPLE_DATASET_PATH) and not force_reload:
        with open(SAMPLE_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Verify if cached data is the new realistic format
            if data and "Lamport" in data[0]["question"]:
                return data[:num_samples]

    return generate_realistic_benchmark_dataset(num_samples=num_samples)

if __name__ == "__main__":
    ds = load_or_fetch_dataset(20, force_reload=True)
    print("Sample 0 Question:", ds[0]["question"])
    print("Sample 0 Answer:", ds[0]["answer"])
    print("Sample 0 P0 Title:", ds[0]["paragraphs"][0]["title"])
