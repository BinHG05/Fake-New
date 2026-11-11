# Limitations & Future Work

> **Critical appraisal of current pipeline boundaries, constraints, and operational limitations.**

---

## 1. Domain & Distribution Constraints

1. **Platform Specificity:**
   - The cascade propagation topology is heavily modeled after Reddit's hierarchical comment tree structure (`reply_to` links and comment voting scores).
   - Direct transfer to flat microblog platforms (e.g., Twitter/X retweets, Bluesky) requires adapting the graph builder to star or quote topologies.
2. **Language Coverage:**
   - The primary canonical training data is English-dominant (Reddit and Fakeddit).
   - Although the model utilizes XLM-RoBERTa, multilingual performance (e.g. Vietnamese forum data) requires dedicated, editorially verified ground truth to prevent label leakage.

---

## 2. Early Detection & Sparse Cascades

1. **Cold Start Latency:**
   - For freshly submitted posts with 0 or few comments, the cascade graph degenerates into a single root node with a self-loop edge.
   - The model includes a pre-GNN global skip connection to ensure meaningful predictions even on empty cascades, but the full topological advantage is realized once discussion begins to develop.
2. **Missing Modality Handling:**
   - Text-only posts have zero-padded image embeddings and a missing-image flag. While the cross-attention module handles this without crashing, classification relies primarily on linguistic features in these instances.

---

## 3. Computational Considerations

1. **Memory Scaling with Deep Cascades:**
   - Exceptionally large Reddit threads (thousands of comments) require graph pruning or depth truncation to prevent GPU out-of-memory errors during real-time batch processing.
2. **Embedding Latency:**
   - Generating dual representations from XLM-RoBERTa and CLIP adds approximately 40–80ms per post on GPU, making high-throughput ingestion pipelines reliant on asynchronous embedding queues.
