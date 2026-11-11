# PR Part 02 - Shared Evaluation Harness

## Muc tieu

Tao mot nen chung de cac thi nghiem sau deu dung cung mot cach:

- load test split
- load binary checkpoint
- chay prediction
- mutate graph de test ablation / robustness
- tinh metric va thong ke topology

## Thay doi chinh

1. Them `scripts/paper_experiments/common.py`
   File nay gom:
   - `load_binary_metadata()`
   - `load_test_graphs()`
   - `load_model()`
   - `predict_graphs()`
   - helper `keep_root_only()` va `keep_root_plus_k_comments()`
   - helper topology: `graph_depth()`, `branching_factor()`, `average_degree()`

2. Ghep cung naming cua repo hien tai
   Harness dung:
   - `data/processed_graphs_v3`
   - `data/reddit_enriched_binary.jsonl`
   - `models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt`

3. Chuan hoa output cho cac script nghien cuu
   Moi prediction duoc dua ve cung schema:
   - `post_id`
   - `y_true`
   - `y_pred`
   - `prob_real`
   - `prob_fake`
   - `confidence`
   - `num_nodes`
   - `num_edges`

## Y nghia

PR nay cat bo rui ro "moi TN mot evaluator rieng", giup so lieu cua TN5, TN2, TN3 thong nhat va de doi chieu khi viet paper.

## Ghi chu

Moi truong shell hien tai chua co `torch`, nen harness duoc wire sẵn theo runtime cua repo va se phat huy day du khi chay trong environment ML cua project.
