# Phase 4: Multimodal GNN — Kế Hoạch & Hướng Dẫn

> **Mục tiêu:** Xây dựng mô hình Multimodal GNN kết hợp Text + Image + Graph  
> **Benchmark:** Vượt Phase 3 baselines (Text F1=0.19, Graph Binary Acc=66.7%)

---

## 1. Tổng Quan Kiến Trúc

```
  XLM-RoBERTa (frozen)                    ResNet-50 (frozen)
       │ [CLS] = 768d                         │ avgpool = 2048d
       ▼                                      ▼
  ┌──────────────────────────────────────────────┐
  │          Cross-Attention Fusion              │
  │  text → Q │ image → K,V │ gate → blend      │
  │  Output: 256d fused embedding                │
  └──────────────────────────────────────────────┘
                      │
                      ▼
  ┌──────────────────────────────────────────────┐
  │         GraphSAGE × 2 (with residual)        │
  │  Message passing qua interaction graph       │
  └──────────────────────────────────────────────┘
                      │
                      ▼
  ┌──────────────────────────────────────────────┐
  │         MLP Classifier (256→128→6)           │
  └──────────────────────────────────────────────┘
                      │
                      ▼
              Logits [N, 6 classes]
```

### Tại sao kiến trúc này?

| Vấn đề Phase 3 | Giải pháp Phase 4 |
|-----------------|-------------------|
| Simple concat fusion thất bại | **Cross-Attention** + **Gated residual** — model tự học weight |
| Image noise làm hại text | **Gate → 0** khi image không hữu ích → fallback về text |
| Unimodal baselines yếu | **GNN** kết hợp thông tin cấu trúc giữa các bài viết |
| Fusion MLP quá lớn (722K params) | Fusion chỉ ~200K params, phù hợp dataset nhỏ |

---

## 2. Cấu Trúc File

```
src/models/
├── multimodal_fusion.py    ← Cross-Attention + Gated Fusion modules
├── advanced_gnn.py         ← MultimodalGraphNet + PrecomputedMultimodalGNN
└── cascade_gnn.py          ← GNN cũ (Phase 2, vẫn giữ)

src/training/
├── train_multimodal_gnn.py ← Training script Phase 4
├── train_baseline.py       ← Training script Phase 3 (baselines)
└── train_gnn.py            ← Training script Phase 2 (GNN cũ)
```

---

## 3. Các Module Chi Tiết

### 3.1 CrossAttentionFusion (`multimodal_fusion.py`)

```python
# Text embedding làm Query, Image embedding làm Key/Value
attended = MultiHeadAttention(Q=text_proj, K=image_proj, V=image_proj)

# Gated residual: khi image yếu → gate ≈ 0 → output ≈ text
gate = σ(W · [text_proj; attended])
output = gate * attended + (1-gate) * text_proj
```

**Trainable params:** ~200K

### 3.2 GatedFusion (`multimodal_fusion.py`)

Phương pháp đơn giản hơn:
```python
gate = σ(W · [text_proj; image_proj])
output = gate * text_proj + (1-gate) * image_proj
```

### 3.3 PrecomputedMultimodalGNN (`advanced_gnn.py`)

Dùng với graph `.pt` đã có (node features = `[text_emb || image_emb]`):
- Tách features → Fusion → GraphSAGE → Classifier
- Hỗ trợ ablation qua parameter `ablation`

### 3.4 MultimodalGraphNet (`advanced_gnn.py`)

End-to-end: nhận raw text + image → frozen encoder → Fusion → GNN → classify.
- Dùng khi muốn train trên dữ liệu mới chưa extract embedding
- Cần nhiều memory hơn (load XLM-R + ResNet)

---

## 4. Cách Chạy

### Full Model (Precomputed)
```bash
conda activate multimodal_gnn
python src/training/train_multimodal_gnn.py --epochs 50 --lr 1e-3
```

### Thay đổi Fusion Type
```bash
# Cross-attention (mặc định)
python src/training/train_multimodal_gnn.py --fusion_type cross_attention

# Gated fusion
python src/training/train_multimodal_gnn.py --fusion_type gated
```

### Ablation Study
```bash
# Chỉ text (zero image)
python src/training/train_multimodal_gnn.py --ablation text_only --epochs 50

# Chỉ image (zero text)
python src/training/train_multimodal_gnn.py --ablation image_only --epochs 50

# Không dùng graph (fusion only)
python src/training/train_multimodal_gnn.py --ablation no_graph --epochs 50
```

### CLI Arguments

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--graph` | `data/04_graph/fakeddit_graph.pt` | File graph |
| `--fusion_type` | `cross_attention` | `cross_attention` hoặc `gated` |
| `--fusion_dim` | 256 | Chiều fused embedding |
| `--hidden_dim` | 256 | GNN hidden dim |
| `--num_gnn_layers` | 2 | Số lớp GraphSAGE |
| `--epochs` | 50 | Max epochs |
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 15 | Early stopping |
| `--ablation` | None | `text_only`, `image_only`, `no_graph` |

---

## 5. Output Kỳ Vọng

Script sẽ in ra:
1. **Training logs** mỗi 5 epochs (loss, val F1, val binary acc)
2. **Final test results** (6-class + binary metrics)
3. **Classification report** per-class
4. **So sánh tự động** với Phase 3 baselines

### Mục tiêu

| Metric | Phase 3 Best | Phase 4 Target |
|--------|:------------:|:--------------:|
| 6-Class Macro-F1 | 0.192 (Text) | > 0.22 |
| Binary Accuracy | 66.7% (Graph) | > 70% |

---

## 6. Checklist

- [x] Implement `CrossAttentionFusion`
- [x] Implement `GatedFusion`
- [x] Implement `MultimodalGraphNet`
- [x] Implement `PrecomputedMultimodalGNN`
- [x] Implement `train_multimodal_gnn.py`
- [ ] Chạy smoke test (1 epoch)
- [ ] Chạy full training + so sánh với Phase 3
- [ ] Chạy ablation study
- [ ] Viết báo cáo `docs/phase4_results.md`
