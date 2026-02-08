# Task: Multimodal Fake News Detection - GNN Pipeline

## Current Status: EXECUTION - Batch Automation

### Phase 1-4: Data & Graph Preparation ✅
- [x] Merge splits utility
- [x] Embedding extractors (XLM-R + CLIP)
- [x] Graph builder (Top-K similarity)
- [x] Dataloader & Model architecture

### Phase 5: GNN Training (Pilot Study) ✅
- [x] Implemented `train_gnn.py` (Simple GCN)
- [x] Tested on 200 samples

### Phase 2B: Cascade Data Collection (New)
- [x] **Enhance `reddit_crawler.py`** to collect comments/replies tree
- [x] Build `CascadeGraph` from reply structure
- [ ] Collect additional datasets (PHEME/Twitter15) if needed

### Phase 3: Baseline Models (The "Control Group")
*Before complex GNNs, we need benchmarks.*
- [ ] **Text Baseline:** Train BERT-base classifier
- [ ] **Image Baseline:** Train ResNet/ViT classifier
- [ ] **Graph Baseline:** Train standard GCN/GAT on Topology Graph
- [ ] **Fusion Baseline:** Early Fusion (Concat BERT+ResNet) -> MLP

### Phase 4: Multimodal GNN (The "Main Event")
- [ ] **Data Integration:** Combine Text+Image+Cascade features
- [ ] **Cross-Attention:** Implement fusion module (Text queries Image)
- [ ] **Advanced GNN:** GraphSAGE or HGT
- [ ] **Propagation Modeling:** Add Tree-LSTM for cascades

### Phase 7: Batch Pipeline Automation (Maintenance)
- [x] Support CLI args in `fakeddit_preprocessor_image.py`
- [/] Support CLI args in `fakeddit_process_text.py`
- [ ] **Batch 200-400 Processing**
    - [x] Batch extraction & Image/Text processing
    - [x] Label Studio Setup & Labeling ✅
    - [ ] Export and Merge (`merge_labeled_batch_200.bat`)
