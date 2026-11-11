# Binary Results Summary

- Refreshed dataset: **9,584** samples
  Train / Val / Test: **6,709 / 1,438 / 1,437**

- Best graph baseline: **Graph GAT**
  Binary Accuracy: **83.29%**
  Binary F1: **74.24%**

- Full multimodal GNN (CrossAtt):
  Binary Accuracy: **87.47%**
  Binary F1: **79.18%**

- Best content-style path in refreshed ablation:
  `No-Graph`
  Binary Accuracy: **88.45%**
  Binary F1: **80.66%**

- Delta vs graph baseline:
  Full model accuracy: **+4.18 pp**
  Full model F1: **+4.94 pp**

- Refreshed ablation notes:
  - Text Only: **78.62%** accuracy, **63.60%** F1
  - Image Only: **87.47%** accuracy, **79.84%** F1
  - No Graph: **88.45%** accuracy, **80.66%** F1
  - Gated Fusion: **86.49%** accuracy, **78.76%** F1

- Important interpretation:
  On the refreshed split, `No-Graph` slightly outperforms the full graph path, so the old narrative "cascade always gives the biggest boost" is no longer supported by the current rerun and must be discussed more carefully in the report.
