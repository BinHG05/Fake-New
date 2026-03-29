# Binary Results Summary

- Best baseline: **Graph GAT**
  Binary Accuracy: **63.93%**
  Binary F1: **49.75%**

- Full model: **66.43%** Binary Accuracy, **56.88%** Binary F1
- Delta vs best baseline:
  Accuracy: **+2.50 pp**
  F1: **+7.13 pp**

- Ablation delta vs Full Model:
  - Text Only: 43.93% (-22.50 pp)
  - Image Only: 42.14% (-24.29 pp)
  - No Graph: 46.07% (-20.36 pp)
  - Text Graph Only: 45.71% (-20.72 pp)
  - Gated Fusion: 46.07% (-20.36 pp)
