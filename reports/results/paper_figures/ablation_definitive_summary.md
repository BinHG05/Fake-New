# Definitive Ablation Summary

- Full model: **87.47%** accuracy, **79.18%** F1.
- Full model vs Graph-only: **+4.18 pp** accuracy.
- Full model vs No-Graph: **-0.98 pp** accuracy.
- Full model vs Text-only: **+8.85 pp** accuracy.
- Cross-Attention vs Gated fusion: **+0.98 pp** accuracy.

## Scientific takeaway

On the refreshed binary split, the no-graph content path slightly outperforms the full graph path, so the contribution of cascade enrichment must be argued carefully and supported by TN5 rather than assumed.
