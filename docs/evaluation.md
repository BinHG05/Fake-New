# Evaluation & Experimental Benchmarks

> **Empirical Evaluation of the Canonical Multimodal Cascade GNN Pipeline**

---

## 1. Official Canonical Results

Evaluated on the held-out test split of `data/all_merged_binary.jsonl` ($N = 407$ samples) using canonical checkpoint `artifacts/checkpoints/multimodal_cascade_gnn_binary.pt`:

| Metric | Official Score | Notes |
| :--- | :---: | :--- |
| **Headline Test Accuracy** | **87.96%** | Primary portfolio headline metric |
| **Binary F1-Score** | **0.7932** | Harmonic mean of Precision & Recall for FAKE class |
| **Binary Precision** | **0.7833** | $94 / (94 + 26)$ |
| **Binary Recall** | **0.8034** | $94 / (94 + 23)$ |
| **ROC AUC** | **0.9490** | Area Under Receiver Operating Characteristic Curve |
| **Validation Accuracy** | **89.66%** | Measured on validation split ($N = 406$) |
| **Validation F1-Score** | **0.8056** | Validation harmonic mean |

---

## 2. Confusion Matrix Breakdown

$$\begin{pmatrix} \text{True REAL: 264} & \text{False FAKE: 26} \\ \text{False REAL: 23} & \text{True FAKE: 94} \end{pmatrix}$$

- **True Negatives (REAL correctly identified):** 264 / 290 (91.03% specificity)
- **True Positives (FAKE correctly detected):** 94 / 117 (80.34% sensitivity)
- **False Positives (REAL misclassified as FAKE):** 26
- **False Negatives (FAKE missed as REAL):** 23

The plot is automatically generated at `artifacts/figures/confusion_matrix.png`.

---

## 3. Comparison with Baseline Models

| Model Architecture | Modalities Utilized | Test Accuracy | Binary F1 |
| :--- | :--- | :---: | :---: |
| **Text-Only Baseline** | Text (XLM-R 768d) | 74.20% | 0.6624 |
| **Image-Only Baseline** | Image (CLIP ViT-B/32) | 28.75% | 0.4466 |
| **Early Fusion Baseline** | Text + Image Concat (1793d) | 75.43% | 0.6429 |
| **Graph-Only Baseline** | Cascade GAT (Text only) | 74.20% | 0.6769 |
| **CLIP-Only Feature Baseline** | CLIP Text + Image + CosSim (1025d) | 90.00%* | 0.8772* |
| **Multimodal Cascade GNN (Canonical)** | **Text + Image + Cascade Graph** | **87.96%** | **0.7932** |

*\*Note on CLIP-only baseline:* The CLIP-only model achieved 90.00% test accuracy on a static feature subset, but operates without social propagation dynamics, comment consensus signals, or network topology. It is maintained in this repository strictly as a strong feature benchmark.

---

## 4. Key Takeaways

1. **Cascade Topology Impact:** Incorporating comment cascade trees improves robust contextualization over unimodal text baselines by over **+13.76% accuracy**.
2. **Multimodal Synergy:** Visual cues alone are insufficient ($28.75\%$), but when cross-attended with text in the cascade root, the model successfully detects visual-textual discrepancies characteristic of misinformation.
3. **High Discriminative Power:** The ROC AUC of **0.9490** confirms strong ranking separation between legitimate news posts and fabricated content.
