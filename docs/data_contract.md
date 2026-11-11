# Data Contract & Dataset Specification

> **Official Canonical Specification for FakeNews Dataset & Graph Storage**

---

## 1. Canonical Record Schema

Every record in the canonical pipeline must strictly conform to the following JSON structure:

```json
{
  "id": "9g2b4h",
  "source": "reddit",
  "raw_text": "Scientists discover chocolate cures aging",
  "clean_text": "scientists discover chocolate cures aging",
  "label_binary": "FAKE",
  "split": "train",
  "media": {
    "url": "https://i.redd.it/sample.jpg",
    "processed_path": "data/02_processed/images/sample.jpg",
    "available": true
  },
  "cascade": [
    {
      "id": "c101",
      "parent_id": "9g2b4h",
      "text": "This study was debunked yesterday.",
      "author": "debunker_bot",
      "score": 45,
      "created_utc": 1537026000
    }
  ],
  "metadata": {
    "user_id": "author_username",
    "timestamp": 1537025291.0,
    "original_label": "False"
  }
}
```

---

## 2. Mandatory Validation Rules

1. **Unique Post ID:** Every post `id` must be globally unique across the entire dataset. Duplicate IDs trigger immediate validation abort.
2. **Strict Binary Labeling:**
   - `REAL` = `0`
   - `FAKE` = `1`
   - No `Unlabeled`, `Mostly True`, `Half True`, or ambiguous ratings are permitted in canonical splits (`train`, `val`, `test`).
3. **Partition Integrity (Zero Leakage):**
   - The set of IDs in `train`, `val`, and `test` must have zero intersection ($\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$).
4. **Graph-Record Alignment:**
   - Every record in the canonical dataset corresponds 1-to-1 with a PyG graph file located at `data/processed_graphs_v3/<id>.pt`.
5. **No Ground Truth Fabrication:**
   - Forum sources without reliable editorial verification (such as Voz or unverified community threads) must NOT be automatically labeled as `FAKE`. They are segregated into exploratory/OOD datasets.

---

## 3. Dataset Distribution Summary

Validated against `data/manifests/dataset_validation_report.json`:

| Split | Records | REAL Count | FAKE Count | Source Datasets |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 8,771 | 6,986 | 1,785 | Reddit (6,117), Fakeddit (1,601), voz (846), news (207) |
| **Validation** | 406 | 300 | 106 | Fakeddit (254), voz (134), news (18) |
| **Test** | 407 | 290 | 117 | Fakeddit (280), voz (104), news (23) |
| **Total** | **9,584** | **7,576** | **2,008** | **9,584 Graphs Matched (100%)** |

- **SHA-256 Dataset Hash:** `ba95493b30e4149df5dd92731bc3c143e78b2e447c46cc7b8e568ae28399b350`
- **Validation Errors:** 0 (100% PASS)

