# PR Part 10 - Training Path Upgrades

## Muc tieu

Tich hop 2 huong cai tien vao train path:

- learned `[MISSING]` image embedding thay cho zero-padding co dinh
- graph augmentation de giam overfitting va tang robustness

## Thay doi chinh

1. Cap nhat `src/training/train_multimodal_gnn.py`
   - them flag `--learn_missing_image_emb`
   - them co che `missing_image_emb` chi bat khi train path moi duoc su dung
   - them 3 augmentation knobs:
     - `--augment_node_drop`
     - `--augment_edge_drop`
     - `--augment_feature_mask`
   - them `AugmentedGraphDataset`
   - checkpoint moi se tu them suffix:
     - `_missingemb`
     - `_aug`

2. Cap nhat loader/build path
   - `scripts/paper_experiments/common.py` va `src/deployment/predictor.py` nhan biet checkpoint co `missing_image_emb`
   - checkpoint cu van tuong thich, checkpoint moi cung load duoc

## Smoke verification

Toi da chay mot smoke train `1 epoch` de xac nhan path moi hoat dong tu dau den cuoi.

Command:

- `src/training/train_multimodal_gnn.py --graph_dir data/processed_graphs_v3 --metadata data/reddit_enriched_binary.jsonl --fusion_type cross_attention --gnn_type gat --epochs 1 --patience 1 --batch_size 128 --learn_missing_image_emb --augment_node_drop 0.05 --augment_edge_drop 0.05 --augment_feature_mask 0.02 --save_dir models/checkpoints/smoke_missing_aug`

Artifact:

- `results/smoke_missing_aug_report.json`

Smoke metrics:

- val accuracy: `75.59%`
- val F1: `75.97%`
- test accuracy: `77.86%`
- test F1: `78.01%`

## Cach dien giai trung thuc

Nhung so nay khong phai final benchmark.

Ly do:

- chi train `1 epoch`
- muc dich la verify train path moi, khong phai toi uu metric

Vi vay, ket luan dung la:

- code path moi da chay duoc that
- co the dung ngay de mo mot vong retrain nghiem tuc sau
- nhung chua du co so de noi rang missing-embedding + augmentation da vuot checkpoint hien tai

## Gia tri cho bao ve

- Ban khong chi noi future work bang loi noi; train path da duoc tich hop that vao codebase.
- Neu hoi dong hoi "neu cai thien tiep thi em se lam gi?" thi ban co the tro thang vao code/train flags da san sang.
