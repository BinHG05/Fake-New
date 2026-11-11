# PR Part 04 - Error Analysis

## Muc tieu

Tra loi ro 4 cau hoi:

- Model sai nhieu hon o graph nho hay graph lon?
- Model thien ve REAL hay FAKE?
- Confidence co dang tin hay khong?
- Nhung case sai dien hinh co pattern gi?

## Thay doi chinh

1. Them `scripts/paper_experiments/02_error_analysis.py`
   Script nay:
   - chay prediction tren test split binary
   - ve confusion matrix
   - tinh accuracy theo bucket graph size
   - phan tich confidence theo 3 band
   - rut top false positive va false negative de lam case study

2. Tao 4 artifact dung cho bao ve
   - `results/paper_figures/confusion_matrix.png`
   - `results/paper_figures/accuracy_by_graph_size.png`
   - `results/paper_figures/confidence_reliability.png`
   - `results/error_analysis_report.md`

3. Dung chung evaluation harness cua Part 2
   Nho vay TN2 va TN5 dung cung checkpoint path, split logic va prediction format.

## Y nghia

Part nay giup ban khong chi dua metric tong, ma con noi duoc:

- model nghiêng ve class nao
- confidence co phan cap hop ly hay khong
- nhung bai sai manh thuong co cascade/text nhu the nao

## Ket qua thuc te

Script da duoc chay bang:

- `D:/anacoda3/envs/multimodal_gnn/python.exe`

Ket qua tong quan:

- Overall accuracy: `87.50%`
- REAL precision / recall: `88.10% / 90.80%`
- FAKE precision / recall: `86.61% / 82.91%`

Theo graph size:

- `0 comments`: `92.86%` tren `168` mau
- `1-10 comments`: `82.50%` tren `80` mau
- `>10 comments`: `71.88%` tren `32` mau

Theo confidence band:

- `medium (0.5-0.7)`: `65.62%` accuracy, avg confidence `60.12%`
- `high (>0.7)`: `90.32%` accuracy, avg confidence `91.14%`
- `low (<0.5)`: khong co mau

Case study noi bat:

- False Positive manh thuong la bai REAL co wording gay soc, celebrity/politics, hoac cascade sau.
- False Negative manh thuong la bai FAKE rat ngan, mo ho, meme-like, it comment, khien model nghieng sang REAL.

## Cach dien giai trung thuc

Part 4 bo sung rat hop voi Part 3:

1. Checkpoint hien tai khong bi lech class qua cuc doan, nhung recall cua FAKE thap hon REAL.
2. Confidence co y nghia: khi model rat tu tin thi accuracy cung cao.
3. Dieu dang luu y nhat la accuracy giam khi graph lon hon, nghia la checkpoint hien tai chua bien cascade phuc tap thanh loi the.

Noi cach khac, TN2 da giup chuyen phat hien o TN5 thanh mot cau chuyen ro hon ve hanh vi model.
