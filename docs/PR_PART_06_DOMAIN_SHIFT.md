# PR Part 06 - Domain Shift

## Muc tieu

Tra loi cau hoi:

- model co generalize duoc ra ngoai tap train Fakeddit hay khong?
- khi gap Reddit moi, confidence co giam hop ly hay khong?

## Thay doi chinh

1. Them `scripts/paper_experiments/04_domain_shift.py`
   Script nay:
   - lay bai Reddit moi tu `worldnews`, `science`, `neutralnews` va `TheOnion`
   - gan nhan theo chinh sach nguon
   - chay `FakeNewsPredictor.predict_url()` tren tung bai
   - ghi `ood_test_set.jsonl`, `domain_shift_report.json` va figure so sanh

2. Tao bo output cho defense
   - `data/ood_test_set.jsonl`
   - `results/domain_shift_report.json`
   - `results/paper_figures/domain_shift_comparison.png`

## Ghi chu quan trong

Part nay la live OOD evaluation, nen no phu thuoc:

- Reddit con truy cap duoc
- image/comment endpoint con phan hoi
- checkpoint demo `predict_url()` van o trang thai su dung duoc

## Cach dien giai dung

Day la bai test ngoai phan phoi train, nen no co gia tri bao ve rat tot.
Neu OOD giam manh so voi in-distribution, do khong phai that bai; no la bang chung trung thuc ve gioi han generalization cua mo hinh hien tai.

## Ket qua thuc te

Script da duoc chay live bang:

- `D:/anacoda3/envs/multimodal_gnn/python.exe`

Chinh sach tao tap OOD:

- REAL: `5 worldnews + 5 science + 5 neutralnews`
- FAKE: `15 TheOnion`

Ket qua:

- In-distribution accuracy: `87.50%`
- OOD accuracy: `33.33%`
- Delta accuracy: `-54.17 pp`
- In-distribution F1: `84.72%`
- OOD F1: `50.00%`
- OOD avg confidence: `77.40%`

Phan bo theo nhan:

- REAL OOD: `0/15` dung, accuracy `0.00%`
- FAKE OOD: `10/15` dung, accuracy `66.67%`

Dieu dang luu y:

- confidence trung binh cua mau sai (`80.02%`) cao hon mau dung (`72.15%`)
- nghia la duoi domain shift, model khong chi sai ma con overconfident

## Cach dien giai trung thuc

Part 6 cho thay checkpoint hien tai generalize rat kem ra ngoai Fakeddit-style distribution.

No co xu huong:

- gan cac bai news/science/neutral moi vao `FAKE`
- giu duoc mot phan kha nang bat satire ro rang tu `TheOnion`

Noi cach khac, model hien tai co the dung tot trong phan phoi gan tap train,
nhung chua san sang de xem la mot he thong phat hien tin gia tong quat cho Reddit moi.
