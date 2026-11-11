# PR Part 08 - Keyword Leakage Test

## Muc tieu

Kiem chung xem checkpoint co dang an shortcut tu cac tu khoa boc phot nhu:

- `fake`
- `hoax`
- `debunked`
- `false`
- `source`

Neu bo cac node chua tu nay ma metric roi manh, model co nguy co hoc shortcut thay vi hoc tin hieu noi dung/cascade that su.

## Thay doi chinh

1. Them `scripts/paper_experiments/test_keyword_leakage.py`
   Script nay:
   - load test graphs
   - map graph ve metadata test
   - tim cac node co text chua keyword
   - thay embedding cua cac node do bang mean embedding cua graph
   - chay lai prediction va so sanh voi baseline

2. Khong sua training hay checkpoint
   Day la mot test inference-time de tra loi nghi ngo valid cua hoi dong.

## Artifact tao ra

- `results/keyword_leakage_report.json`

## Ket qua thuc te

Script da duoc chay bang:

- `D:/anacoda3/envs/multimodal_gnn/python.exe`

So lieu:

- test samples: `280`
- affected graphs: `14`
- total masked nodes: `19`

Baseline:

- accuracy: `87.50%`
- F1: `84.72%`

Sau khi mask keyword nodes:

- accuracy: `87.50%`
- F1: `84.72%`

Delta:

- accuracy: `0.00 pp`
- F1: `0.00 pp`

## Cach dien giai trung thuc

Voi bo keyword dang test, checkpoint hien tai khong cho thay dau hieu phu thuoc vao shortcut lexical nay.

No khong co nghia la model hoan toan mien nhiem voi moi kieu leakage, nhung no tra loi rat tot mot nghi ngo cu the:

- bo keyword boc phot trong comment khong lam ket qua xau di

## Gia tri cho bao ve

Day la mot bang chung rat de noi va rat co luc:

- model khong chi bat chu `fake` roi ket luan FAKE
- metric giu nguyen sau khi mask keyword nodes cho thay checkpoint dang dua nhieu hon vao tin hieu khac
