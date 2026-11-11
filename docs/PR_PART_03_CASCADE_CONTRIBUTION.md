# PR Part 03 - Cascade Contribution

## Muc tieu

Bien dong gop cot loi cua de tai thanh mot script co the chay lai:

> Comment cascade co thuc su huu ich khong?

## Thay doi chinh

1. Them `scripts/paper_experiments/05_cascade_contribution.py`
   Script nay thuc hien 3 viec:
   - so sanh `with cascade` va `root only`
   - tinh accuracy theo bucket so comment `0`, `1-5`, `6-20`, `>20`
   - thong ke topology REAL vs FAKE

2. Tao bo output dung cho defense
   - `cascade_vs_no_cascade.png`
   - `accuracy_vs_cascade_size.png`
   - `topology_real_vs_fake.png`
   - `cascade_contribution_report.json`

3. Dung lai shared harness
   TN5 la script dau tien dung `common.py`, nen no dong vai tro mau cho TN2 va TN3.

## Y nghia khoa hoc mong muon

Neu report cho thay:

- `with cascade` > `root only`
- delta tang o nhom `6-20` va `>20`
- topology REAL va FAKE co phan bo khac nhau

thi ban da co mot luan diem rat manh de bao ve dong gop chinh cua de tai.

## Ghi chu

Script da duoc chay thanh cong bang env:

- `D:/anacoda3/envs/multimodal_gnn/python.exe`

## Ket qua thuc te

- `With cascade`: `87.50%` accuracy, `84.72%` binary F1
- `Root only`: `87.50%` accuracy, `84.72%` binary F1
- `Delta`: `0.00 pp`

Theo bucket so comment:

- `0 comments`: `92.86%`
- `1-5 comments`: `80.00%`
- `6-20 comments`: `78.79%`
- `>20 comments`: `78.57%`

Thong ke topology van rat manh:

- REAL co `depth` trung binh `0.70`, FAKE la `1.56`
- REAL co `branching_factor` trung binh `0.45`, FAKE la `1.15`
- Cac phep kiem dinh Mann-Whitney deu cho `p < 1e-8`

## Cach dien giai trung thuc

Ket qua nay cho thay hai dieu:

1. Ve mat cau truc du lieu, cascade REAL va FAKE khac nhau ro rang co y nghia thong ke.
2. Nhưng voi checkpoint binary hien tai, khi cat comment nodes chi giu root node thi metric khong giam.

Dieu nay goi y rang:

- checkpoint hien tai co the dang dua chu yeu vao root content feature
- hoac training setup chua khai thac manh thong tin message passing tu comment cascade

Noi cach khac, TN5 da sanh ra mot ket qua co gia tri khoa hoc, du ket qua do khong trung voi gia thuyet ky vong ban dau.
