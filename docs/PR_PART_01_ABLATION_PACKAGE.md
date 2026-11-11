# PR Part 01 - Definitive Ablation Package

## Muc tieu

Chot mot goi so lieu va artifact "ke chuyen" cho hoi dong, tra loi ro:

- Moi modality dong gop gi?
- Cascade graph co gia tri them bao nhieu?
- Cross-attention co tot hon gated fusion khong?

## Thay doi chinh

1. Them script `scripts/paper_experiments/01_ablation_table.py`
   Script nay gom hai che do:
   - Neu co `all_results_*.json`: doc tu JSON de dung lai cho paper.
   - Neu JSON khong con trong repo: fallback ve bo so lieu da duoc xac thuc trong artifact hien co.

2. Chuan hoa bo config definitive cho defense
   Cac config duoc dat lai theo dung cau hoi khoa hoc:
   - `T`
   - `T+G`
   - `T+I`
   - `T+I+G`
   - `I`
   - `CrossAtt vs Gated`

3. Dinh nghia thong diep khoa hoc tu dong
   Script tinh san 4 delta can noi tren slide:
   - Full vs Graph-only
   - Full vs No-Graph
   - Full vs Text-only
   - Cross-Attention vs Gated

4. Tao output phuc vu paper
   - `ablation_definitive.png`
   - `ablation_definitive.tex`
   - `ablation_definitive_summary.md`

## Y nghia

Phan nay bien TN1 tu mot y tuong thanh mot goi artifact co the nhung vao:

- slide bao ve
- paper report
- phan Q&A khi giam khao hoi "dong gop lon nhat la gi?"

## Ghi chu thuc te

- Moi truong shell hien tai chua co `torch`, nen PR nay uu tien khoa bo so lieu va bo script tao hinh/latex truoc.
- Cac phan can evaluate checkpoint truc tiep se duoc noi vao mot shared harness o Part 2.
