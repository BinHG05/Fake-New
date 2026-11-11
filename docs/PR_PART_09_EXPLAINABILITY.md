# PR Part 09 - Attention Explainability

## Muc tieu

Bo sung mot lop giai thich cho checkpoint GAT hien tai:

- comment nao duoc model chu y nhieu nhat
- co the trich ra top influential comments cho tung case
- co artifact de dua vao slide bao ve

## Thay doi chinh

1. Mo rong `MultimodalCascadeGNN`
   - `forward(..., return_attention=True)` co the tra them attention weights tu cac layer GAT

2. Mo rong `predictor.py`
   - attach node texts vao graph luc inference
   - trich `top_influential_comments` tu layer attention dau tien
   - tra ve top-3 comment co attention cao nhat trong output API neu co

3. Cap nhat frontend
   - hien thi `Top Influential Comments` trong ket qua predict URL

4. Them script:
   - `scripts/paper_experiments/06_attention_explainability.py`

## Artifact tao ra

- `results/explainability_report.json`
- `results/paper_figures/attention_case_studies.png`

## Ket qua thuc te

Script da duoc chay bang:

- `D:/anacoda3/envs/multimodal_gnn/python.exe`

Ket qua:

- test graphs da quet: `280`
- da trich duoc 3 case study:
  - `correct_real`
  - `correct_fake`
  - `incorrect_high_conf`

Case ro nhat trong report hien tai:

- `post_id = 8u71mr`
- label that: `FAKE`
- model doan dung voi confidence `98.66%`
- comment node co attention cao nhat:
  - node `1`
  - attention score `0.50`

## Luu y quan trong

Explainability hien tai co 2 gioi han:

1. Day la attention-based explanation, khong phai causal proof.
2. Nhieu graph trong test set co `0 comment`, nen se khong co top comment de hien thi.

## Gia tri cho bao ve

- Ban da co mot lop XAI that su thay vi chi noi ly thuyet.
- Co the demo live mot bai Reddit va cho hoi dong thay comment nao dang duoc model chu y nhieu nhat.
- No cung cung co mot thong diep khop voi cac Part truoc: nhieu mau test set gan nhu khong co cascade de khai thac.
