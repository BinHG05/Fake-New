# PR Part 05 - Robustness Test

## Muc tieu

Do do ben cua checkpoint hien tai trong 3 tinh huong:

- thieu image modality
- cascade bi cat ngắn
- bi chen them comment doi khang

## Thay doi chinh

1. Them `scripts/paper_experiments/03_robustness_test.py`
   Script nay chay 3 test:
   - `missing image`: zero phan image feature o inference time
   - `sparse cascade`: giu root + 1 comment, root + 3 comments
   - `adversarial injection`: chen them comment node gia o feature-space

2. Mo rong helper trong `scripts/paper_experiments/common.py`
   Them:
   - `zero_image_features()`
   - `append_comment_nodes()`

3. Tao 4 output cho paper/slide
   - `results/paper_figures/robustness_missing_image.png`
   - `results/paper_figures/robustness_sparse_cascade.png`
   - `results/paper_figures/robustness_adversarial.png`
   - `results/robustness_report.json`

## Ghi chu ky thuat quan trong

Phep `adversarial injection` trong PR nay la mot surrogate hop ly o feature-space:

- lay prototype comment node trung binh tu nhom FAKE
- chen them vao graph REAL dang duoc predict dung

No khong phai ban sao nguyen xi quy trinh encode text "This is fake news" tu XLM-R, nhung rat huu ich de do checkpoint co nhay cam voi fake-like comment signal hay khong.

## Ket qua thuc te

Script da duoc chay bang:

- `D:/anacoda3/envs/multimodal_gnn/python.exe`

### 1. Missing Image

- Full: `87.50%` accuracy, `84.72%` F1
- Missing image: `63.57%` accuracy, `68.52%` F1
- Delta: `-23.93 pp` accuracy, `-16.20 pp` F1

Ket luan truc tiep:

- checkpoint hien tai phu thuoc image modality rat manh

### 2. Sparse Cascade

Tren nhom `47` graph co hon `5` comment:

- Full: `78.72%`
- Root + 1 comment: `78.72%`
- Root + 3 comments: `78.72%`

Ket luan truc tiep:

- cat bot comment trong nhom nay khong lam metric thay doi
- dieu nay phu hop voi Part 3 va Part 4: checkpoint hien tai chua khai thac tot cascade phuc tap

### 3. Adversarial Injection

- Sample REAL dang predict dung: `50`
- So comment fake-like inject moi graph: `5`
- Flip count: `45`
- Flip rate: `90%`

Ket luan truc tiep:

- checkpoint rat nhay voi fake-like comment signal khi no duoc chen o feature-space
- nghia la model khong "gio`n" theo cach thieu image, nhung lai "gio`n" theo cach de bi thao tung boi comment signal doi khang

## Cach dien giai trung thuc

Part 5 bo sung mot buc tranh rat ro:

1. Image la tru cot thuc su cua checkpoint hien tai.
2. Cascade tu nhien dang khong tao loi ich ro rang trong test set.
3. Nhưng neu them fake-like comment signal, prediction co the bi lat rat manh.

Noi cach khac, checkpoint hien tai:

- manh o content multimodal, dac biet la image
- yeu o kha nang khang nhieu comment doi khang
- chua tan dung duoc thong tin cascade thuc te nhu ky vong ban dau
