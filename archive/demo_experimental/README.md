# Huong dan su dung Web Demo

Tai lieu nay mo ta cach chay va cach dung `demo-website` bang tieng Viet, gon va dung cho nhu cau demo.

## 1. Cach mo Web Demo

Tu thu muc goc project:

```powershell
cd demo-website
python app.py
```

Sau do mo trinh duyet tai:

`http://localhost:5000`

## 2. Can chuan bi gi truoc khi demo

- Moi truong Python dung de chay `app.py`
- Model checkpoint va cac file phu tro da co san
- Neu demo live URL thi can co ket noi mang
- Neu demo batch thi dung file:

`demo-website/demo_batch_curated.jsonl`

## 3. Cach dung nhanh

Man hinh demo co 3 khu vuc chinh:

- `Predict`
- `Training`
- `Training Results`

### Predict URL

1. Bam mot trong cac nut URL mau o khung `Demo Inputs`, hoac tu dan URL Reddit vao.
2. Bam `Predict URL`.
3. Xem ket qua o ben phai:
   - nhan du doan
   - confidence
   - margin
   - post info
   - graph info
   - model info

### Run Batch Prediction

1. O muc `Batch JSONL`, chon file `demo-website/demo_batch_curated.jsonl`.
2. Bam `Run Batch Prediction`.
3. He thong se hien:
   - tong so mau
   - coverage
   - accuracy / F1
   - bang ket qua tung dong

### Training

1. Chon model trong `Training Configuration`.
2. Chinh `epochs`, `learning rate`, `batch size` neu can.
3. Bam `Start Training` de chay mot model.
4. Bam `Train All` neu muon chay ca chuoi.

### Training Results

Phan nay doc tu cac ket qua da co san trong project va hien:

- kich thuoc dataset
- split train / val / test
- champion accuracy
- champion F1
- bang so sanh
- hinh tong hop ket qua

## 4. Cac nut nhanh da co san

Trong `Demo Inputs` da co:

- `Science A`
- `Science B`
- `TheOnion A`
- `TheOnion B`

Bam vao la URL duoc dua san vao o nhap.

Ngoai ra co nut `Copy Path` de copy nhanh duong dan:

`demo-website/demo_batch_curated.jsonl`

## 5. Loi thuong gap

### Predictor Health khong hien Ready

Nguyen nhan thuong la model chua load duoc hoac thieu dependency.

### Batch prediction loi

Kiem tra lai:

- file JSONL co ton tai
- file dung dinh dang
- cac asset can thiet cho predictor van con

### URL prediction loi

Thuong do:

- URL Reddit khong con truy cap duoc
- mang khong on dinh
- Reddit khong tra ve du lieu day du

Khi do nen doi sang URL mau khac.

## 6. Ghi chu cho demo

- Neu quay video, nen uu tien `Batch JSONL` truoc.
- Neu can giao dien de quay dep hon, bat `Presentation Mode`.
- Neu hoi dong muon test truc tiep, co the dung cac URL mau da gan san tren giao dien.
