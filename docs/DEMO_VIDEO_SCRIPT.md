# Demo Video Script

Tai lieu nay dung de quay video demo nhanh, dep, an toan, khong phu thuoc qua nhieu vao live behavior.

## 1. Muc tieu khi quay

- Mo dau bang mot phan demo on dinh, ket qua ra nhanh.
- Sau do moi chuyen sang live URL de cho thay pipeline end-to-end.
- Ket thuc bang bang ket qua va y nghia cua cac chi so.

Neu hoi dong co yeu cau test live truc tiep, co the chay sau. Ban quay video nen uu tien tinh muot va de thuyet trinh.

## 2. Thu tu quay khuyen nghi

1. Mo dashboard tai phan `Predict`.
2. Quay `Batch JSONL` truoc.
3. Quay `Science` URL.
4. Quay `TheOnion` URL.
5. Chuyen xuong `Training Results` de chot bang ket qua.

Ly do:

- Batch demo on dinh nhat.
- Science URL de giai thich truong hop tin tuc nghiem tuc.
- TheOnion URL de tao do tuong phan de hoi dong nhin ra ngay.
- Results section giup ket man bang so lieu, khong bi cam giac chi la UI.

## 3. Tai nguyen da chuan bi

### Batch file an toan

- `demo-website/demo_batch_curated.jsonl`

Mo ta:

- 6 mau da duoc chon san
- co nhan ground truth
- dung duoc cho doan quay mo dau

### URL mau de quay

#### Science A

`https://www.reddit.com/r/science/comments/1stxb1d/voting_is_linked_to_living_longer_among_older/`

#### Science B

`https://www.reddit.com/r/science/comments/1su8s27/new_findings_demonstrate_that_a_slight_daily/`

#### TheOnion A

`https://www.reddit.com/r/TheOnion/comments/1sti554/trump_conducts_marathon_reading_of_arbys_menu_in/`

#### TheOnion B

`https://www.reddit.com/r/TheOnion/comments/1sti55c/study_finds_gen_z_drinking_fewer_flagons_of_mead/`

Khuyen nghi:

- Dung `Science A` lam vi du chinh.
- Dung `TheOnion A` lam vi du satire.
- Giu `Science B` va `TheOnion B` lam backup.

## 4. Script quay ngan gon

### Doan 1 - Mo dau

Noi:

`Day la dashboard demo cho bai toan phat hien fake news da phuong thuc tren Reddit. He thong ket hop thong tin van ban, hinh anh va do thi lan truyen binh luan de dua ra du doan.`

### Doan 2 - Batch prediction

Thao tac:

1. Tai `demo-website/demo_batch_curated.jsonl`
2. Bam `Run Batch Prediction`

Noi:

`Dau tien em chay mot batch da duoc chuan bi san de minh xem he thong hoat dong on dinh tren tap du lieu co nhan. O day dashboard tra ve coverage, accuracy, F1, va tung du doan cho moi mau.`

Can chi vao:

- `Total items`
- `Coverage`
- `Accuracy / F1`
- bang item-level prediction

### Doan 3 - Science URL

Thao tac:

1. Bam nut `Science A` hoac `Science B`
2. Bam `Predict URL`

Noi:

`Tiep theo em demo suy luan tren mot bai Reddit moi. He thong se lay bai viet, thu comment cascade, tao graph, trich xuat dac trung, va dua ra ket luan cuoi cung.`

Khi ket qua ra, noi theo thu tu:

1. `Prediction`
2. `Confidence`
3. `Margin`
4. `Binary probabilities`
5. `Nodes / Edges`

Cau noi mau:

`Khi trinh bay ket qua, em uu tien giai thich nhan du doan truoc, sau do den do tin cay, do chenh lech giua hai lop, va cuoi cung la quy mo graph ma model da dung.`

### Doan 4 - TheOnion URL

Thao tac:

1. Bam `TheOnion A`
2. Bam `Predict URL`

Noi:

`Cuoi cung em thu voi mot bai theo phong cach cham bien cua TheOnion. Vi du nay giup hoi dong nhin ro hon cach he thong phan biet noi dung mang tinh fake-like hoac satire.`

Neu co warning thi noi them:

`Neu co warning, dieu do co nghia la mot phan input nhu image hoac comments chua lay duoc, nen can doc ket qua than trong hon.`

## 5. Cach noi cho dep va gon

Moi case chi can theo mot khung:

1. Bai nay la gi.
2. He thong du doan REAL hay FAKE.
3. Do tin cay bao nhieu.
4. Margin lon hay nho.
5. Graph co bao nhieu node va edge.

Tranh noi dai dong ve kien truc trong luc model dang xu ly. Phan kien truc chi nhac 1 cau la du:

`Model ket hop thong tin noi dung va thong tin lan truyen trong comment graph thay vi chi doc tieu de.`

## 6. Checklist truoc khi bam quay

- Dashboard mo duoc va giao dien dung.
- `Predictor Health` hien `Ready`.
- Batch file `demo-website/demo_batch_curated.jsonl` san sang.
- Thu truoc it nhat 1 science URL va 1 TheOnion URL.
- Neu URL A loi, doi sang URL B.
- Cuon san den section `Predict` truoc khi quay.
- Neu can, bat `Presentation Mode` tren top bar.

## 7. Backup plan

Neu live URL bi loi trong luc quay:

- Giu doan batch lam doan chinh.
- Quay 1 URL thanh cong duy nhat.
- Chuyen sang `Training Results` de ket thuc.

Cau noi backup:

`Phan batch giup xac nhan he thong tren du lieu da gan nhan, con phan URL cho thay cung backend do co the suy luan tren bai Reddit moi theo thoi gian thuc.`
