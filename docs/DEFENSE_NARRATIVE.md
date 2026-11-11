# Defense Narrative

## Ban mo dau

De tai cua em xuat phat tu mot cau hoi rat thuc te:

> Neu chi nhin vao noi dung bai viet thi co du de phat hien tin gia tren mang xa hoi khong?

Cau tra loi cua em la: chua du.

Ly do la vi tren Reddit, mot bai viet khong chi ton tai o dang text hay image.
No con di kem cach nguoi dung phan ung, binh luan, tranh luan va lan truyen thanh mot comment cascade.
Vi vay, muc tieu cua de tai la xay dung mot he thong phat hien tin gia da phuong thuc, ket hop:

- text
- image
- va propagation graph tu comment cascade

De tai khong chi dung lai o muc train model, ma di den mot pipeline end-to-end:

- crawl du lieu
- xu ly text va image
- build cascade graph
- train checkpoint
- va infer truc tiep tren bai Reddit moi

## Dong gop chinh

Dong gop thu nhat cua de tai la ve he thong.
Em da xay dung duoc mot pipeline hoan chinh cho bai toan fake news detection tren Reddit, tu du lieu tho den infer thuc te.

Dong gop thu hai la ve mo hinh.
He thong ket hop text embedding, image embedding va graph neural network tren cascade graph, trong do fusion giua modalities duoc thuc hien bang cross-attention.

Dong gop thu ba la ve phan tich khoa hoc.
Em khong chi bao cao mot con so accuracy tong, ma da chia bai toan thanh cac nhom thi nghiem rieng:

- ablation study
- cascade contribution
- error analysis
- robustness test
- domain shift evaluation

Chinh nhung phan tich nay cho phep em khong chi noi model tot hay xau, ma con noi model tot o dau, yeu o dau, va vi sao.

## Ket qua noi bat

Tren tap test in-distribution binary, checkpoint hien tai dat:

- Accuracy: `87.50%`
- Binary F1: `84.72%`

Day la ket qua cho thay he thong hoat dong tot trong phan phoi du lieu gan voi tap train.

Trong ablation study, fusion bang cross-attention vuot troi hon gated fusion.
Dieu nay ung ho gia thuyet rang co che attention phu hop hon cho bai toan ket hop da phuong thuc.

Ngoai ra, phan robustness test cho thay image dong vai tro rat manh:

- khi loai bo image feature o inference time, accuracy giam `23.93` diem phan tram

Nghia la checkpoint hien tai khong chi la multimodal theo ten goi, ma thuc te dang dua rat nhieu vao image signal.

## Phat hien quan trong nhat

Phat hien quan trong nhat cua em lai khong nam o con so cao nhat, ma nam o dieu du lieu va model dang ke khac nhau.

Tu phan tich topology, em thay cascade REAL va FAKE thuc su khac nhau ro rang:

- depth khac nhau co y nghia thong ke
- branching factor khac nhau co y nghia thong ke
- average degree khac nhau co y nghia thong ke

Noi cach khac, comment cascade co tin hieu cau truc that su.

Tuy nhien, khi em lam thi nghiem cat bo cascade, giu lai root node, hoac chi giu root + 1 comment, root + 3 comments, ket qua gan nhu khong doi.

Dieu nay dan den mot ket luan trung thuc va rat quan trong:

> Du lieu cascade co tin hieu, nhung checkpoint hien tai chua khai thac duoc tin hieu do mot cach hieu qua.

Day la mot ket qua khoa hoc co gia tri, vi no cho thay su khac nhau giua:

- “tin hieu ton tai trong du lieu”
- va “mo hinh da hoc duoc tin hieu do hay chua”

## Error analysis

Error analysis giup em hieu ro hanh vi cua checkpoint hien tai.

Thu nhat, model khong bi lech class qua muc, nhung recall cua FAKE thap hon REAL.
Nghia la model van bo sot mot phan bai FAKE.

Thu hai, accuracy giam khi graph lon hon:

- `92.86%` voi bai khong co comment
- `82.50%` voi nhom `1-10` comment
- `71.88%` voi nhom `>10` comment

Neu cascade thuc su dang duoc khai thac tot, ta ky vong dieu nguoc lai.
Vi vay, phan nay cung ung ho phat hien rang model hien tai chua dung cascade phuc tap thanh loi the.

Thu ba, confidence score co y nghia trong in-distribution:

- nhom high confidence dat hon `90%` accuracy
- nhom medium confidence thap hon ro ret

Tuc la trong phan phoi gan tap train, model biet luc nao no tu tin hon.

## Robustness va gioi han

Robustness test cho thay 2 mat rat khac nhau cua he thong.

Mat thu nhat, khi mat image thi hieu nang giam rat manh.
Dieu nay cho thay checkpoint hien tai phu thuoc image modality rat ro.

Mat thu hai, khi em inject cac fake-like comment node vao bai REAL dang duoc du doan dung, ty le bi lat sang FAKE len den `90%`.

Dieu nay cho thay model rat nhay voi social signal doi khang.
No khong chi ra mot diem yeu ve robustness, ma con goi y rang phan graph/social component hien tai de bi thao tung.

## Domain shift

Phan nghiem khac nghiet nhat cua de tai la domain shift.
Em lay 30 bai Reddit moi, ngoai tap train:V

- 15 bai REAL tu `worldnews`, `science`, `neutralnews`
- 15 bai FAKE tu `TheOnion`

Ket qua:

- OOD accuracy chi con `33.33%`
- model sai `15/15` bai REAL moi
- model dung `10/15` bai `TheOnion`

Quan trong hon, confidence trung binh cua nhung mau sai lai cao hon nhung mau dung.
Nghia la duoi domain shift, model khong chi sai ma con overconfident.

Day la gioi han lon nhat cua checkpoint hien tai neu xem bai toan nhu mot he thong phat hien tin gia tong quat tren Reddit moi.

## Cach ket luan trung thuc

Neu em phai tom tat de tai trong vai cau, em se noi nhu sau:

1. Em da xay dung duoc mot he thong multimodal fake news detection end-to-end tren Reddit, ket hop text, image va cascade graph.
2. Trong in-distribution, he thong dat ket qua binary kha tot va cho thay image la modality dong gop rat manh.
3. Phan tich sau cho thay cascade co tin hieu cau truc that su, nhung checkpoint hien tai chua khai thac duoc tin hieu do thanh gain performance.
4. Hai gioi han lon nhat cua he thong hien tai la generalization ra Reddit moi va do ben truoc social adversarial signal.

## Huong phat trien

Tu toan bo ket qua, huong phat trien tiep theo rat ro:

- giam phu thuoc image bang modality dropout
- tai huan luyen de buoc model dung cascade that su
- adversarial training voi fake-like comments
- va quan trong nhat la domain adaptation tren tap Reddit moi co gan nhan

Neu lam duoc cac buoc nay, de tai se chuyen tu muc “pipeline va checkpoint tot trong phan phoi gan tap train”
len muc “he thong co kha nang generalize va robust hon trong boi canh thuc te”.

## Ban ket

Gia tri lon nhat cua de tai nay khong chi o mot con so accuracy.
Gia tri lon nhat nam o cho em da xay dung duoc:

- mot he thong chay duoc
- mot bo thi nghiem day du
- va mot buc tranh trung thuc ve diem manh, diem yeu, va gioi han cua mo hinh

Em tin day la nen tang tot de tiep tuc nang cap thanh mot huong nghien cuu sau hon, chac hon va co gia tri ung dung thuc te hon.
