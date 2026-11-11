# Final Review And Improvements

Tai lieu nay tong hop nhung diem can luu y va huong cai thien sau khi hoan tat cac Part trong `WINNING_EXPERIMENT_PLAN`.

## Muc tieu

- tom tat nhung phat hien quan trong nhat
- chi ra diem manh that su cua he thong
- chi ra gioi han can noi trung thuc khi bao ve
- de xuat cac huong cai thien co gia tri cao nhat

## Trang thai

Da cap nhat sau khi Part 10 hoan tat.

## Tong hop phat hien quan trong

1. Trong in-distribution binary test, checkpoint hien tai dat ket qua cao:
   - Accuracy `87.50%`
   - Binary F1 `84.72%`

2. Ve dong gop modality:
   - image dong vai tro rat manh
   - cross-attention vuot gated fusion ro rang trong bo ket qua da co

3. Ve cascade:
   - du lieu cascade REAL va FAKE khac nhau ro rang ve topology
   - nhung checkpoint hien tai khong bien duoc khac biet do thanh gain performance
   - cat bo comment nodes hoac giu root+1/root+3 deu gan nhu khong doi ket qua

4. Ve do ben:
   - thieu image lam accuracy giam rat manh (`-23.93 pp`)
   - fake-like adversarial comment injection lam flip `90%` tren mau REAL dang dung

5. Ve generalization:
   - OOD accuracy chi con `33.33%`
   - model sai toan bo `15/15` bai REAL moi
   - model overconfident duoi domain shift

6. Ve Responsible AI va confidence:
   - da bo sung `UNCERTAIN` zone voi margin threshold `0.15`
   - temperature scaling tim duoc `T = 1.101246`
   - NLL cai thien nhe sau calibration
   - nhung ECE tren validation khong dep hon, nen can noi trung thuc rang calibration moi chi giai quyet duoc mot phan van de confidence

7. Ve keyword leakage:
   - mask cac node chua keyword boc phot khong lam metric thay doi
   - voi bo keyword dang test, checkpoint khong cho thay dau hieu phu thuoc shortcut lexical

8. Ve explainability:
   - da trich duoc GAT attention cho case study
   - co the hien thi top influential comments trong demo
   - nhung nhieu graph test set co `0 comment`, nen XAI theo social cascade se thuong rat ngan

9. Ve train-path future work:
   - learned missing-image embedding va graph augmentation da duoc tich hop vao code
   - da co smoke run xac nhan train path moi hoat dong
   - chua co retrain day du de ket luan loi ich metric

## Diem manh that su cua project

- Ban da xay duoc mot pipeline end-to-end co the crawl, build graph, infer va sinh artifact bao ve.
- Ban co bo evidence day du: ablation, error analysis, robustness, domain shift.
- Ban dang co mot ket qua khoa hoc trung thuc, khong chi mot dashboard dep.
- He thong nhan ra satire ro rang o mot phan dang ke cua `TheOnion`.

## Diem can noi trung thuc khi bao ve

1. `Cascade signal exists in data, but the current checkpoint does not exploit it well.`
   Day la cau rat quan trong vi Part 3, 4, 5 deu cung chi ve mot huong.

2. `The current checkpoint is strongly image-dependent.`
   Missing image lam metric roi rat ro.

3. `The model is vulnerable to fake-like adversarial comment signals.`
   Flip rate `90%` la con so khong nen giau.

4. `Generalization to fresh Reddit posts is currently weak.`
   Part 6 cho thay day la gioi han lon nhat neu xem bai toan la production-like fake news detection.

## Uu tien cai thien cao nhat

1. Cai thien domain adaptation / recalibration
   - finetune tren tap Reddit moi co gan nhan
   - them calibration layer hoac temperature scaling
   - danh gia lai confidence duoi OOD

2. Buoc tai huan luyen de buoc model dung cascade that su
   - tang trong so cho graph message passing
   - so sanh voi training scheme phat de root-only shortcut
   - bo sung loss/regularizer de phan biet root signal va cascade signal

3. Giam phu thuoc image
   - train voi modality dropout
   - random image masking trong training
   - bao dam model van dung duoc khi image vang mat

4. Tang suc de khang comment adversarial
   - adversarial training voi spam/fake-like comments
   - robust aggregation tren comment nodes
   - pruning / trust scoring cho social nodes

5. Tao bo OOD benchmark co gan nhan thu cong tot hon
   - 30-100 bai Reddit moi
   - can bang news, science, politics, satire
   - luu ro tieu chi gan nhan va ngay crawl

6. Danh gia calibration sau OOD va theo class
   - hien tai calibration moi fit tren validation in-distribution
   - can do them ECE theo tung mien du lieu va tung nhom confidence
   - neu can, can nhac classwise calibration hoac conformal-style abstention

7. Mo mot vong retrain day du cho `missing_image_emb + augmentation`
   - smoke run da xac nhan code path
   - buoc tiep theo la train du epoch, luu checkpoint rieng, va lap lai Parts 3-6 tren checkpoint moi

## Cach ke chuyen de bao ve

Neu can noi rat gon, ban co the chot theo 4 y:

1. Chung toi xay dung duoc he thong multimodal graph fake-news detection chay end-to-end.
2. Trong phan phoi gan tap train, he thong dat ket qua binary cao va rat phu thuoc image.
3. Phan tich sau cho thay cascade co tin hieu cau truc, nhung checkpoint hien tai chua khai thac tin hieu do tot.
4. Domain shift va adversarial social signal la hai gioi han lon nhat, va cung chinh la huong nghien cuu tiep theo ro rang nhat.
5. He thong da co `UNCERTAIN` zone va co test keyword leakage, nen lap luan Responsible AI va validity hien da chat hon ban dau.
6. Ban da co mot lop explainability va train-path upgrades that su trong codebase, nen huong phat trien sau bao ve khong con la y tuong mo ho nua.
