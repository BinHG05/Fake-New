# PR Part 07 - Uncertain Zone And Calibration

## Muc tieu

Bo sung 2 nang luc con thieu trong demo va inference:

- `UNCERTAIN` zone de tranh ep model phai luon chon REAL/FAKE
- temperature scaling de tach bai toan `ranking` ra khoi bai toan `probability calibration`

## Thay doi chinh

1. Cap nhat `src/deployment/predictor.py`
   - them `uncertain_threshold` voi mac dinh `0.15`
   - them `temperature` va tu dong nap calibration sidecar neu co
   - inference dung:
     - `softmax(logits / T)`
     - `UNCERTAIN` khi `|P(REAL) - P(FAKE)| < threshold`
   - batch summary co them:
     - `uncertain_count`
     - `coverage`
     - `margin`

2. Bo sung `scripts/paper_experiments/calibrate_temperature.py`
   Script nay:
   - load validation set
   - fit scalar temperature bang NLL minimization
   - luu sidecar calibration cho checkpoint
   - ve reliability diagram

3. Cap nhat demo frontend
   - them mau hien thi cho `UNCERTAIN`
   - them logic health check va render ket qua predict URL / batch predict

## File thay doi

- `src/deployment/predictor.py`
- `demo-website/script.js`
- `demo-website/styles.css`
- `scripts/paper_experiments/calibrate_temperature.py`

## Artifact tao ra

- `models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt.calibration.json`
- `results/temperature_scaling_report.json`
- `results/paper_figures/calibration_curve.png`

## Ket qua thuc te

Script da duoc chay bang:

- `D:/anacoda3/envs/multimodal_gnn/python.exe`

Ket qua validation calibration:

- best temperature: `1.101246`
- validation samples: `254`
- NLL truoc calibration: `0.3832`
- NLL sau calibration: `0.3790`
- ECE truoc calibration: `0.0486`
- ECE sau calibration: `0.0532`

## Cach dien giai trung thuc

Co 2 ket luan phai noi ro:

1. Temperature scaling co loi theo `NLL`, nghia la xac suat sau calibration hop ly hon theo tieu chi log-loss.
2. Tren validation set nay, `ECE` khong cai thien ma con xau di nhe.

Vi vay, calibration hien tai nen duoc dien giai la:

- da bo sung dung co che probabilistic
- da luu duoc temperature thuc te de demo khong con dung softmax thuan
- nhung khong nen noi qua rang model da "duoc calibrate dep" theo moi metric

## Gia tri cho bao ve

- `UNCERTAIN` zone giai quyet thang cau hoi Responsible AI: he thong biet khi nao no chua chac.
- calibration cho thay ban khong chi nhin accuracy/F1, ma con kiem soat y nghia cua confidence score.
