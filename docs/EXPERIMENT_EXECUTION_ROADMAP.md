# Experiment Execution Roadmap

Tai lieu nay chia `docs/WINNING_EXPERIMENT_PLAN.md` thanh cac phan co the trien khai va review doc lap.

## Phan chia thuc thi

1. `Part 1 - Definitive Ablation Package`
   Muc tieu: chot bo so lieu so sanh config va thong diep khoa hoc co the dua vao slide/paper ngay.
   Dau ra:
   - `scripts/paper_experiments/01_ablation_table.py`
   - `results/paper_figures/ablation_definitive.png`
   - `results/paper_figures/ablation_definitive.tex`
   - `docs/PR_PART_01_ABLATION_PACKAGE.md`

2. `Part 2 - Shared Evaluation Harness`
   Muc tieu: tao bo helper dung chung cho TN5, TN2, TN3 de load graph test set, checkpoint, prediction va thong ke.
   Dau ra du kien:
   - `scripts/paper_experiments/common.py`
   - `docs/PR_PART_02_EVALUATION_HARNESS.md`

3. `Part 3 - TN5 Cascade Contribution`
   Muc tieu: tra loi cau hoi dong gop cot loi cua de tai: cascade co thuc su huu ich khong?
   Dau ra du kien:
   - `scripts/paper_experiments/05_cascade_contribution.py`
   - `results/cascade_contribution_report.json`
   - `docs/PR_PART_03_CASCADE_CONTRIBUTION.md`

4. `Part 4 - TN2 Error Analysis`
   Muc tieu: phan tich model sai o dau theo graph size, class, confidence va case study.
   Dau ra du kien:
   - `scripts/paper_experiments/02_error_analysis.py`
   - `results/error_analysis_report.md`
   - `docs/PR_PART_04_ERROR_ANALYSIS.md`

5. `Part 5 - TN3 Robustness Test`
   Muc tieu: do missing image, sparse cascade, adversarial comment injection.
   Dau ra du kien:
   - `scripts/paper_experiments/03_robustness_test.py`
   - `results/robustness_report.json`
   - `docs/PR_PART_05_ROBUSTNESS.md`

6. `Part 6 - TN4 Domain Shift`
   Muc tieu: danh gia kha nang generalize tren Reddit moi ngoai Fakeddit.
   Dau ra du kien:
   - `scripts/paper_experiments/04_domain_shift.py`
   - `data/ood_test_set.jsonl`
   - `results/domain_shift_report.json`
   - `docs/PR_PART_06_DOMAIN_SHIFT.md`

7. `Part 7 - Uncertain Zone And Calibration`
   Muc tieu: bo sung Responsible AI guardrail va calibration cho confidence.
   Dau ra:
   - `scripts/paper_experiments/calibrate_temperature.py`
   - `results/temperature_scaling_report.json`
   - `results/paper_figures/calibration_curve.png`
   - `docs/PR_PART_07_UNCERTAIN_AND_CALIBRATION.md`

8. `Part 8 - Keyword Leakage Test`
   Muc tieu: kiem chung model co hoc shortcut lexical tu cac tu khoa boc phot hay khong.
   Dau ra:
   - `scripts/paper_experiments/test_keyword_leakage.py`
   - `results/keyword_leakage_report.json`
   - `docs/PR_PART_08_KEYWORD_LEAKAGE.md`

9. `Part 9 - Attention Explainability`
   Muc tieu: trich top influential comments tu GAT attention de phuc vu demo va case study.
   Dau ra:
   - `scripts/paper_experiments/06_attention_explainability.py`
   - `results/explainability_report.json`
   - `results/paper_figures/attention_case_studies.png`
   - `docs/PR_PART_09_EXPLAINABILITY.md`

10. `Part 10 - Training Path Upgrades`
   Muc tieu: tich hop learned missing-image embedding va graph augmentation vao train path.
   Dau ra:
   - `src/training/train_multimodal_gnn.py`
   - `results/smoke_missing_aug_report.json`
   - `docs/PR_PART_10_TRAINING_UPGRADES.md`

## Thu tu khuyen nghi

1. Hoan thanh Part 1 de khoa thong diep va bo so lieu chinh.
2. Dung Part 2 lam nen cho TN5, TN2, TN3.
3. Chay Part 3 truoc vi day la dong gop chinh cua de tai.
4. Chay Part 4 va Part 5 de bo sung chieu sau khoa hoc.
5. Chot Part 6 cuoi cung vi phu thuoc crawl va gan nhan du lieu moi.

## Trang thai hien tai

- `Part 1`: Completed
- `Part 2`: Completed
- `Part 3`: Completed
- `Part 4`: Completed
- `Part 5`: Completed
- `Part 6`: Completed
- `Part 7`: Completed
- `Part 8`: Completed
- `Part 9`: Completed
- `Part 10`: Completed
