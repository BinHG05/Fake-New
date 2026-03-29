# Team Research Pipeline

Tai lieu nay mo ta luong lam viec chuan cho nhom khi bo sung du lieu Reddit, gan nhan, enrich comment tree, build graph va train model.

Muc tieu:
- Moi thanh vien co the chay dung cung mot quy trinh.
- Khong merge nham, khong tao ban ghi trung trong `labeled_master.jsonl`.
- Sau moi dot gan nhan, team co the nhanh chong train lai de kiem tra metric.

## Tong quan 6 buoc

1. Cao du lieu Reddit moi.
2. Chay pipeline xu ly de tao batch cho Label Studio.
3. Gan nhan, export JSON, merge vao file tong.
4. Enrich de lay comment tree.
5. Build graph text va multimodal.
6. Train va kiem tra ket qua.

Entry point chinh:

```powershell
python src/utils/research_pipeline.py --help
```

Windows menu runner:

```powershell
run_team_cycle.bat
```

## File quan trong

- Raw crawl: `data/01_raw/reddit/reddit_realtime_data.jsonl`
- Batch cho Label Studio: `data/03_clean/Reddit/<batch_name>/reddit_for_ls.json`
- Master da gan nhan: `data/03_clean/Fakeddit/labeled_master.jsonl`
- Master binary: `data/03_clean/Fakeddit/labeled_master_binary.jsonl`
- Enriched data: `data/reddit_enriched_data.jsonl`
- Enriched binary: `data/reddit_enriched_binary.jsonl`
- Graph text-only: `data/processed_graphs/`
- Graph multimodal: `data/processed_graphs_multimodal/`

## Buoc 1: Crawl Reddit

Muc dich:
- Lay bai viet Reddit moi.
- Day vao file raw chung de xu ly tiep.

Lenh co ban:

```powershell
python src/utils/research_pipeline.py crawl --limit 25
```

Lenh neu uu tien bai co anh:

```powershell
python src/utils/research_pipeline.py crawl --limit 25 --images-only
```

Ghi chu:
- Script nay goi `src/data/reddit_crawler.py`.
- Neu crawl lai, script se bo qua ID da ton tai trong file raw.

Output:
- `data/01_raw/reddit/reddit_realtime_data.jsonl`

## Buoc 2: Tao batch cho Label Studio

Muc dich:
- Tach du lieu Reddit moi.
- Chay image preprocessing va text preprocessing.
- Tao file JSON de import vao Label Studio.

Che do tu dong:

```powershell
python src/utils/research_pipeline.py prepare-label
```

Che do chi dinh batch:

```powershell
python src/utils/research_pipeline.py prepare-label --start 0 --count 50
```

Ghi chu:
- Script nay goi `src/utils/reddit_pipeline.py`.
- Batch sau khi xu ly nam trong `data/03_clean/Reddit/<batch_name>/`.
- File team dung de import Label Studio la `reddit_for_ls.json`.

Output quan trong:
- `data/03_clean/Reddit/<batch_name>/Reddit/train.jsonl`
- `data/03_clean/Reddit/<batch_name>/reddit_for_ls.json`

## Buoc 3: Gan nhan va merge vao master

### 3.1 Gan nhan tren Label Studio

Thanh vien labeling thuc hien:
- Import file `reddit_for_ls.json`
- Gan nhan
- Export duoi dang JSON

Vi du file export:
- `data/03_clean/Reddit/reddit_run_001/export_reddit_run_001.json`

### 3.2 Merge vao file tong

Dung lenh sau:

```powershell
python src/utils/research_pipeline.py merge-labels --input "data/03_clean/Reddit/reddit_run_001/export_reddit_run_001.json"
```

Lenh nay se:
- Merge theo `id` vao `data/03_clean/Fakeddit/labeled_master.jsonl`
- Neu `id` da ton tai trong master hoac bi lap ngay trong file export, record do se bi bo qua, khong merge vao
- Refresh lai `data/03_clean/Fakeddit/labeled_master_binary.jsonl`
- Trong file binary, `label` se la `REAL/FAKE`, con nhan 6-class cu duoc giu trong `label_6class`

Script duoc goi:
- `src/utils/merge_ls_export_by_id.py`
- `src/utils/build_binary_labeled_master.py`

Checkpoint sau buoc nay:
- `labeled_master.jsonl` da co them record moi hoac cap nhat record cu
- `labeled_master_binary.jsonl` da san sang cho baseline training

## Buoc 4: Enrich de lay comment tree

Muc dich:
- Doc `labeled_master.jsonl`
- Goi Reddit de lay comment tree
- Tao file enriched cho graph pipeline

Lenh co ban:

```powershell
python src/utils/research_pipeline.py enrich
```

Lenh test nho:

```powershell
python src/utils/research_pipeline.py enrich --limit 20 --delay 1.0
```

Lenh nay se:
- Tao hoac noi tiep `data/reddit_enriched_data.jsonl`
- Refresh lai `data/reddit_enriched_binary.jsonl`
- Trong file binary, `label` se la `REAL/FAKE`, con nhan cu duoc giu trong `label_6class`

Script duoc goi:
- `src/utils/enrich_master_with_comments.py`
- `src/utils/build_binary_labeled_master.py`

Luu y:
- Enrich co co che resume theo `id`
- Chay lai an toan, script se bo qua bai da enrich

## Buoc 5: Build graph

### 5.1 Build graph text-only

```powershell
python src/utils/research_pipeline.py build-graphs
```

### 5.2 Build ca multimodal graph

```powershell
python src/utils/research_pipeline.py build-graphs --multimodal
```

Lenh nay se:
- Build graph text vao `data/processed_graphs/`
- Neu co `--multimodal`, build them graph multimodal vao `data/processed_graphs_multimodal/`

Script duoc goi:
- `src/utils/build_cascade_graphs.py`
- `src/utils/rebuild_graphs_with_images.py`

Checkpoint:
- Mỗi bai viet co 1 file `.pt`
- Text-only graph dung cho `train_gnn.py`
- Multimodal graph dung cho `train_multimodal_gnn.py`

## Buoc 6: Train va danh gia

### 6.1 Baseline text

```powershell
python src/utils/research_pipeline.py train --model baseline_text --epochs 20 --batch_size 16
```

### 6.2 Baseline image

```powershell
python src/utils/research_pipeline.py train --model baseline_image --epochs 20 --batch_size 16
```

### 6.3 Baseline fusion

```powershell
python src/utils/research_pipeline.py train --model baseline_fusion --epochs 20 --batch_size 16
```

### 6.4 Graph GNN

```powershell
python src/utils/research_pipeline.py train --model gnn --epochs 30 --batch_size 32
```

### 6.5 Multimodal GNN

```powershell
python src/utils/research_pipeline.py train --model multimodal_gnn --epochs 30 --batch_size 32
```

Mac dinh:
- Baseline train tren `labeled_master_binary.jsonl`
- GNN train tren `data/processed_graphs/` voi `data/reddit_enriched_binary.jsonl`
- Multimodal GNN train tren `data/processed_graphs_multimodal/` voi `data/reddit_enriched_binary.jsonl`

Checkpoint model:
- `models/checkpoints/team_runs/`

## Luong lam viec de xuat cho nhom

Member A:
- Crawl Reddit
- Chay `prepare-label`

Member B:
- Labeling tren Label Studio
- Export JSON

Member C:
- Chay `merge-labels`
- Chay `enrich`
- Chay `build-graphs --multimodal`

Member D:
- Chay train baseline/GNN/multimodal
- Tong hop metric

## Checklist moi dot cap nhat du lieu

- Da crawl du lieu moi
- Da tao batch cho Label Studio
- Da export file JSON tu Label Studio
- Da merge vao `labeled_master.jsonl`
- Da refresh `labeled_master_binary.jsonl`
- Da enrich va tao `reddit_enriched_binary.jsonl`
- Da build graph text va multimodal
- Da train lai model can so sanh

## Lenh chuan cho ca chu ky

```powershell
python src/utils/research_pipeline.py crawl --limit 25 --images-only
python src/utils/research_pipeline.py prepare-label
python src/utils/research_pipeline.py merge-labels --input "data/03_clean/Reddit/reddit_run_001/export_reddit_run_001.json"
python src/utils/research_pipeline.py enrich
python src/utils/research_pipeline.py build-graphs --multimodal
python src/utils/research_pipeline.py train --model multimodal_gnn --epochs 30 --batch_size 32
```

Neu team muon thao tac nhanh theo menu tren Windows:

```powershell
run_team_cycle.bat
```

## Xu ly su co thuong gap

- Merge bi trung:
  Dung `src/utils/research_pipeline.py merge-labels`, khong dung append tay vao master.
- Train khong thay du lieu binary:
  Kiem tra da chay xong `merge-labels` va `enrich` hay chua.
- GNN train fail vi chua co graph:
  Chay lai `build-graphs`.
- Multimodal GNN train fail vi chua co graph 1280 chieu:
  Chay `build-graphs --multimodal`.

## File script moi cho team

- `src/utils/research_pipeline.py`
- `src/utils/merge_ls_export_by_id.py`
- `src/utils/enrich_master_with_comments.py`
- `src/utils/build_cascade_graphs.py`
