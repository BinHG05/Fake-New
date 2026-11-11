# Demo Inference Plan - Fake News Detection

## Muc tieu

Xay dung mot luong demo inference on dinh, de trinh bay duoc gia tri cua model fake news detection ngay tren dashboard hien tai.

Demo can ho tro 2 kieu su dung:

1. Nhap URL Reddit post -> he thong crawl metadata va comment cascade -> build graph -> predict REAL/FAKE.
2. Upload file JSONL da co san trong project -> batch predict -> hien thi bang ket qua va thong ke tong hop.

Plan nay uu tien:

- Demo chay duoc that trong repo hien tai.
- Giam rui ro vo demo do API, checkpoint, hoac duong di du lieu.
- Dung mot backend chinh cho demo.
- Bao dam co cach verify ro rang truoc khi demo.

## Pham vi va quyet dinh thiet ke

### Scope chinh

- Su dung Flask app hien tai trong `demo-website/app.py` lam backend demo chinh.
- Them inference endpoints vao Flask app.
- Them tab moi "Live Prediction" vao dashboard hien tai.
- Viet `src/deployment/predictor.py` thanh core inference engine cho demo.

### Scope khong uu tien trong phase demo

- Khong bat buoc dung `src/deployment/api.py` cho FastAPI standalone.
- Khong tach rieng production deployment trong phase nay.
- Khong cam ket ">=90% accuracy" neu chua co ket qua verify thuc te voi checkpoint cuoi cung.

### Quyet dinh quan trong

1. Demo se chi co mot duong vao chinh: `python demo-website/app.py`.
2. FastAPI, neu can, chi la huong mo rong sau demo.
3. Claim ve metric se dua tren checkpoint da verify, khong dua tren ky vong.

## Ket qua mong doi sau khi hoan thanh

Khi plan hoan thanh, he thong demo se hoat dong nhu sau:

1. Ban mo dashboard va thay them tab `Live Prediction`.
2. Ban paste mot Reddit URL vao o nhap va bam `Predict`.
3. Frontend goi API Flask `/api/predict/url`.
4. Backend:
   - Parse URL.
   - Goi Reddit JSON API de lay post metadata.
   - Lay comment cascade bang `RedditCrawler.fetch_comments()`.
   - Build text graph bang `CascadeGraphBuilder`.
   - Neu post co anh, tai anh ve file tam va trich xuat image feature bang `ResNetFeatureExtractor`.
   - Ghep feature thanh tensor dung format model da train.
   - Load checkpoint da chot va predict.
5. Frontend hien thi:
   - Nhan prediction REAL hoac FAKE.
   - Confidence score.
   - Probability cua 2 lop.
   - Thong tin post: title, subreddit, so comment, co anh hay khong.
   - Thong tin graph: so node, so edge, feature dimension.
6. Neu upload file JSONL:
   - He thong chay batch predict.
   - Hien thi bang ket qua tung item.
   - Hien thi summary: total, real_count, fake_count, avg_confidence.
   - Neu file co label ground truth, he thong co the hien thi accuracy/F1 cho batch do.

Noi ngan gon: sau khi xong, ban se co mot dashboard co the demo "predict truc tiep" tren du lieu Reddit that va cung co the demo "predict hang loat" tren mot file co san de doi chieu ket qua.

## Kien truc sau khi chot

```text
User
  |
  v
demo-website/index.html + script.js
  |
  v
Flask app: demo-website/app.py
  |
  v
src/deployment/predictor.py
  |
  +-- Reddit metadata fetch
  +-- RedditCrawler.fetch_comments()
  +-- CascadeGraphBuilder.build_graph()
  +-- optional image download/cache
  +-- ResNetFeatureExtractor.extract_batch()
  +-- MultimodalCascadeGNN checkpoint inference
  |
  v
JSON response -> render on dashboard
```

## File phai sua

| Action | File | Vai tro |
|---|---|---|
| MODIFY | `src/deployment/predictor.py` | Core inference engine |
| MODIFY | `demo-website/app.py` | Flask endpoints cho prediction |
| MODIFY | `demo-website/index.html` | Them tab `Live Prediction` |
| MODIFY | `demo-website/script.js` | Goi API va render ket qua |
| MODIFY | `demo-website/styles.css` | UI cho prediction |

## File khong nam trong MVP demo

| Action | File | Ly do |
|---|---|---|
| OPTIONAL | `src/deployment/api.py` | De sau, neu can FastAPI standalone |

## Du lieu va checkpoint can chot truoc khi code

### Checkpoint demo

Checkpoint uu tien:

- `models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt`

Fallback text/graph-only neu multimodal gap su co:

- `models/checkpoints/best_cascade_gnn_gat.pt`

Luu y:

- Ten fallback phai dung voi file dang ton tai trong repo.
- Khong hardcode metric "90%" vao UI neu chua verify bang script.

### Graph feature contract

Predictor phai tao `Data` object co shape tuong thich voi checkpoint:

- Text-only graph: `x.shape = [N, 768]`
- Multimodal graph: `x.shape = [N, 1280]` voi:
  - `768` dim dau la text embedding
  - `512` dim cuoi la image embedding

Rule cho image feature:

- Root node nhan image feature that neu co anh.
- Cac child node nhan zero-padding `512d`.
- Neu khong tai duoc anh, toan bo image branch dung zero vector.

## Chi tiet implementation

### Component 1: Core inference engine

#### File

`src/deployment/predictor.py`

#### Muc tieu

Day la noi duy nhat chua logic inference.

#### De xuat interface

```python
class FakeNewsPredictor:
    def __init__(self, checkpoint_path=None, device="auto", allow_fallback=True):
        ...

    def predict_url(self, reddit_url: str) -> dict:
        ...

    def predict_batch(self, jsonl_path: str) -> dict:
        ...

    def health(self) -> dict:
        ...
```

#### Trach nhiem

1. Load model mot lan.
2. Khoi tao `CascadeGraphBuilder`.
3. Khoi tao `ResNetFeatureExtractor` neu dang o che do multimodal.
4. Parse va validate Reddit URL.
5. Fetch metadata cua post.
6. Fetch comments bang `RedditCrawler.fetch_comments()`.
7. Build text graph.
8. Download anh ve temp file neu co `media_url`.
9. Extract image features va concat vao graph.
10. Chay forward pass.
11. Tra response JSON giau metadata de frontend render duoc.

#### Output schema cho `predict_url`

```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "probabilities": {
    "REAL": 0.13,
    "FAKE": 0.87
  },
  "post_info": {
    "id": "xyz123",
    "title": "....",
    "subreddit": "worldnews",
    "comment_count": 42,
    "has_image": true,
    "source_url": "https://www.reddit.com/..."
  },
  "graph_info": {
    "num_nodes": 43,
    "num_edges": 42,
    "feature_dim": 1280
  },
  "model_info": {
    "checkpoint": "best_multimodal_cascade_binary_cross_attention.pt",
    "model_type": "multimodal",
    "device": "cuda"
  },
  "warnings": []
}
```

#### Output schema cho `predict_batch`

```json
{
  "total": 50,
  "predictions": [
    {
      "id": "abc123",
      "prediction": "REAL",
      "confidence": 0.92,
      "ground_truth": "REAL",
      "correct": true
    }
  ],
  "summary": {
    "real_count": 28,
    "fake_count": 22,
    "avg_confidence": 0.85,
    "evaluated_count": 50,
    "accuracy": 0.90,
    "f1_binary": 0.89
  }
}
```

#### Luu y ky thuat bat buoc

- Phai co ham download anh tu remote URL sang temp path local, vi `ResNetFeatureExtractor` dang doc file tren dia.
- Phai cleanup temp files sau khi predict.
- Khong reload checkpoint moi request.
- Phai co che do degrade gracefully:
  - Loi crawl comment -> van co the predict graph chi voi root node.
  - Loi tai anh -> van predict voi image zeros.
  - Loi multimodal checkpoint -> neu bat `allow_fallback`, fallback sang graph-only checkpoint.

### Component 2: Flask backend integration

#### File

`demo-website/app.py`

#### Endpoints can them

```text
POST /api/predict/url
POST /api/predict/batch
GET  /api/predict/health
```

#### Nguyen tac

- Lazy init predictor mot lan.
- Khong tao FastAPI rieng trong phase nay.
- Validate input ro rang.
- Response loi phai co message de frontend hien thi duoc.

#### Flow endpoint `/api/predict/url`

1. Nhan JSON body `{ "url": "..." }`
2. Validate rong/sai format -> `400`
3. Goi `get_predictor().predict_url(url)`
4. Bat exception -> `500` voi error message da duoc sanitize
5. Return JSON result

#### Flow endpoint `/api/predict/batch`

1. Nhan file upload JSONL
2. Save tam vao `temp`
3. Goi `predict_batch(temp_path)`
4. Xoa file tam
5. Return JSON ket qua

#### Flow endpoint `/api/predict/health`

Tra thong tin:

- model_loaded
- checkpoint
- model_type
- device
- ready

### Component 3: Frontend demo tab

#### Files

- `demo-website/index.html`
- `demo-website/script.js`
- `demo-website/styles.css`

#### Muc tieu UX

Tab moi phai de demo vien thao tac nhanh, it can giai thich.

#### UI can co

1. O nhap Reddit URL.
2. Nut `Predict`.
3. Upload JSONL.
4. Nut `Batch Predict`.
5. Vung loading state.
6. Vung hien thi loi.
7. Card ket qua prediction.
8. Bang ket qua batch.
9. Khu summary.

#### Yeu cau UX

- Disable nut khi request dang chay.
- Hien thi loading message ro rang.
- Hien thi warning neu he thong fallback hoac khong lay duoc anh/comment.
- Mau sac nhat quan:
  - REAL = xanh
  - FAKE = do

## Luong xu ly chi tiet cho `predict_url`

```text
Input Reddit URL
  ->
parse permalink va post id
  ->
fetch post metadata tu Reddit JSON API
  ->
fetch comment cascade bang RedditCrawler.fetch_comments()
  ->
tao item trung gian:
  {
    id,
    raw_text,
    media_url,
    user_id,
    comment_count,
    cascade,
    metadata
  }
  ->
CascadeGraphBuilder.build_graph(item)
  ->
neu co media_url:
  download anh -> local temp file
  ->
  ResNetFeatureExtractor.extract_batch([temp_file])
  ->
  gan image feature vao root node
else:
  image zeros
  ->
tao batch tensor
  ->
model.eval() + softmax
  ->
response JSON
```

## Xu ly loi va fallback

Day la phan rat quan trong de demo khong vo.

### Truong hop 1: URL sai

- Return `400`
- Frontend hien thi thong bao "URL Reddit khong hop le"

### Truong hop 2: Reddit rate limit hoac khong fetch duoc comment

- Khong fail ngay neu van lay duoc post metadata
- Build graph toi thieu voi root node
- Tra warning trong response

### Truong hop 3: Khong tai duoc anh

- Van predict
- Dung image zero vector
- Tra warning `image_unavailable`

### Truong hop 4: Checkpoint multimodal loi

- Neu `allow_fallback=True`, fallback sang graph-only
- Health endpoint phai noi ro dang dung model nao

### Truong hop 5: JSONL batch co dong loi

- Bo qua dong loi
- Tang `skipped_count`
- Tra warning va thong ke so dong bi bo qua

## Acceptance criteria

Plan duoc xem la hoan thanh khi dat du cac dieu kien sau.

### Muc A - Backend

- `python demo-website/app.py` khoi dong thanh cong.
- `GET /api/predict/health` tra `ready=true`.
- `POST /api/predict/url` predict duoc it nhat 1 Reddit URL hop le.
- `POST /api/predict/batch` predict duoc 1 file JSONL hop le.

### Muc B - Frontend

- Tab `Live Prediction` xuat hien trong dashboard.
- User predict duoc URL ma khong can reload trang.
- User upload duoc JSONL va thay bang ket qua.
- Co loading state va error state.

### Muc C - Reliability

- He thong khong crash neu post khong co anh.
- He thong khong crash neu comment crawl fail.
- Temp files duoc cleanup.
- Checkpoint chi load mot lan sau khi app start.

### Muc D - Verification

- Co script/test xac nhan output schema hop le.
- Co batch evaluation tren mot file labeled co san.
- Metric duoc ghi ro la metric thuc te cua checkpoint dang dung.

## Verification plan

### 1. Smoke test predictor

```powershell
python -c "
from src.deployment.predictor import FakeNewsPredictor
p = FakeNewsPredictor('models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt')
print(p.health())
"
```

Pass condition:

- Khong crash khi init
- Health tra dung checkpoint, device, model_type

### 2. Smoke test URL prediction

```powershell
python -c "
from src.deployment.predictor import FakeNewsPredictor
p = FakeNewsPredictor('models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt')
r = p.predict_url('https://www.reddit.com/r/worldnews/comments/POST_ID/sample_title/')
print(r['prediction'], r['confidence'])
assert r['prediction'] in ['REAL', 'FAKE']
assert 0 <= r['confidence'] <= 1
"
```

Pass condition:

- Tra ve du prediction schema
- Khong bat buoc phai co anh

### 3. Batch test tren file labeled

```powershell
python -c "
from src.deployment.predictor import FakeNewsPredictor
p = FakeNewsPredictor('models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt')
r = p.predict_batch('data/03_clean/Fakeddit/labeled_master_binary.jsonl')
print(r['summary'])
"
```

Pass condition:

- Co summary
- Neu file co ground truth thi co accuracy/F1

### 4. Flask API smoke test

```powershell
curl -X GET http://localhost:5000/api/predict/health
```

```powershell
curl -X POST http://localhost:5000/api/predict/url ^
  -H "Content-Type: application/json" ^
  -d "{\"url\": \"https://www.reddit.com/r/worldnews/comments/POST_ID/sample_title/\"}"
```

### 5. Manual demo test

1. Mo dashboard.
2. Vao tab `Live Prediction`.
3. Paste 1 Reddit URL hop le.
4. Xem prediction va metadata.
5. Upload 1 file JSONL nho.
6. Xem batch table va summary.

## Cach trinh bay khi demo

Neu moi thu hoan thanh, ban co the demo theo kich ban sau:

1. Mo dashboard va gioi thieu day la giao dien tong hop cua project.
2. Chuyen sang tab `Live Prediction`.
3. Paste 1 Reddit post URL that.
4. Bam `Predict` va noi:
   - He thong dang crawl metadata va comments
   - Dang build propagation graph
   - Dang ket hop text va image neu co
   - Dang predict bang checkpoint da train
5. Cho ket qua hien ra va chi vao:
   - Prediction
   - Confidence
   - Graph size
   - Co anh hay khong
6. Sau do upload file JSONL da co label de cho thay he thong xu ly hang loat va doi chieu voi ground truth.

## Cau hoi can chot truoc khi implement

1. Checkpoint nao duoc chot de demo?
2. File JSONL nao se la file batch demo chinh?
3. Co muon hien thi metric batch trong UI neu file co ground truth khong?

## Ke hoach thuc hien de xuat

### Phase 1

- Viet `FakeNewsPredictor`
- Chay duoc health + local predictor smoke test

### Phase 2

- Tich hop vao Flask app
- Tao 3 endpoints prediction

### Phase 3

- Them tab frontend
- Noi API vao UI

### Phase 4

- Chay verification
- Chot checkpoint
- Chot du lieu demo
- Chuan bi kich ban demo

## Tieu chi thanh cong cuoi cung

Plan nay thanh cong neu sau cung ban co the noi mot cach tu tin:

"He thong nay khong chi train model. Chung toi co the dua vao mot Reddit post that, build graph tu metadata va comment cascade, ket hop text va image, va tra ra du doan truc tiep ngay tren dashboard."
