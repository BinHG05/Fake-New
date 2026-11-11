# Kế hoạch & Hướng dẫn Cập nhật Công nghệ Xử lý (Feature Versioning)

Tài liệu này giải thích tường tận quy trình chuẩn (Dành cho Data Engineer / Team AI) để thử nghiệm các công nghệ xử lý Text/Hình ảnh mới **MÀ KHÔNG LÀM HỎNG DATA CŨ**, ứng dụng tư duy Data Lake và Feature Versioning.

> [!IMPORTANT]
> **Nguyên tắc "Bất Di Bất Dịch" của hệ thống:**
> Không bao giờ được dùng lệnh `rm` xóa thư mục đồ thị đang chạy ổn định.
> Luôn giữ nguyên file **`labeled_master_binary.jsonl`** làm Source of Truth (Chiếc rương kho báu). Mọi sự thay đổi công nghệ chỉ diễn ra ở khâu **BĂM DATA RA VECTOR (Feature Extraction)**.

---

## 1. Bản chất của việc "Thay đổi Công nghệ"

Trong Machine Learning, khi ta nói "Tôi muốn dùng mô hình xử lý Text mới (VD: PhoBERT thay cho XLM-Roberta)" hoặc "Dùng mô hình Ảnh mới (VD: ViT thay cho ResNet-50)", thực chất là ta đang thay đổi chiếc máy xay xát (Feature Extractor).

- **Thịt nguyên miếng (Raw Data):** Chữ tiếng Anh, chửi thề, link URL, file `.jpg`. Nằm yên trong `labeled_master_binary.jsonl`.
- **Thịt băm (Feature Vectors):** Ma trận số (`.pt` files). Nằm trong thư mục `data/processed_graphs_multimodal/`. Model GNN chỉ ăn được loại thịt băm này.

Khi đổi máy xay xát, **miếng thịt băm ra sẽ có hình dáng khác** (Ví dụ ResNet nặn ra vector 2048 chiều, ViT nặn ra vector 768 chiều). Do đó, bạn không thể trộn thịt băm mới và thịt băm cũ vào chung một hộp được! Đó là lý do ta phải tạo một "Hộp mới" (Data Lake Folder mới) và xay lại toàn bộ thịt nguyên miếng đang có.

---

## 2. Quy trình Thực thi (Dành cho Team Member)

Để thử nghiệm một model tiền xử lý mới, các thành viên cần thực hiện đúng 3 bước sau:

### Bước 1: Sửa code thuật toán (Máy xay)

Chỉ có 2 file bạn sẽ đụng vào khi muốn thay đổi công nghệ băm vector:
- **Xử lý Ảnh:** `src/data/fakeddit_preprocessor_image.py` (Đổi model ResNet sang model khác).
- **Xử lý Đồ thị/Text:** `src/utils/build_cascade_graphs.py` và `src/utils/rebuild_graphs_with_images.py` (Nơi gọi XLM-R để nhúng chữ).

> [!NOTE]
> Bạn cứ sửa code thoải mái trong file Python, tạo branch mới trên Git là được.

### Bước 2: Build lại Đồ thị ra một Data Lake mới (Tạo hộp thịt mới)

Thay vì xóa thư mục cũ, bạn chạy các script sinh đồ thị nhưng **Truyền tham số `--output` sang một thư mục CÓ HẬU TỐ PHIÊN BẢN**.

Ví dụ, muốn nhúng bằng thuật toán V2:

**Lệnh 1: Xây dựng Đồ thị Cascade Text**
```bash
python src/utils/build_cascade_graphs.py \
  --input data/reddit_enriched_data.jsonl \
  --output data/processed_graphs_v2
```

**Lệnh 2: Gắn thêm ảnh vào Đồ thị (Multimodal)**
```bash
python src/utils/rebuild_graphs_with_images.py \
  --metadata data/reddit_enriched_data.jsonl \
  --graph_dir data/processed_graphs_v2 \
  --output_dir data/processed_graphs_multimodal_v2
```

Lúc này, trong ổ cứng của bạn sẽ tồn tại song song 2 thư mục:
- `data/processed_graphs_multimodal/` (Data cũ, đang dùng tốt)
- `data/processed_graphs_multimodal_v2/` (Data mới, dùng công nghệ mới)

### Bước 3: Đưa vào Train và So sánh

Đây là lúc ta thu quả ngọt. Bạn không cần đổi code train. Khi chạy training, bạn chỉ cần trỏ tham số `--graph_dir` vào Data Lake mới!

```bash
python src/training/train_multimodal_gnn.py \
  --graph_dir data/processed_graphs_multimodal_v2 \
  --metadata data/reddit_enriched_binary.jsonl \
  --epochs 30 \
  --save_dir models/checkpoints/run_v2
```

---

## 3. Kết quả mong đợi

1. **Không gián đoạn (Zero Downtime):** Model hiện tại vẫn hoạt động bình thường, web dashboard vẫn hiển thị kết quả cũ nếu dùng tham số cũ.
2. **Khả năng A/B Testing trực tiếp:** Khi model mới train xong, bạn có thể nhìn vào Bảng kết quả (Results) để so sánh chỉ số F1 và Accuracy của V1 và V2 một cách cực kỳ trong suốt.
3. **Rollback 1 giây:** Nếu V2 quá tệ, bạn chỉ cần bỏ thùng rác thư mục `v2` và quay lại dùng câu lệnh `--graph_dir data/processed_graphs_multimodal` ban đầu mà không mất đi bất kỳ lượng công sức cào/gán nhãn nào!

> [!TIP]
> Lời khuyên cho Data Engineer: Ổ cứng máy tính giá cực kỳ rẻ. Thời gian để ngồi gán nhãn và cào lại data là vô giá. Hãy tận dụng tối đa việc lưu ra nhiều thư mục Version (v1, v2, v_test_resnet...) cho đến khi bạn ưng ý nhất!
