# Hướng dẫn Gán nhãn Dữ liệu với Label Studio

> **Dành cho:** Thành viên nhóm NCKH - Multimodal Fake News Detection  
> **Cập nhật:** 2026-02-06

---

## Yêu cầu

- Docker Desktop đã cài đặt và đang chạy
- Python 3.8+
- Clone project về máy

---

## Bước 1: Khởi động Label Studio

Mở Terminal/PowerShell tại thư mục project và chạy:

```powershell
docker-compose up -d
```

Đợi 10-15 giây, sau đó truy cập: **http://localhost:8080**

> **Lần đầu:** Tạo tài khoản mới (email + password bất kỳ).

---

## Bước 2: Tạo Project mới

1. Click **"Create Project"**
2. Đặt tên: `Fake-News-BatchXXX`
3. Trong tab **"Labeling Setup"**, chọn **"Custom template"** và dán:

```xml
<View>

  <!-- 1. Văn bản -->
  <Header value="1. Văn bản Bài viết (clean_text)"/>
  <Text
    name="post_text"
    value="$clean_text"
  />

  <!-- 2. Ảnh / Media (tham khảo) -->
  <Header value="2. Ảnh/Media (nếu có)"/>
  <Image
    name="post_image"
    value="$image_info.processed_path"
    zoom="true"
    zoomControl="true"
    max-width="100%"
    max-height="600px"
  />

  <!-- 3. Nhãn chi tiết -->
  <View style="border:1px solid #ccc; padding:10px; margin-top:15px;">
    <Header value="3. Nhãn Chi tiết (Ground Truth – Chọn 1)"/>

    <Choices
      name="label_fine"
      toName="post_text"
      choice="single"
      required="true">

      <Choice value="TRUE"/>
      <Choice value="MOSTLY_TRUE"/>
      <Choice value="HALF_TRUE"/>
      <Choice value="BARELY_TRUE"/>
      <Choice value="FALSE"/>
      <Choice value="PANTS_ON_FIRE"/>

    </Choices>
  </View>

</View>
```

4. Click **"Save"**

---

## Bước 3: Cấu hình Cloud Storage (BẮT BUỘC)

**Bước này rất quan trọng để ảnh hiển thị!**

1. Vào **Settings** → **Cloud Storage**
2. Click **"Add Source Storage"**
3. Chọn **"Local files"**
4. Nhập đường dẫn:
   ```
   /label-studio/project/data/02_processed/images
   ```
5. Click **"Add Storage"**
6. Click **"Sync Storage"**

---

## Bước 4: Import dữ liệu

1. Vào tab **"Import"** của project
2. Chọn file JSON batch của bạn:
   - `data/03_clean/Fakeddit/batch_200_400/Fakeddit/train_for_ls.json`
   - (hoặc val_for_ls.json, test_for_ls.json)
3. Click **"Import"**

---

## Bước 5: Gán nhãn

1. Click vào từng task
2. Đọc văn bản, xem ảnh
3. Chọn nhãn: **Real / Fake / Satire / Misleading**
4. Click **"Submit"**

---

## Bước 6: Export kết quả

1. Vào **"Export"**
2. Chọn **"JSON"**
3. Download và đổi tên thành `export_batch_XXX.json`

---

## Bước 7: Merge (Leader làm)

```powershell
python src/utils/convert_ls_export_to_jsonl.py "path/to/export.json" "data/03_clean/Fakeddit/labeled_master.jsonl" --append
```

---

## Xử lý sự cố

### Ảnh không hiển thị
- Kiểm tra đã làm Bước 3 chưa
- Tạo Project mới và làm lại

### Docker không chạy
```powershell
docker-compose down
docker-compose up -d
```

### Reset hoàn toàn
```powershell
docker-compose down
del labels_storage\label_studio.sqlite3
docker-compose up -d
```
