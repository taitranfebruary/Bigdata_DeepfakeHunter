# 📸 Hướng dẫn Chụp Screenshots cho Báo cáo

## Sau khi chạy xong Pipeline, làm theo các bước sau:

---

## 1. HDFS NameNode (http://localhost:9870)

### Bước 1: Mở browser và truy cập `http://localhost:9870`

### Bước 2: Screenshot trang Overview
- Cho thấy cluster healthy
- Hiển thị storage utilization

### Bước 3: Chụp Utilities > Browse the file system
- Navigate đến `/raw/cifake/` - chụp cấu trúc thư mục
- Navigate đến `/processed/` - chụp parquet files
- Navigate đến `/results/` - chụp kết quả
- Navigate đến `/spark-logs/` - chụp event logs

---

## 2. Spark Master (http://localhost:8080)

### Bước 1: Mở browser và truy cập `http://localhost:8080`

### Bước 2: Screenshot main page
- Cho thấy Workers connected (ít nhất 1 worker)
- Hiển thị cluster resources (Memory, Cores)

### Bước 3: Click vào một Completed Application
- Chụp chi tiết application

---

## 3. ⭐ Spark History Server (http://localhost:18080) - QUAN TRỌNG NHẤT

### Bước 1: Mở browser và truy cập `http://localhost:18080`

### Bước 2: Screenshot danh sách Applications
- Cho thấy các jobs đã chạy:
  - DeepfakeHunter-FeatureExtraction
  - DeepfakeHunter-Classification
  - DeepfakeHunter-BusinessInsight

### Bước 3: Click vào job "DeepfakeHunter-FeatureExtraction"
- Chụp **Jobs** tab: cho thấy stages
- Chụp **Stages** tab: cho thấy tasks
- Chụp **Event Timeline**: cho thấy parallel execution
- Chụp **Executors** tab: cho thấy task distribution

### Bước 4: Click vào job "DeepfakeHunter-Classification"
- Chụp tương tự như trên
- Đây là bước TRAINING quan trọng

### ⚠️ LƯU Ý: Screenshots từ History Server là bằng chứng quan trọng nhất!

---

## 4. Terminal Output

### Chạy lệnh và chụp output:

```bash
# Xem metrics
docker exec -it spark-master spark-submit \
    --master spark://spark-master:7077 \
    /scripts/06_view_results.py
```

### Chụp các phần:
- Model Performance Metrics (Accuracy, Precision, Recall)
- Confusion Matrix
- Business Question Answer
- Dataset Statistics

---

## 5. HTML Report

### Bước 1: Copy file ra máy local
```bash
docker cp spark-master:/scripts/report.html ./report.html
```

### Bước 2: Mở trong browser
```bash
open report.html  # macOS
```

### Bước 3: Chụp các phần của report:
- Executive Summary
- Model Performance Comparison (charts)
- Confusion Matrices
- Business Question Answer

---

## 6. Validation Check

### Chạy script validation:
```bash
docker exec -it spark-master spark-submit \
    --master spark://spark-master:7077 \
    /scripts/09_validate_compliance.py
```

### Chụp kết quả:
- Compliance Summary (5/5 Requirements)

---

## 📋 Checklist Screenshots

```
[ ] HDFS NameNode - Overview
[ ] HDFS NameNode - /raw/cifake/ directory
[ ] HDFS NameNode - /processed/ directory  
[ ] HDFS NameNode - /results/ directory
[ ] HDFS NameNode - /spark-logs/ directory
[ ] Spark Master - Main page with workers
[ ] Spark History - Application list
[ ] Spark History - FeatureExtraction job (Jobs, Stages, Timeline)
[ ] Spark History - Classification job (Jobs, Stages, Timeline)
[ ] Terminal - Model metrics output
[ ] Terminal - Confusion matrix
[ ] Terminal - Business insight
[ ] Terminal - Validation compliance (5/5)
[ ] HTML Report - Full page
[ ] HTML Report - Charts section
```

---

## 🎯 Mẹo

1. **Zoom out browser** để capture nhiều thông tin hơn trong 1 screenshot
2. **Highlight** các con số quan trọng (accuracy, etc.)
3. **Annotate** screenshots để giải thích cho thầy/cô
4. **Sắp xếp** screenshots theo thứ tự pipeline steps

---

## 📝 Nội dung cần có trong Báo cáo

1. **Giới thiệu**
   - Mô tả bài toán Deepfake Detection
   - Dataset CIFAKE

2. **Kiến trúc Pipeline**
   - Sơ đồ pipeline (có sẵn trong README)
   - Mô tả từng bước

3. **Triển khai**
   - Docker compose setup
   - HDFS + Spark configuration
   - Screenshots cấu hình

4. **Feature Extraction**
   - MobileNetV2 architecture
   - UDF implementation
   - Screenshots Spark History

5. **Training & Evaluation**
   - LogisticRegression vs RandomForest
   - Metrics comparison
   - Confusion matrix analysis

6. **Business Insight**
   - Trả lời câu hỏi đề bài
   - Kết luận về model

7. **Tuân thủ yêu cầu**
   - Bảng checklist 5 yêu cầu
   - Screenshots bằng chứng

8. **Kết luận**
   - Tổng kết
   - Hạn chế và cải thiện
