# BÁO CÁO ĐỒ ÁN: DEEPFAKE HUNTER - BIG DATA PIPELINE

## THÔNG TIN CHUNG

**Đề tài:** The Deepfake Hunter - Xây dựng Pipeline Big Data Phân tán  
**Môn học:** Thực hành Big Data  
**Thời gian thực hiện:** 2 tuần  

---

## 1. GIỚI THIỆU

### 1.1. Bối cảnh

Deepfake (ảnh do AI tạo ra) đang trở thành mối đe dọa nghiêm trọng với khả năng tạo nội dung giả mạo tinh vi. Việc phát hiện deepfake đòi hỏi:
- Xử lý khối lượng lớn dữ liệu ảnh
- Trích xuất features phức tạp
- Tính toán phân tán để đảm bảo hiệu năng

### 1.2. Mục tiêu

Xây dựng một pipeline Big Data end-to-end để:
1. ✅ Lưu trữ phân tán 120,000 ảnh trên HDFS
2. ✅ Trích xuất features bằng MobileNetV2 (pretrained ImageNet)
3. ✅ Huấn luyện models phân tán với Spark MLlib
4. ✅ Đạt accuracy >85% trong phát hiện deepfake
5. ✅ Tuân thủ 100% yêu cầu kỹ thuật đồ án

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1. Tổng quan

```
┌─────────────────┐
│  Data Source    │  CIFAKE Dataset (120K images)
│  (Local)        │
└────────┬────────┘
         │ Upload
         ▼
┌─────────────────┐
│  HDFS Storage   │  /raw/cifake/
│  (Distributed)  │  - train/REAL, train/FAKE
└────────┬────────┘  - test/REAL, test/FAKE
         │ Read via Spark binaryFile
         ▼
┌─────────────────┐
│ Feature Extract │  MobileNetV2 (ImageNet)
│ (Spark UDF)     │  → 1280-dim vectors
└────────┬────────┘
         │ Save Parquet
         ▼
┌─────────────────┐
│  /processed/    │  train_features.parquet
│  (HDFS)         │  test_features.parquet
└────────┬────────┘
         │ Load for training
         ▼
┌─────────────────┐
│  Spark MLlib    │  LogisticRegression
│  (Distributed)  │  RandomForest
└────────┬────────┘
         │ Save results
         ▼
┌─────────────────┐
│   /results/     │  models/, metrics.parquet
│   (HDFS)        │  predictions.parquet
└─────────────────┘
```

### 2.2. Components

**Docker Cluster:**
- **spark-master**: Cluster coordinator (1 node)
- **spark-worker**: Task executor (1 node, 4 cores, 10GB RAM)
- **namenode**: HDFS metadata manager
- **datanode**: HDFS block storage  
- **spark-history**: Job monitoring (port 18080)

**Tech Stack:**
- Apache Spark 3.3.0
- Hadoop HDFS 3.2.1
- PyTorch 2.0.1 + MobileNetV2
- Python 3.9

---

## 3. TUÂN THỦ YÊU CẦU KỸ THUẬT

### 3.1. Yêu cầu 1: Bắt buộc dùng HDFS

✅ **ĐẠT YÊU CẦU**

**Implementation:**
```python
# 02_upload_to_hdfs.py
hdfs_base = "hdfs://namenode:8020/raw/cifake"
subprocess.run(f'hdfs dfs -put "{local_path}" "{hdfs_folder}"', shell=True)
```

**Bằng chứng:**
```bash
$ docker exec spark-master hdfs dfs -count /raw/cifake
    7  120000  110717853 /raw/cifake
```

120,000 files được lưu trên HDFS, không đọc trực tiếp từ local filesystem.

---

### 3.2. Yêu cầu 2: Cấm dùng vòng lặp Local

✅ **ĐẠT YÊU CẦU**

**Implementation:**
```python
# 03_feature_extraction.py - KHÔNG dùng os.listdir
df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load(hdfs_path)
```

Sử dụng Spark DataFrame để đọc files, không có vòng lặp Python `for` hoặc `os.listdir`.

---

### 3.3. Yêu cầu 3: AI Phân tán (Distributed Inference)

✅ **ĐẠT YÊU CẦU**

**Implementation:**
```python
@udf(returnType=ArrayType(FloatType()))
def extract_mobilenet_features(image_bytes):
    # MobileNetV2 inference runs inside Spark Worker
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    features = model(img_tensor)
    return features.tolist()

df_features = df.select(
    col("path"),
    extract_mobilenet_features(col("content")).alias("features")
)
```

MobileNetV2 model chạy **bên trong Spark Workers** qua UDF (User Defined Function), không phải trên driver.

**Bằng chứng phân tán:**
- Model được cache trên mỗi worker
- Tasks chạy song song trên 4 cores
- Spark History Server logs chứng minh parallel execution

---

### 3.4. Yêu cầu 4: Lưu trữ Parquet

✅ **ĐẠT YÊU CẦU**

**Implementation:**
```python
train_output = "hdfs://namenode:8020/processed/train_features.parquet"
train_df.write.mode("overwrite").parquet(train_output)
```

**Bằng chứng:**
```bash
$ docker exec spark-master hdfs dfs -ls /processed/
/processed/train_features.parquet
/processed/test_features.parquet

$ docker exec spark-master hdfs dfs -ls /results/
/results/lr_predictions.parquet
/results/rf_predictions.parquet
/results/metrics.parquet
/results/confusion_matrix.parquet
```

Tất cả kết quả trung gian và cuối cùng đều lưu dưới định dạng Parquet.

---

### 3.5. Yêu cầu 5: Spark History Server

✅ **ĐẠT YÊU CẦU**

**Cấu hình:**
```yaml
# docker-compose.yml
spark-history:
  environment:
    - SPARK_NO_DAEMONIZE=true
  ports:
    - "18080:18080"
```

```python
# All scripts
.config("spark.eventLog.enabled", "true")
.config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs")
.config("spark.history.fs.logDirectory", "hdfs://namenode:8020/spark-logs")
```

**Bằng chứng:**
```bash
$ docker exec spark-master hdfs dfs -ls /spark-logs/
/spark-logs/app-20251217104339-0004  # Feature extraction
/spark-logs/app-20251217133829-0005  # Classification
```

Spark History Server hoạt động tại http://localhost:18080 với logs lưu trên HDFS.

**Screenshots:** Xem phần Phụ lục

---

## 4. PIPELINE IMPLEMENTATION

### 4.1. Step 1: Data Ingestion

**Script:** `02_upload_to_hdfs.py`

```python
# Batch upload - KHÔNG dùng vòng lặp file-by-file
folders = [
    ("train", f"{hdfs_base}/train"),
    ("test", f"{hdfs_base}/test"),
]

for local_folder, hdfs_folder in folders:
    cmd = f'hdfs dfs -put "{local_path}" "{hdfs_folder}"'
    subprocess.run(cmd, shell=True)
```

**Kết quả:**
- 100,000 training images (50K REAL, 50K FAKE)
- 20,000 test images (10K REAL, 10K FAKE)
- Lưu trên HDFS: `/raw/cifake/train/`, `/raw/cifake/test/`

---

### 4.2. Step 2: Feature Extraction

**Script:** `03_feature_extraction.py`

**Model:** MobileNetV2 (pretrained on ImageNet)
- Input: 224x224 RGB images
- Output: 1280-dimensional feature vectors
- Architecture: Inverted residuals with linear bottlenecks

**Distributed Processing:**
```python
# Đọc ảnh từ HDFS qua Spark
df = spark.read.format("binaryFile").load(hdfs_path)

# Partition để phân tán workload
df = df.coalesce(4)  # 4 partitions cho 4 cores

# Extract features qua UDF (chạy trên workers)
df_features = df.select(
    extract_mobilenet_features(col("content")).alias("features"),
    lit(label_value).alias("label")
)

# Lưu Parquet lên HDFS
df_features.write.parquet(output_path)
```

**Tối ưu:**
- Model cached per worker (tránh reload liên tục)
- Batch processing 4 partitions song song
- Lưu từng folder riêng để tránh timeout

**Kết quả:**
- `train_features.parquet`: 100,000 samples × 1280 dims
- `test_features.parquet`: 20,000 samples × 1280 dims

---

### 4.3. Step 3: Distributed Classification

**Script:** `04_train_classifier.py`

**Models trained:**

1. **Logistic Regression**
   - Algorithm: L-BFGS optimizer
   - Regularization: L2
   - Max iterations: 100

2. **Random Forest**
   - Trees: 100
   - Max depth: 10
   - Feature subset: sqrt(features)

**Pipeline:**
```python
# Scaling features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Train model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label")

# Distributed training trên Spark cluster
lr_model = lr.fit(train_df)
rf_model = rf.fit(train_df)

# Evaluate
predictions = model.transform(test_df)
accuracy = evaluator.evaluate(predictions)
```

**Distributed execution:**
- Training data được partition trên cluster
- Gradient computations chạy song song
- Model aggregation trên driver

---

### 4.4. Step 4: Business Insights

**Script:** `05_business_insight.py`

**Analysis performed:**
- Confusion matrix analysis
- Error rate by class (Real vs Fake)
- Model comparison metrics
- Feature importance (for RF)

**Key insights:**
1. MobileNetV2 features capture discriminative patterns
2. Logistic Regression performs better than Random Forest
3. False Positive rate: 10.6%
4. False Negative rate: 11.4%

---

## 5. KẾT QUẢ THỰC NGHIỆM

### 5.1. Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| **Accuracy** | **88.99%** | 87.02% |
| **Precision** | 88.99% | 87.03% |
| **Recall** | 88.99% | 87.02% |
| **F1-Score** | 88.99% | 87.02% |
| **AUC-ROC** | **95.85%** | 94.45% |
| **Train Time** | 21.10s | 97.33s |

### 5.2. Confusion Matrix (Logistic Regression)

|  | Predicted REAL | Predicted FAKE |
|---|----------------|----------------|
| **Actual REAL** | 8,940 (TN) | 1,060 (FP) |
| **Actual FAKE** | 1,142 (FN) | 8,858 (TP) |

**Phân tích:**
- **True Negative Rate**: 89.4% (correctly identified real images)
- **True Positive Rate**: 88.6% (correctly identified fake images)
- **False Positive Rate**: 10.6% (real images wrongly flagged as fake)
- **False Negative Rate**: 11.4% (fake images missed)

### 5.3. So sánh với Baseline

| Approach | Accuracy | Features | Training Time |
|----------|----------|----------|---------------|
| Histogram + Stats | 75.88% | 780-dim | 15s |
| **MobileNetV2 (Ours)** | **88.99%** | 1280-dim | 21s |

MobileNetV2 features cải thiện **+13% accuracy** so với histogram features đơn giản.

---

## 6. BUSINESS QUESTION ANSWER

### Câu hỏi

**"Liệu model được chọn có trích xuất đủ thông tin để phát hiện Deepfake không?"**

### Trả lời

✅ **CÓ - MobileNetV2 features ĐỦ KHẢ NĂNG phát hiện Deepfake**

**Bằng chứng:**

1. **High Accuracy (88.99%)**
   - Vượt ngưỡng 85% yêu cầu cho production systems
   - Balanced performance trên cả Real và Fake classes

2. **Excellent AUC-ROC (95.85%)**
   - Model có khả năng phân biệt tốt giữa 2 classes
   - Thresholds có thể điều chỉnh linh hoạt

3. **Feature Quality**
   - MobileNetV2 (pretrained ImageNet) học được:
     - Low-level: Textures, edges, colors
     - Mid-level: Shapes, patterns
     - High-level: Semantic concepts
   - 1280 dimensions đủ để capture AI artifacts:
     - Smooth textures không tự nhiên
     - Inconsistent lighting
     - Subtle pattern repetitions

4. **Production Ready**
   - Training time: 21s (rất nhanh)
   - Inference: Có thể batch process hàng nghìn ảnh/ngày
   - Scalability: Pipeline phân tán, dễ mở rộng

### Recommendation

**Deploy Logistic Regression model** với confidence threshold = 0.5:
- Balance giữa False Positives và False Negatives
- Fast inference
- Interpretable results
- Monitor performance và retrain định kỳ

---

## 7. CHALLENGES & SOLUTIONS

### 7.1. Upload bottleneck

**Problem:** Upload 120K files lên HDFS mất 20+ giờ

**Root cause:**
- Docker I/O bottleneck trên macOS
- Mỗi file nhỏ cần network round-trip riêng

**Solution:**
- Batch upload với `hdfs dfs -put folder/`
- Upload song song train và test folders
- Kết quả: Giảm từ 40h → 20h

### 7.2. OOM during feature extraction

**Problem:** Executor OOM với 4GB memory

**Initial approach:** Process tất cả files cùng lúc

**Solution:**
- Giảm partitions từ 100 → 4
- Cache model per worker (không reload)
- Process từng folder riêng, lưu checkpoint
- Tăng executor memory → 8GB

### 7.3. Worker không nhận job

**Problem:** Task không được schedule

**Root cause:** Resource request > available

**Solution:**
- Kiểm tra worker resources qua Master UI
- Điều chỉnh `--executor-memory` phù hợp
- Rebuild worker với cấu hình cao hơn

---

## 8. KẾT LUẬN

### 8.1. Đánh giá chung

Đồ án đã **hoàn thành 100% mục tiêu**:

✅ Xây dựng thành công pipeline Big Data phân tán  
✅ Tuân thủ tuyệt đối 5 yêu cầu kỹ thuật  
✅ Đạt accuracy 88.99%, vượt mục tiêu 85%  
✅ Chứng minh MobileNetV2 features đủ khả năng phát hiện Deepfake  
✅ Pipeline sẵn sàng cho production deployment  

### 8.2. Kiến thức đạt được

- Thiết kế và triển khai Hadoop/Spark cluster với Docker
- HDFS: Distributed storage, replication, fault tolerance
- Spark: RDD, DataFrame, distributed computing
- Spark MLlib: Distributed machine learning
- PyTorch integration với Spark UDF
- Performance tuning và troubleshooting

### 8.3. Hướng phát triển

**Short-term:**
- Deploy web API cho real-time inference
- Add monitoring và alerting
- Implement A/B testing framework

**Long-term:**
- Try ResNet50 (2048-dim features)
- Ensemble multiple feature extractors
- Fine-tune MobileNetV2 trên CIFAKE dataset
- Scale to multi-node cluster

---

## PHỤ LỤC

### A. System Requirements

- **Hardware:**
  - CPU: 4+ cores
  - RAM: 16GB recommended
  - Disk: 20GB free space

- **Software:**
  - Docker Desktop
  - macOS/Linux/Windows

### B. Screenshots

**B.1. Spark History Server - Application List**
![Application List](images/history-server-apps.png)

**B.2. Event Timeline - Parallel Execution**
![Event Timeline](images/event-timeline.png)

**B.3. Stages - Tasks Distribution**
![Stages](images/stages-parallel.png)

**B.4. HDFS NameNode - Storage Status**
![HDFS Status](images/hdfs-status.png)

### C. Code Repository

https://github.com/taitranfebruary/Bigdata_DeepfakeHunter

### D. References

1. Apache Spark Documentation. https://spark.apache.org/docs/latest/
2. Hadoop HDFS Guide. https://hadoop.apache.org/docs/stable/
3. Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
4. CIFAKE Dataset. https://www.kaggle.com/datasets/birdy654/cifake

---

**HẾT BÁO CÁO**
