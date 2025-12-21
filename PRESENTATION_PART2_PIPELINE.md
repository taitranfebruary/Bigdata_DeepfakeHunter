# 🎤 THUYẾT TRÌNH PHẦN 2: YÊU CẦU PIPELINE & BUSINESS INSIGHT

## 📋 Tổng quan Phần 2

**Nội dung:** Trình bày 4 bước Pipeline + Kết quả Business Insight  
**Thời gian:** ~10-12 phút  
**Mục tiêu:** Chứng minh pipeline hoạt động từ đầu đến cuối, đạt kết quả tốt

---

# 🔄 STEP 1: NẠP DỮ LIỆU (DATA INGESTION)

## 📊 Script: `02_upload_to_hdfs.py`

### Thuyết trình:

> "**Bước đầu tiên trong pipeline là nạp dữ liệu.**
> 
> Em sử dụng dataset **CIFAKE** - một dataset chuyên về phát hiện ảnh AI-generated với **120,000 ảnh** được chia thành:
> - **Training set:** 100,000 ảnh (50,000 REAL + 50,000 FAKE)
> - **Test set:** 20,000 ảnh (10,000 REAL + 10,000 FAKE)
>
> Như yêu cầu đồ án, em **bắt buộc phải upload toàn bộ dữ liệu lên HDFS** trước khi xử lý, không được đọc trực tiếp từ ổ cứng local."

### 📸 Screenshot cần chỉ (đã chuẩn bị):
1. **Mở HDFS NameNode:** http://localhost:9870
2. **Browse the file system** → `/raw/cifake/`
3. **Chỉ vào màn hình:**
   - `/raw/cifake/train/REAL/` - 50,000 files
   - `/raw/cifake/train/FAKE/` - 50,000 files
   - `/raw/cifake/test/REAL/` - 10,000 files
   - `/raw/cifake/test/FAKE/` - 10,000 files

### 💬 Đọc kèm:

> "Các thầy cô có thể thấy trên HDFS Web UI, em đã upload thành công 120,000 ảnh vào HDFS theo đúng cấu trúc thư mục. Đây là bằng chứng em tuân thủ yêu cầu 1: **Bắt buộc dùng HDFS**."

### 📝 Code quan trọng (nếu được hỏi):
```python
# File: 02_upload_to_hdfs.py (Line 45)
cmd = f'hdfs dfs -put "{local_path}" "{hdfs_folder}"'
subprocess.run(cmd, shell=True)

# Kết quả: Toàn bộ data đã ở hdfs://namenode:8020/raw/cifake/
```

---

# 🤖 STEP 2: TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION)

## 📊 Script: `03_feature_extraction.py`

### Thuyết trình:

> "**Bước thứ hai là trích xuất đặc trưng từ ảnh.**
>
> Theo yêu cầu đồ án:
> - ❌ **KHÔNG được dùng** model Deepfake Detector có sẵn
> - ✅ **Phải dùng** model pretrained trên ImageNet như ResNet50 hoặc MobileNetV2
>
> Em đã chọn **MobileNetV2** pretrained trên ImageNet vì:
> 1. **Nhẹ hơn ResNet50** - phù hợp với tài nguyên cluster
> 2. **Tốc độ inference nhanh** - quan trọng khi xử lý 120,000 ảnh
> 3. **Feature dimension 1280** - đủ rich để capture AI artifacts
>
> Quan trọng hơn, em đã implement **Distributed AI Inference** - model MobileNetV2 chạy **bên trong mỗi Spark Worker** qua UDF, không phải trên Driver."

### 🎯 Điểm nhấn quan trọng:

> "Đây là điểm khác biệt với code Python thường. Thay vì dùng vòng lặp `for` để xử lý từng ảnh, em dùng **Spark DataFrame** với **UDF** để MobileNetV2 chạy **phân tán** trên 1 Worker với 4 cores - data được chia thành 4 partitions xử lý song song."

### 📸 Screenshot cần chỉ:

1. **Mở Code Editor** → File `03_feature_extraction.py`
   - **Chỉ vào dòng 46-93:** UDF definition
   ```python
   @udf(returnType=ArrayType(FloatType()))
   def extract_mobilenet_features(image_bytes):
       # Model load TRONG mỗi Worker
       _mobilenet_model = mobilenet_v2(
           weights=MobileNet_V2_Weights.IMAGENET1K_V1
       )
   ```

2. **Mở Spark History Server:** http://localhost:18080
   - **Click vào job:** "DeepfakeHunter-MobileNetV2-FeatureExtraction"
   - **Tab "Stages":** Chỉ vào danh sách tasks chạy parallel
   - **Tab "Event Timeline":** Chỉ vào biểu đồ tasks overlap (chạy đồng thời)
   - **Tab "Executors":** Chỉ vào executor với 4 cores đang xử lý tasks

3. **Mở HDFS:** Browse `/processed/`
   - `train_features.parquet` - 100,000 samples × 1280 dims
   - `test_features.parquet` - 20,000 samples × 1280 dims

### 💬 Đọc kèm:

> "Các thầy cô xem Spark History Server, có thể thấy rõ:
> - Job chạy với **nhiều Stages**
> - Mỗi Stage có **nhiều Tasks** chạy song song trên 4 cores
> - **Event Timeline** cho thấy tasks overlap - chứng minh parallel execution
> - Kết quả được lưu dưới dạng **Parquet** trên HDFS
>
> Đây là bằng chứng em tuân thủ yêu cầu 2, 3, và 4:
> - ✅ KHÔNG dùng os.listdir
> - ✅ AI chạy phân tán qua UDF trên 1 worker với 4 cores
> - ✅ Kết quả lưu Parquet trên HDFS"

### 📝 Code quan trọng:

```python
# Đọc từ HDFS bằng Spark (KHÔNG dùng os.listdir)
df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .load("hdfs://namenode:8020/raw/cifake/train/REAL")

# Extract features phân tán qua UDF
df_features = df.select(
    extract_mobilenet_features(col("content")).alias("features")
)

# Lưu Parquet vào HDFS
df_features.write.parquet("hdfs://namenode:8020/processed/train_features.parquet")
```

---

# 🧠 STEP 3: PHÂN LOẠI PHÂN TÁN (DISTRIBUTED CLASSIFICATION)

## 📊 Script: `04_train_classifier.py`

### Thuyết trình:

> "**Bước thứ ba là huấn luyện bộ phân loại phân tán.**
>
> Theo yêu cầu đồ án, em phải:
> 1. **Convert mảng features thành Spark ML Vector**
> 2. **Sử dụng Spark MLlib** để train classifier cổ điển
>
> Em đã train 2 models để so sánh:
> - **Logistic Regression** - Simple, fast, linear classifier
> - **Random Forest** - Complex, non-linear, ensemble method
>
> Cả hai đều chạy **phân tán trên Spark cluster**, không phải local Python."

### 🎯 Điểm nhấn - Giải thích Hybrid Model:

> "Đây là một **Hybrid Model approach:**
> - **Deep Learning (MobileNetV2):** Extract high-level features từ ảnh
> - **Classical ML (LogisticRegression):** Phân loại dựa trên features đó
>
> Ưu điểm:
> - **Tận dụng transfer learning** từ ImageNet
> - **Training nhanh** - chỉ cần train classifier, không cần train toàn bộ CNN
> - **Scalable** - dễ dàng phân tán với Spark MLlib"

### 📸 Screenshot cần chỉ:

1. **Mở Code Editor** → File `04_train_classifier.py`
   
   **A. Convert to Vector (chỉ vào dòng):**
   ```python
   # File: 03_feature_extraction.py (Line 95-102)
   @udf(returnType=VectorUDT())
   def array_to_vector(arr):
       return Vectors.dense(arr)  # Spark ML Vector
   ```
   
   **B. Train với Spark MLlib (chỉ vào dòng):**
   ```python
   # Logistic Regression
   lr = LogisticRegression(
       featuresCol="scaled_features",
       labelCol="label",
       maxIter=100
   )
   lr_model = lr.fit(train_scaled)  # Distributed training
   
   # Random Forest
   rf = RandomForestClassifier(
       featuresCol="scaled_features",
       labelCol="label",
       numTrees=50
   )
   rf_model = rf.fit(train_scaled)  # Distributed training
   ```

2. **Mở Spark History Server:** http://localhost:18080
   - **Click vào job:** "DeepfakeHunter-Classification"
   - **Chỉ vào Stages** của training process
   - "Các thầy cô thấy, quá trình training cũng được phân tán"

3. **Mở HDFS:** Browse `/results/models/`
   - `logistic_regression/` folder
   - `random_forest/` folder

### 💬 Đọc kèm:

> "Models đã được train thành công và lưu trên HDFS. Spark MLlib tự động phân tán:
> - Gradient computations (cho LR)
> - Tree building (cho RF)
> - Model aggregation
>
> Đây là sức mạnh của Spark MLlib so với Scikit-learn thông thường."

---

# ✅ STEP 4: KIỂM TRA KẾT QUẢ MÔ HÌNH

## 📊 Script: `04_train_classifier.py` (tiếp)

### Thuyết trình:

> "**Bước cuối cùng là đánh giá model trên test set.**
>
> Em sử dụng các **Evaluators từ Spark MLlib** để tính toán metrics trên test set 20,000 ảnh:
> - **Accuracy** - Tỉ lệ dự đoán đúng
> - **Precision** - Khi dự đoán FAKE, đúng bao nhiêu %
> - **Recall** - Phát hiện được bao nhiêu % ảnh FAKE thực sự
> - **F1-Score** - Harmonic mean của Precision và Recall
> - **AUC-ROC** - Khả năng phân biệt giữa 2 classes"

### 📸 Screenshot cần chỉ:

1. **Chạy lệnh trong terminal:**
   ```bash
   docker exec spark-master spark-submit \
       --master spark://spark-master:7077 \
       /scripts/06_view_results.py
   ```

2. **Chỉ vào output trên terminal:**
   ```
   ┌─────────────────┬──────────────┬────────────────┐
   │ Metric          │ LogisticReg  │ RandomForest   │
   ├─────────────────┼──────────────┼────────────────┤
   │ Accuracy        │ 88.99%       │ 87.02%         │
   │ Precision       │ 88.99%       │ 87.03%         │
   │ Recall          │ 88.99%       │ 87.02%         │
   │ F1-Score        │ 88.99%       │ 87.02%         │
   │ AUC-ROC         │ 95.85%       │ 94.45%         │
   │ Train Time      │ 21.10s       │ 97.33s         │
   └─────────────────┴──────────────┴────────────────┘
   ```

3. **Hoặc mở HTML Report:** `open report.html`
   - Chỉ vào biểu đồ so sánh
   - Chỉ vào Confusion Matrix

### 💬 Đọc kèm:

> "Kết quả cho thấy:
> - **Logistic Regression đạt 88.99% accuracy** - vượt ngưỡng 85% cho production
> - **AUC-ROC đạt 95.85%** - model phân biệt rất tốt giữa REAL và FAKE
> - **Training time chỉ 21 giây** - rất nhanh so với deep learning end-to-end
> - **Random Forest chậm hơn** (97s) nhưng accuracy thấp hơn một chút
>
> Vì vậy em chọn **Logistic Regression** làm model chính."

---

# 📈 PHẦN 3: KẾT QUẢ (BUSINESS INSIGHT)

## 🎯 Báo cáo các chỉ số của Hybrid Model

### Thuyết trình:

> "**Bây giờ em sẽ trả lời phần Business Insight.**
>
> Model của em là **Hybrid Model** kết hợp:
> - **MobileNetV2** (Deep Learning) để extract features
> - **Logistic Regression** (Classical ML) để classification
>
> Các chỉ số chính:"

### 📊 Bảng Metrics (chỉ vào màn hình):

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **Accuracy** | **88.99%** | Dự đoán đúng ~9/10 ảnh |
| **Precision** | **88.99%** | Khi nói FAKE, đúng 89% |
| **Recall** | **88.99%** | Phát hiện được 89% ảnh FAKE |
| **F1-Score** | **88.99%** | Balanced performance |
| **AUC-ROC** | **95.85%** | Rất tốt trong phân biệt classes |

### 💬 Giải thích cho người không chuyên:

> "Để dễ hiểu hơn:
> - **Accuracy 89%:** Nếu cho 100 ảnh, model đoán đúng 89 ảnh
> - **Precision 89%:** Khi model nói 'ảnh này FAKE', thì 89% là đúng
> - **Recall 89%:** Trong số ảnh FAKE thật sự, model tìm ra được 89%
> - **AUC-ROC 96%:** Model phân biệt REAL/FAKE rất tốt, gần hoàn hảo"

---

## 📊 Confusion Matrix (Phân tích chi tiết)

### 📸 Screenshot: Mở HTML report hoặc terminal output

```
Confusion Matrix (Test set: 20,000 ảnh)
┌─────────────────┬──────────────┬──────────────┐
│                 │ Predicted    │ Predicted    │
│                 │ REAL (0)     │ FAKE (1)     │
├─────────────────┼──────────────┼──────────────┤
│ Actual REAL (0) │ TN = 8,940   │ FP = 1,060   │
│ Actual FAKE (1) │ FN = 1,142   │ TP = 8,858   │
└─────────────────┴──────────────┴──────────────┘
```

### 💬 Giải thích:

> "Confusion Matrix cho thấy:
> - **True Negative (8,940):** Ảnh THẬT được nhận đúng là THẬT
> - **True Positive (8,858):** Ảnh GIẢ được nhận đúng là GIẢ
> - **False Positive (1,060):** Ảnh THẬT bị nhầm là GIẢ - 10.6%
> - **False Negative (1,142):** Ảnh GIẢ bị nhầm là THẬT - 11.4%
>
> False Negative nguy hiểm hơn vì bỏ sót deepfake, nhưng tỉ lệ chỉ 11% là chấp nhận được."

---

## ❓ TRẢ LỜI CÂU HỎI BUSINESS

### Câu hỏi:
> **"Liệu model được chọn có trích xuất đủ thông tin để phát hiện Deepfake không?"**

### 🎤 Trả lời (ĐỌC CHẬM, RÕ RÀNG):

> "**Câu trả lời là: CÓ, MobileNetV2 features đủ khả năng phát hiện Deepfake.**
>
> Em có 4 bằng chứng để khẳng định điều này:"

### 📌 Bằng chứng 1: Accuracy cao

> "**Accuracy 88.99%** vượt ngưỡng 85% được coi là đạt yêu cầu cho production systems. Với độ chính xác này, model đủ tin cậy để triển khai thực tế."

### 📌 Bằng chứng 2: AUC-ROC xuất sắc

> "**AUC-ROC 95.85%** chứng tỏ model có khả năng phân biệt rất tốt giữa ảnh thật và ảnh giả. Điểm số gần 96% cho thấy features đã capture được sự khác biệt quan trọng."

### 📌 Bằng chứng 3: Feature quality

> "MobileNetV2 được pretrain trên ImageNet với 1.4 triệu ảnh, nên nó đã học được:
> - **Low-level features:** Textures, edges, colors
> - **Mid-level features:** Shapes, patterns
> - **High-level features:** Semantic concepts
>
> Vector 1280 chiều này đủ để capture các **AI artifacts** tinh vi như:
> - Smooth textures không tự nhiên
> - Inconsistent lighting
> - Pattern repetitions của GAN"

### 📌 Bằng chứng 4: So sánh với baseline

> "Em có test với features đơn giản hơn (histogram + statistics):
> - Histogram features: **75.88% accuracy**
> - MobileNetV2 features: **88.99% accuracy**
>
> Cải thiện **+13%** chứng tỏ deep features chứa nhiều thông tin hơn."

---

## 🎯 KẾT LUẬN

### Tóm tắt:

> "**Kết luận cuối cùng:**
>
> ✅ **Pipeline đã tuân thủ 100% yêu cầu đồ án**
> - Dữ liệu trên HDFS
> - Không dùng vòng lặp local
> - AI chạy phân tán qua UDF
> - Kết quả lưu Parquet
> - Spark History Server có đầy đủ logs
>
> ✅ **Pipeline hoạt động hiệu quả**
> - 120,000 ảnh được xử lý phân tán
> - Training time nhanh (21 giây)
> - Accuracy cao (88.99%)
>
> ✅ **Model đủ khả năng phát hiện Deepfake**
> - MobileNetV2 features chứa đủ thông tin
> - Hybrid approach hiệu quả và scalable
> - Sẵn sàng cho production với monitoring"

---

## 💡 HỎI ĐÁP (DỰ ĐOÁN CÂU HỎI)

### Q1: "Tại sao không dùng ResNet50?"

**Trả lời:**
> "Em đã cân nhắc ResNet50 (2048-dim features) nhưng chọn MobileNetV2 vì:
> - **Lighter:** Phù hợp với tài nguyên cluster hiện tại
> - **Faster inference:** Quan trọng khi xử lý 120K ảnh
> - **Feature dimension 1280** đã đủ tốt, tăng lên 2048 chưa chắc cải thiện nhiều
>
> Trong thực tế production, có thể ensemble cả 2 models."

---

### Q2: "Tại sao Logistic Regression tốt hơn Random Forest?"

**Trả lời:**
> "Với features tốt từ MobileNetV2, decision boundary có vẻ gần linear:
> - **LR:** Simple, fast, tận dụng tốt linear separability
> - **RF:** Overkill cho bài toán này, train lâu hơn mà accuracy không cao hơn
>
> Đây cũng là insight: **Good features + Simple classifier** thường hiệu quả hơn **Bad features + Complex classifier**."

---

### Q3: "False Negative 11.4% có cao không?"

**Trả lời:**
> "Phụ thuộc use case:
> - **Social media screening:** 11.4% chấp nhận được, có human review bổ sung
> - **Critical applications:** Cần improve, có thể:
>   - Điều chỉnh threshold (trade-off Precision/Recall)
>   - Ensemble models
>   - Thêm features từ metadata
>
> Em recommend **threshold=0.4** để tăng Recall, giảm False Negative."

---

### Q4: "Pipeline này có scale được không?"

**Trả lời:**
> "✅ **Hoàn toàn có thể scale:**
> - **Horizontal scaling:** Thêm Workers vào cluster
> - **HDFS replication:** Đảm bảo fault tolerance
> - **Spark auto-partition:** Tự động chia data
> - **Model serving:** Deploy qua Spark Streaming hoặc REST API
>
> Kiến trúc này giống các hệ thống production tại Facebook, Netflix."

---

### Q5: "Có thể improve accuracy hơn nữa không?"

**Trả lời:**
> "Có nhiều hướng improve:
>
> **Short-term:**
> - Fine-tune MobileNetV2 trên CIFAKE dataset
> - Ensemble LR + RF
> - Data augmentation
>
> **Long-term:**
> - Try ResNet50, EfficientNet features
> - Ensemble multiple feature extractors
> - Add attention mechanism
> - Use transformer-based models
>
> Nhưng với yêu cầu đồ án (ImageNet pretrained + Classical ML), em đã đạt kết quả tốt."

---

## ⏱️ TIMELINE THUYẾT TRÌNH (10 phút)

```
00:00 - 02:00  Step 1: Data Ingestion
               → Show HDFS với 120K files
               
02:00 - 04:30  Step 2: Feature Extraction
               → Show code UDF
               → Show Spark History (parallel execution)
               → Show Parquet output
               
04:30 - 06:30  Step 3: Distributed Classification
               → Explain Hybrid Model
               → Show MLlib training code
               → Show model artifacts
               
06:30 - 07:30  Step 4: Model Evaluation
               → Show metrics table
               → Show Confusion Matrix
               
07:30 - 09:30  Business Insight
               → Answer key question với 4 bằng chứng
               → Conclusion
               
09:30 - 10:00  Q&A buffer
```

---

## ✅ CHECKLIST TRƯỚC KHI THUYẾT TRÌNH

- [ ] Docker containers đang chạy
- [ ] Tất cả Web UIs đã mở trong browser tabs:
  - [ ] http://localhost:9870 (HDFS)
  - [ ] http://localhost:8080 (Spark Master)
  - [ ] http://localhost:18080 (Spark History)
- [ ] Code editor mở các files:
  - [ ] 03_feature_extraction.py
  - [ ] 04_train_classifier.py
- [ ] Terminal sẵn sàng chạy lệnh
- [ ] File report.html đã mở
- [ ] Slide/PDF backup (nếu có)

---

## 🎯 KEY MESSAGES CẦN NHỚ

1. **Pipeline 4 bước:** Ingestion → Extraction → Classification → Evaluation
2. **Hybrid Model:** MobileNetV2 (DL) + LogisticRegression (ML)
3. **Tuân thủ 100%** yêu cầu kỹ thuật
4. **Kết quả tốt:** 88.99% accuracy, 95.85% AUC-ROC
5. **Trả lời Business:** CÓ, features đủ khả năng phát hiện Deepfake

---

**CHÚC BẠN THUYẾT TRÌNH THÀNH CÔNG! 🎉**
