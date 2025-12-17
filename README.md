# 🔍 Deepfake Hunter - Big Data Pipeline

**Đồ Án Môn Học: Xây dựng Pipeline Big Data Phân tán**  
**Môn học:** Thực hành Big Data  
**Nền tảng:** Local Hadoop/Spark Cluster (Docker)

[![Spark](https://img.shields.io/badge/Spark-3.3.0-orange)](https://spark.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop-3.2.1-yellow)](https://hadoop.apache.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)

## 📋 Tổng quan

Pipeline End-to-End phát hiện ảnh AI-generated (Deepfake) sử dụng:
- **HDFS** cho distributed storage  
- **Apache Spark** cho distributed processing
- **MobileNetV2** (ImageNet pretrained) cho feature extraction
- **Spark MLlib** cho distributed machine learning

### 🎯 Kết quả đạt được
- ✅ **Accuracy: 88.99%** (Logistic Regression)
- ✅ **AUC-ROC: 95.85%**
- ✅ Xử lý **120,000 ảnh** phân tán trên Spark cluster
- ✅ Tuân thủ 100% yêu cầu đồ án

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   CIFAKE    │───▶│    HDFS     │───▶│ MobileNetV2  │───▶│  Spark ML   │───▶│   Results    │
│   Images    │    │   Storage   │    │  Features    │    │  Training   │    │   Report     │
│ (120,000)   │    │   /raw/     │    │  (1280-dim)  │    │  LR + RF    │    │   Parquet    │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

---

## ✅ Tuân thủ yêu cầu kỹ thuật

| Yêu cầu | Trạng thái | Bằng chứng |
|---------|------------|------------|
| **1. HDFS Storage** | ✅ | 120,000 files in `/raw/cifake/` |
| **2. No os.listdir** | ✅ | `spark.read.format("binaryFile")` |
| **3. Distributed AI (UDF)** | ✅ | MobileNetV2 runs in Spark Workers |
| **4. Parquet Output** | ✅ | All results in Parquet format |
| **5. Spark History Server** | ✅ | Logs in `/spark-logs/` (HDFS) |

---

## 🚀 Cài đặt & Chạy

### Prerequisites
- Docker Desktop
- RAM: 16GB recommended
- Disk: 20GB free space

### 1. Clone repository
```bash
git clone https://github.com/taitranfebruary/Bigdata_DeepfakeHunter.git
cd Bigdata_DeepfakeHunter
```

### 2. Download Dataset
Download CIFAKE dataset và giải nén vào `dataset/archive/`

### 3. Khởi động Cluster
```bash
docker compose up -d
```

### 4. Chạy Pipeline

**Upload dữ liệu lên HDFS:**
```bash
docker exec -it spark-master python3 /scripts/02_upload_to_hdfs.py
```

**Feature Extraction:**
```bash
docker exec -it spark-master spark-submit \
  --master spark://spark-master:7077 \
  --executor-memory 8g \
  --executor-cores 4 \
  --driver-memory 3g \
  /scripts/03_feature_extraction.py
```

**Train Models:**
```bash
docker exec -it spark-master spark-submit \
  --master spark://spark-master:7077 \
  --executor-memory 8g \
  --executor-cores 4 \
  --driver-memory 3g \
  /scripts/04_train_classifier.py
```

**Generate Report:**
```bash
docker exec -it spark-master spark-submit /scripts/05_business_insight.py
docker exec -it spark-master python3 /scripts/08_generate_html_report.py
docker cp spark-master:/scripts/report.html ./
```

---

## 📊 Kết quả

### Model Performance

| Model | Accuracy | AUC-ROC | Train Time |
|-------|----------|---------|------------|
| **Logistic Regression** | **88.99%** | **95.85%** | 21.10s |
| Random Forest | 87.02% | 94.45% | 97.33s |

### Business Question Answer

**Q: "Liệu model được chọn có trích xuất đủ thông tin để phát hiện Deepfake không?"**

**A: ✅ CÓ** - Với accuracy 88.99%, MobileNetV2 features đủ khả năng phát hiện Deepfake.

---

## 🌐 Web UIs

- Spark Master: http://localhost:8080
- **Spark History Server:** **http://localhost:18080**
- HDFS NameNode: http://localhost:9870

---

## 📁 Cấu trúc project

```
├── docker-compose.yml
├── requirements.runtime.txt
├── images/
│   ├── spark-master/Dockerfile
│   └── spark-worker/Dockerfile
└── scripts/
    ├── 02_upload_to_hdfs.py
    ├── 03_feature_extraction.py
    ├── 04_train_classifier.py
    ├── 05_business_insight.py
    └── 08_generate_html_report.py
```

---

## 📚 Documentation

Xem file [REPORT.md](REPORT.md) để biết chi tiết về:
- Kiến trúc chi tiết
- Phương pháp implementation
- Kết quả phân tích
- Screenshots từ Spark History Server

---

**⭐ Star repo nếu hữu ích!**
