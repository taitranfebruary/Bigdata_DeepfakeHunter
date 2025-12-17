#!/usr/bin/env python3
"""
Script 03: Feature Extraction using MobileNetV2 (Pre-trained on ImageNet)
Trích xuất đặc trưng từ ảnh bằng MobileNetV2 - model được train trên ImageNet
Chạy phân tán trên Spark Workers sử dụng UDF

QUAN TRỌNG - TUÂN THỦ YÊU CẦU ĐỒ ÁN:
- Dùng Spark để đọc ảnh từ HDFS (binaryFile format)
- KHÔNG dùng os.listdir để duyệt file
- Dùng MobileNetV2 (ImageNet pretrained) để trích xuất features
- UDF chạy phân tán trên Spark Workers
- Kết quả lưu về HDFS dưới dạng Parquet

Feature Vector: 1280 dimensions (MobileNetV2 output)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT

# Feature dimension from MobileNetV2
FEATURE_DIM = 1280

def main():
    print("=" * 60)
    print("STEP 3: Feature Extraction with MobileNetV2")
    print("=" * 60)
    
    # Khởi tạo Spark Session với cấu hình cho distributed processing
    spark = SparkSession.builder \
        .appName("DeepfakeHunter-MobileNetV2-FeatureExtraction") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
        .config("spark.history.fs.logDirectory", "hdfs://namenode:8020/spark-logs") \
        .config("spark.python.worker.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"Spark Version: {spark.version}")
    print(f"App ID: {spark.sparkContext.applicationId}")
    
    # UDF để extract features từ image bytes sử dụng MobileNetV2
    @udf(returnType=ArrayType(FloatType()))
    def extract_mobilenet_features(image_bytes):
        """
        Extract features từ image bytes sử dụng MobileNetV2 pre-trained on ImageNet
        Chạy bên trong mỗi Spark Worker (Distributed AI Inference)
        
        Model: MobileNetV2 (ImageNet pretrained)
        Output: 1280-dimensional feature vector
        """
        import torch
        import torchvision.transforms as transforms
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        from PIL import Image
        import io
        
        if image_bytes is None:
            return [0.0] * 1280
        
        try:
            # Load and preprocess image
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert('RGB')
            
            # MobileNetV2 preprocessing (ImageNet normalization)
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
            
            # Load MobileNetV2 model (cached per worker)
            # Sử dụng global variable để cache model
            global _mobilenet_model
            if '_mobilenet_model' not in globals() or _mobilenet_model is None:
                _mobilenet_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                _mobilenet_model.eval()
                # Remove classifier to get features only
                _mobilenet_model.classifier = torch.nn.Identity()
            
            # Extract features (no gradient computation needed)
            with torch.no_grad():
                features = _mobilenet_model(img_tensor)
                features = features.squeeze().numpy()
            
            return features.tolist()
            
        except Exception as e:
            print(f"Error extracting MobileNetV2 features: {e}")
            return [0.0] * 1280
    
    # UDF để convert array to Spark ML Vector
    @udf(returnType=VectorUDT())
    def array_to_vector(arr):
        if arr is None:
            return Vectors.dense([0.0] * FEATURE_DIM)
        return Vectors.dense(arr)
    
    # Process function cho mỗi dataset split
    def process_split(split_name, label_name, label_value):
        """Process một split của dataset từ HDFS"""
        hdfs_path = f"hdfs://namenode:8020/raw/cifake/{split_name}/{label_name}"
        print(f"\n{'='*40}")
        print(f"Processing: {split_name}/{label_name}")
        print(f"HDFS Path: {hdfs_path}")
        print(f"Model: MobileNetV2 (ImageNet pretrained)")
        print(f"{'='*40}")
        
        # Đọc ảnh từ HDFS sử dụng Spark binaryFile format
        # KHÔNG dùng os.listdir - tuân thủ quy tắc đồ án
        df = spark.read.format("binaryFile") \
            .option("pathGlobFilter", "*.jpg") \
            .option("recursiveFileLookup", "true") \
            .load(hdfs_path)
        
        count = df.count()
        print(f"Found {count} images")
        
        if count == 0:
            # Return empty DataFrame with correct schema
            schema = StructType([
                StructField("path", StringType(), True),
                StructField("features", VectorUDT(), True),
                StructField("label", IntegerType(), True),
                StructField("label_name", StringType(), True),
                StructField("split", StringType(), True)
            ])
            return spark.createDataFrame([], schema)
        
        # Giảm số partition để giảm overhead (4 partitions cho mỗi split)
        # Mỗi partition xử lý ~12,500 ảnh
        df = df.coalesce(4)
        
        # Extract features sử dụng MobileNetV2 UDF (Distributed AI Inference)
        # Model chạy bên trong Spark Workers
        df_features = df.select(
            col("path"),
            extract_mobilenet_features(col("content")).alias("features_array"),
            lit(label_value).alias("label"),
            lit(label_name).alias("label_name"),
            lit(split_name).alias("split")
        )
        
        # Convert to Spark ML Vector (yêu cầu để dùng Spark MLlib)
        df_final = df_features.select(
            col("path"),
            array_to_vector(col("features_array")).alias("features"),
            col("label"),
            col("label_name"),
            col("split")
        )
        
        return df_final
    
    # Process tất cả splits - LƯU TỪNG FOLDER RIÊNG để tránh timeout
    print("\n" + "=" * 60)
    print("Processing Training Data with MobileNetV2...")
    print("=" * 60)
    
    # TRAIN REAL - Process và lưu riêng
    train_real = process_split("train", "REAL", 0)
    train_real_output = "hdfs://namenode:8020/processed/train_real_features.parquet"
    print(f"\nSaving train/REAL to: {train_real_output}")
    train_real.write.mode("overwrite").parquet(train_real_output)
    print("✓ train/REAL saved!")
    
    # TRAIN FAKE - Process và lưu riêng
    train_fake = process_split("train", "FAKE", 1)
    train_fake_output = "hdfs://namenode:8020/processed/train_fake_features.parquet"
    print(f"\nSaving train/FAKE to: {train_fake_output}")
    train_fake.write.mode("overwrite").parquet(train_fake_output)
    print("✓ train/FAKE saved!")
    
    # Merge train data
    print("\nMerging training data...")
    train_real_loaded = spark.read.parquet(train_real_output)
    train_fake_loaded = spark.read.parquet(train_fake_output)
    train_df = train_real_loaded.union(train_fake_loaded)
    
    train_count = train_df.count()
    print(f"Total training samples: {train_count}")
    
    # Lưu merged training features
    train_output = "hdfs://namenode:8020/processed/train_features.parquet"
    print(f"\nSaving merged to: {train_output}")
    train_df.write.mode("overwrite").parquet(train_output)
    print("✓ Training features saved to HDFS!")
    
    # TEST - Cũng lưu từng folder riêng
    print("\n" + "=" * 60)
    print("Processing Test Data with MobileNetV2...")
    print("=" * 60)
    
    # TEST REAL
    test_real = process_split("test", "REAL", 0)
    test_real_output = "hdfs://namenode:8020/processed/test_real_features.parquet"
    print(f"\nSaving test/REAL to: {test_real_output}")
    test_real.write.mode("overwrite").parquet(test_real_output)
    print("✓ test/REAL saved!")
    
    # TEST FAKE
    test_fake = process_split("test", "FAKE", 1)
    test_fake_output = "hdfs://namenode:8020/processed/test_fake_features.parquet"
    print(f"\nSaving test/FAKE to: {test_fake_output}")
    test_fake.write.mode("overwrite").parquet(test_fake_output)
    print("✓ test/FAKE saved!")
    
    # Merge test data
    print("\nMerging test data...")
    test_real_loaded = spark.read.parquet(test_real_output)
    test_fake_loaded = spark.read.parquet(test_fake_output)
    test_df = test_real_loaded.union(test_fake_loaded)
    
    test_count = test_df.count()
    print(f"Total test samples: {test_count}")
    
    # Lưu merged test features
    test_output = "hdfs://namenode:8020/processed/test_features.parquet"
    print(f"\nSaving merged to: {test_output}")
    test_df.write.mode("overwrite").parquet(test_output)
    print("✓ Test features saved to HDFS!")
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED!")
    print("=" * 60)
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           MOBILENETV2 FEATURE EXTRACTION SUMMARY              ║
╠══════════════════════════════════════════════════════════════╣
║ Model: MobileNetV2 (Pre-trained on ImageNet)                  ║
║ Feature Dimension: {FEATURE_DIM}                                      ║
║ Training samples: {train_count}                                    ║
║ Test samples: {test_count}                                        ║
╠══════════════════════════════════════════════════════════════╣
║ Output Files (HDFS):                                          ║
║ - {train_output}          ║
║ - {test_output}           ║
╠══════════════════════════════════════════════════════════════╣
║ ✓ Compliant with project requirements:                        ║
║   [✓] No os.listdir - using Spark binaryFile                  ║
║   [✓] MobileNetV2 (ImageNet) for feature extraction           ║
║   [✓] Distributed inference via Spark UDF                     ║
║   [✓] Results saved as Parquet on HDFS                        ║
╚══════════════════════════════════════════════════════════════╝
""")
    print(f"📊 Check Spark History Server at http://localhost:18080")
    
    spark.stop()

if __name__ == "__main__":
    main()
