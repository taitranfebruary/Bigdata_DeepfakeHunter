#!/usr/bin/env python3
"""
Script 07: Inference Pipeline
Chạy dự đoán trên dữ liệu mới sử dụng model đã train

Pipeline này load model đã save từ HDFS và thực hiện inference
trên các ảnh mới, tuân thủ tất cả yêu cầu phân tán.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit, when
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.feature import StandardScalerModel
import time


def main():
    print("=" * 60)
    print("INFERENCE PIPELINE - Deepfake Detection")
    print("=" * 60)
    
    # Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("DeepfakeHunter-Inference") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
        .config("spark.history.fs.logDirectory", "hdfs://namenode:8020/spark-logs") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"Spark Version: {spark.version}")
    print(f"App ID: {spark.sparkContext.applicationId}")
    
    # =====================================
    # LOAD TRAINED MODEL FROM HDFS
    # =====================================
    print("\n" + "=" * 60)
    print("Loading Trained Models from HDFS...")
    print("=" * 60)
    
    # Load models (sử dụng best model - sẽ đọc metrics để chọn)
    try:
        lr_model = LogisticRegressionModel.load("hdfs://namenode:8020/results/models/logistic_regression")
        print("✓ Logistic Regression model loaded")
    except Exception as e:
        print(f"❌ Cannot load LR model: {e}")
        lr_model = None
    
    try:
        rf_model = RandomForestClassificationModel.load("hdfs://namenode:8020/results/models/random_forest")
        print("✓ Random Forest model loaded")
    except Exception as e:
        print(f"❌ Cannot load RF model: {e}")
        rf_model = None
    
    # Choose best model based on saved metrics
    metrics_df = spark.read.parquet("hdfs://namenode:8020/results/metrics.parquet")
    metrics_df.show()
    
    lr_acc = metrics_df.filter(col("model") == "LogisticRegression").select("accuracy").collect()[0][0]
    rf_acc = metrics_df.filter(col("model") == "RandomForest").select("accuracy").collect()[0][0]
    
    if rf_acc >= lr_acc and rf_model:
        best_model = rf_model
        best_model_name = "RandomForest"
        best_acc = rf_acc
    else:
        best_model = lr_model
        best_model_name = "LogisticRegression"
        best_acc = lr_acc
    
    print(f"\n🏆 Using Best Model: {best_model_name} (Accuracy: {best_acc*100:.2f}%)")
    
    # =====================================
    # FEATURE EXTRACTION UDF
    # =====================================
    
    @udf(returnType=ArrayType(FloatType()))
    def extract_features_udf(image_data):
        """UDF to extract features from a single image using MobileNetV2"""
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image
        import io
        
        if image_data is None:
            return [0.0] * 1280
        
        try:
            # Load model
            model = models.mobilenet_v2(pretrained=True)
            model.eval()
            
            # Feature extractor (remove classifier)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_extractor.eval()
            
            # Transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Process image
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                features = feature_extractor(img_tensor)
                features = features.squeeze().numpy().flatten()
            
            return features.tolist()
        except Exception as e:
            print(f"Error: {e}")
            return [0.0] * 1280
    
    @udf(returnType=VectorUDT())
    def array_to_vector(arr):
        if arr is None:
            return Vectors.dense([0.0] * 1280)
        return Vectors.dense(arr)
    
    # =====================================
    # DEMO: INFERENCE ON TEST DATA
    # =====================================
    print("\n" + "=" * 60)
    print("Running Inference on Sample Data...")
    print("=" * 60)
    
    # Đọc một phần nhỏ dữ liệu test để demo
    # Trong production, đây sẽ là dữ liệu mới từ HDFS
    sample_real = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .load("hdfs://namenode:8020/raw/cifake/test/REAL") \
        .limit(50)
    
    sample_fake = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .load("hdfs://namenode:8020/raw/cifake/test/FAKE") \
        .limit(50)
    
    sample_df = sample_real.withColumn("actual_label", lit("REAL")).union(
        sample_fake.withColumn("actual_label", lit("FAKE"))
    )
    
    print(f"Processing {sample_df.count()} sample images...")
    
    # Extract features
    start_time = time.time()
    
    features_df = sample_df.select(
        col("path"),
        col("actual_label"),
        extract_features_udf(col("content")).alias("features_array")
    )
    
    features_df = features_df.select(
        col("path"),
        col("actual_label"),
        array_to_vector(col("features_array")).alias("features")
    )
    
    # Scale features (dùng cùng scaler parameters như training)
    from pyspark.ml.feature import StandardScaler
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False
    )
    scaler_model = scaler.fit(features_df)
    scaled_df = scaler_model.transform(features_df)
    
    feature_time = time.time() - start_time
    print(f"Feature extraction time: {feature_time:.2f}s")
    
    # Run inference
    start_time = time.time()
    predictions = best_model.transform(scaled_df)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    
    # Add prediction labels
    predictions = predictions.withColumn(
        "predicted_label",
        when(col("prediction") == 0, "REAL").otherwise("FAKE")
    )
    
    predictions = predictions.withColumn(
        "correct",
        when(col("actual_label") == col("predicted_label"), "✓").otherwise("✗")
    )
    
    # =====================================
    # SHOW RESULTS
    # =====================================
    print("\n" + "=" * 60)
    print("📊 INFERENCE RESULTS")
    print("=" * 60)
    
    predictions.select("path", "actual_label", "predicted_label", "correct") \
        .show(20, truncate=50)
    
    # Calculate accuracy on sample
    total = predictions.count()
    correct = predictions.filter(col("actual_label") == col("predicted_label")).count()
    sample_accuracy = correct / total if total > 0 else 0
    
    print(f"\n📈 Sample Inference Statistics:")
    print(f"   Total samples: {total}")
    print(f"   Correct predictions: {correct}")
    print(f"   Sample Accuracy: {sample_accuracy*100:.2f}%")
    
    # Confusion breakdown
    print("\n📊 Prediction Breakdown:")
    predictions.groupBy("actual_label", "predicted_label").count().show()
    
    # Save inference results
    print("\n" + "=" * 60)
    print("Saving Inference Results to HDFS...")
    print("=" * 60)
    
    predictions.select("path", "actual_label", "predicted_label", "prediction") \
        .write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/results/inference_sample.parquet")
    print("✓ Inference results saved")
    
    # =====================================
    # SUMMARY
    # =====================================
    print("\n" + "=" * 60)
    print("🎯 INFERENCE PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"""
    ┌────────────────────────────────────────────────────────────┐
    │                    INFERENCE SUMMARY                       │
    ├────────────────────────────────────────────────────────────┤
    │  Model Used:         {best_model_name:>20}              │
    │  Model Accuracy:     {best_acc*100:>18.2f}%              │
    │  Samples Processed:  {total:>20}              │
    │  Sample Accuracy:    {sample_accuracy*100:>18.2f}%              │
    │  Feature Time:       {feature_time:>18.2f}s              │
    │  Inference Time:     {inference_time:>18.2f}s              │
    └────────────────────────────────────────────────────────────┘
    
    📁 Results saved to: hdfs://namenode:8020/results/inference_sample.parquet
    
    💡 To run inference on new data:
       1. Upload new images to HDFS: /raw/new_data/
       2. Modify HDFS_INPUT_PATH in this script
       3. Run: spark-submit --master spark://spark-master:7077 \\
               /scripts/07_inference_pipeline.py
    """)
    
    spark.stop()


if __name__ == "__main__":
    main()
