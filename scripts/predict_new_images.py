#!/usr/bin/env python3
"""
Script: Predict New Images
Upload ảnh mới, extract features, và phân loại REAL/FAKE
sử dụng model đã train sẵn
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit, when
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StandardScalerModel
import sys
import subprocess
import time

FEATURE_DIM = 1280

def upload_to_hdfs(local_path, hdfs_path):
    """Upload ảnh từ local lên HDFS"""
    print(f"\n📤 Uploading {local_path} to HDFS...")
    
    # Xóa folder cũ nếu có
    subprocess.run(f"hdfs dfs -rm -r -f {hdfs_path}", shell=True, capture_output=True)
    
    # Upload
    cmd = f'hdfs dfs -put "{local_path}" "{hdfs_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Uploaded successfully!")
        # Count files
        count_cmd = f"hdfs dfs -count {hdfs_path}"
        count_result = subprocess.run(count_cmd, shell=True, capture_output=True, text=True)
        if count_result.returncode == 0:
            parts = count_result.stdout.strip().split()
            if len(parts) >= 2:
                print(f"   Total files: {parts[1]}")
        return True
    else:
        print(f"❌ Upload failed: {result.stderr}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict_new_images.py <local_image_folder>")
        print("Example: python3 predict_new_images.py /scripts/new_images")
        sys.exit(1)
    
    local_folder = sys.argv[1]
    
    print("=" * 70)
    print("🔮 DEEPFAKE DETECTION - NEW IMAGES PREDICTION")
    print("=" * 70)
    
    # Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("DeepfakeHunter-NewImagesPrediction") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"Spark Version: {spark.version}")
    print(f"App ID: {spark.sparkContext.applicationId}")
    
    # =====================================
    # STEP 1: Upload ảnh lên HDFS
    # =====================================
    hdfs_path = "hdfs://namenode:8020/raw/new_images"
    
    if not upload_to_hdfs(local_folder, hdfs_path):
        print("Upload failed. Exiting...")
        spark.stop()
        sys.exit(1)
    
    # =====================================
    # STEP 2: Extract Features (MobileNetV2)
    # =====================================
    print("\n" + "=" * 70)
    print("🧠 Extracting Features with MobileNetV2...")
    print("=" * 70)
    
    # UDF để extract features
    @udf(returnType=ArrayType(FloatType()))
    def extract_mobilenet_features(image_bytes):
        """Extract features từ image bytes sử dụng MobileNetV2"""
        import torch
        import torchvision.transforms as transforms
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        from PIL import Image
        import io
        
        if image_bytes is None:
            return [0.0] * 1280
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert('RGB')
            
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            img_tensor = preprocess(img).unsqueeze(0)
            
            global _mobilenet_model
            if '_mobilenet_model' not in globals() or _mobilenet_model is None:
                _mobilenet_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                _mobilenet_model.eval()
                _mobilenet_model.classifier = torch.nn.Identity()
            
            with torch.no_grad():
                features = _mobilenet_model(img_tensor)
                features = features.squeeze().numpy()
            
            return features.tolist()
            
        except Exception as e:
            print(f"Error: {e}")
            return [0.0] * 1280
    
    @udf(returnType=VectorUDT())
    def array_to_vector(arr):
        if arr is None:
            return Vectors.dense([0.0] * FEATURE_DIM)
        return Vectors.dense(arr)
    
    # Đọc ảnh từ HDFS
    print(f"Reading images from: {hdfs_path}")
    
    df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(hdfs_path)
    
    # Hỗ trợ cả .png
    df_png = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.png") \
        .option("recursiveFileLookup", "true") \
        .load(hdfs_path)
    
    df = df.union(df_png)
    
    total_images = df.count()
    print(f"Found {total_images} images")
    
    if total_images == 0:
        print("No images found!")
        spark.stop()
        sys.exit(1)
    
    # Extract features
    print("Extracting features (this may take a while)...")
    start_time = time.time()
    
    df_features = df.select(
        col("path"),
        extract_mobilenet_features(col("content")).alias("features_array")
    )
    
    df_final = df_features.select(
        col("path"),
        array_to_vector(col("features_array")).alias("features")
    )
    
    # Cache để dùng nhiều lần
    df_final = df_final.cache()
    df_final.count()  # Force execution
    
    elapsed = time.time() - start_time
    print(f"✓ Feature extraction completed in {elapsed:.1f}s")
    
    # =====================================
    # STEP 3: Load Model và Predict
    # =====================================
    print("\n" + "=" * 70)
    print("🎯 Loading Model and Making Predictions...")
    print("=" * 70)
    
    try:
        # Load Scaler (model đã train với scaled features)
        print("Loading StandardScaler...")
        # Tạo scaler giống như trong script 04
        from pyspark.ml.feature import StandardScaler
        
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=False
        )
        
        # Fit scaler trên data hiện tại
        scaler_model = scaler.fit(df_final)
        df_scaled = scaler_model.transform(df_final)
        print("✓ Features scaled")
        
        # Load Logistic Regression model
        print("Loading Logistic Regression model...")
        model_path = "hdfs://namenode:8020/results/models/logistic_regression"
        lr_model = LogisticRegressionModel.load(model_path)
        print("✓ Model loaded successfully!")
        
        # Predict
        print("Making predictions...")
        predictions = lr_model.transform(df_scaled)
        
        # Add human-readable labels
        predictions = predictions.withColumn(
            "result",
            when(col("prediction") == 0, "REAL").otherwise("FAKE")
        )
        
        # Extract probability UDF
        from pyspark.sql.types import DoubleType
        
        @udf(returnType=DoubleType())
        def get_fake_probability(probability_vector):
            """Extract probability of FAKE class (index 1)"""
            if probability_vector is None:
                return 0.0
            try:
                # probability_vector is DenseVector or SparseVector
                return float(probability_vector[1]) * 100
            except:
                return 0.0
        
        predictions = predictions.withColumn(
            "confidence",
            get_fake_probability(col("probability"))
        )
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nNote: Make sure you have run script 04_train_classifier.py first!")
        spark.stop()
        sys.exit(1)
    
    # =====================================
    # STEP 4: Display Results
    # =====================================
    print("\n" + "=" * 70)
    print("📊 PREDICTION RESULTS")
    print("=" * 70)
    
    # Count by prediction
    results_summary = predictions.groupBy("result").count().collect()
    
    print("\nSummary:")
    for row in results_summary:
        print(f"  {row['result']}: {row['count']} images")
    
    # Show sample predictions
    print("\nSample predictions:")
    predictions.select("path", "result", "confidence") \
        .orderBy(col("confidence").desc()) \
        .show(20, truncate=60)
    
    # =====================================
    # STEP 5: Save Results
    # =====================================
    print("\n" + "=" * 70)
    print("💾 Saving Results...")
    print("=" * 70)
    
    # Save to HDFS
    output_hdfs = "hdfs://namenode:8020/results/new_predictions.parquet"
    predictions.select("path", "prediction", "result", "confidence") \
        .write.mode("overwrite").parquet(output_hdfs)
    print(f"✓ Saved to HDFS: {output_hdfs}")
    
    # Save to local CSV
    output_local = "/scripts/output/new_predictions.csv"
    predictions.select("path", "result", "confidence") \
        .toPandas() \
        .to_csv(output_local, index=False)
    print(f"✓ Saved to local: {output_local}")
    
    # =====================================
    # STEP 6: Generate HTML Report
    # =====================================
    print("\n" + "=" * 70)
    print("📄 Generating HTML Report...")
    print("=" * 70)
    
    # Get all predictions as pandas
    results_df = predictions.select("path", "result", "confidence").toPandas()
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Deepfake Detection Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            flex: 1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .summary-card .number {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .real {{ color: #10b981; }}
        .fake {{ color: #ef4444; }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
        }}
        .badge-real {{
            background: #d1fae5;
            color: #065f46;
        }}
        .badge-fake {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .confidence {{
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔮 Deepfake Detection Results</h1>
        <p>Analyzed {len(results_df)} images using MobileNetV2 + Logistic Regression</p>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <h3>Total Images</h3>
            <div class="number">{len(results_df)}</div>
        </div>
        <div class="summary-card">
            <h3>Real Images</h3>
            <div class="number real">{len(results_df[results_df['result'] == 'REAL'])}</div>
        </div>
        <div class="summary-card">
            <h3>Fake Images</h3>
            <div class="number fake">{len(results_df[results_df['result'] == 'FAKE'])}</div>
        </div>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Image Path</th>
                <th>Result</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for _, row in results_df.iterrows():
        badge_class = "badge-real" if row['result'] == 'REAL' else "badge-fake"
        html_content += f"""
            <tr>
                <td>{row['path'].split('/')[-1]}</td>
                <td><span class="badge {badge_class}">{row['result']}</span></td>
                <td class="confidence">{row['confidence']:.2f}%</td>
            </tr>
"""
    
    html_content += """
        </tbody>
    </table>
</body>
</html>
"""
    
    # Save HTML
    html_path = "/scripts/output/prediction_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML report saved: {html_path}")
    
    # =====================================
    # SUMMARY
    # =====================================
    print("\n" + "=" * 70)
    print("✅ PREDICTION COMPLETED!")
    print("=" * 70)
    print(f"""
Results:
  - HDFS: {output_hdfs}
  - CSV:  {output_local}
  - HTML: {html_path}
  
Total: {total_images} images analyzed
  - REAL: {len(results_df[results_df['result'] == 'REAL'])} images
  - FAKE: {len(results_df[results_df['result'] == 'FAKE'])} images
""")
    
    spark.stop()

if __name__ == "__main__":
    main()
