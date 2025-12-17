#!/usr/bin/env python3
"""
Script 06: View Results - Advanced Visualization
Xem kết quả từ HDFS sau khi chạy pipeline với visualization chi tiết
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, sum as spark_sum, round as spark_round


def print_banner(title, width=70):
    """Print a formatted banner"""
    print("\n" + "╔" + "═" * (width-2) + "╗")
    print("║" + title.center(width-2) + "║")
    print("╚" + "═" * (width-2) + "╝")


def print_section(title, width=60):
    """Print section header"""
    print("\n" + "─" * width)
    print(f"📊 {title}")
    print("─" * width)


def print_metric_box(metrics):
    """Print metrics in a nice box format"""
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                      MODEL PERFORMANCE METRICS                      │
    ├──────────────────┬────────────────────┬────────────────────────────┤
    │ Metric           │ Logistic Regression│ Random Forest              │
    ├──────────────────┼────────────────────┼────────────────────────────┤""")
    
    for m in metrics:
        if m['model'] == 'LogisticRegression':
            lr = m
        else:
            rf = m
    
    print(f"    │ Accuracy         │ {lr['accuracy']*100:>16.2f}% │ {rf['accuracy']*100:>24.2f}% │")
    print(f"    │ Precision        │ {lr['precision']*100:>16.2f}% │ {rf['precision']*100:>24.2f}% │")
    print(f"    │ Recall           │ {lr['recall']*100:>16.2f}% │ {rf['recall']*100:>24.2f}% │")
    print(f"    │ F1-Score         │ {lr['f1_score']*100:>16.2f}% │ {rf['f1_score']*100:>24.2f}% │")
    print(f"    │ AUC-ROC          │ {lr['auc_roc']*100:>16.2f}% │ {rf['auc_roc']*100:>24.2f}% │")
    print(f"    │ Training Time    │ {lr['train_time_seconds']:>16.2f}s │ {rf['train_time_seconds']:>24.2f}s │")
    print("    └──────────────────┴────────────────────┴────────────────────────────┘")
    
    # Determine winner
    winner = "Random Forest 🏆" if rf['accuracy'] > lr['accuracy'] else "Logistic Regression 🏆"
    print(f"\n    🎯 Best Model: {winner}")


def print_confusion_matrix(cm_data, model_name):
    """Print confusion matrix visualization"""
    for cm in cm_data:
        if cm['model'] == model_name:
            tp, tn, fp, fn = cm['true_positive'], cm['true_negative'], cm['false_positive'], cm['false_negative']
            total = tp + tn + fp + fn
            
            print(f"""
    {model_name} Confusion Matrix:
    ┌─────────────────────┬───────────────────┬───────────────────┐
    │                     │ Predicted REAL    │ Predicted FAKE    │
    ├─────────────────────┼───────────────────┼───────────────────┤
    │ Actual REAL         │ TN = {tn:>6}       │ FP = {fp:>6}       │
    │ Actual FAKE         │ FN = {fn:>6}       │ TP = {tp:>6}       │
    └─────────────────────┴───────────────────┴───────────────────┘
    
    📈 Derived Metrics:
       • Accuracy:    {(tp+tn)/total*100:.2f}%
       • Precision:   {tp/(tp+fp)*100:.2f}% (of predicted FAKE, how many are correct)
       • Recall:      {tp/(tp+fn)*100:.2f}% (of actual FAKE, how many detected)
       • Specificity: {tn/(tn+fp)*100:.2f}% (of actual REAL, how many correct)
       • FPR:         {fp/(tn+fp)*100:.2f}% (False Positive Rate)
       • FNR:         {fn/(tp+fn)*100:.2f}% (False Negative Rate - CRITICAL!)
            """)


def print_ascii_bar_chart(values, labels, title, max_width=40):
    """Print ASCII bar chart"""
    print(f"\n    {title}")
    print("    " + "─" * 50)
    max_val = max(values) if values else 1
    for label, value in zip(labels, values):
        bar_len = int((value / max_val) * max_width) if max_val > 0 else 0
        bar = "█" * bar_len
        print(f"    {label:>15} │{bar} {value:.2f}%")
    print()


def main():
    print_banner("DEEPFAKE HUNTER - RESULTS DASHBOARD", 70)
    
    # Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("DeepfakeHunter-ViewResults") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # =====================================
    # 1. View Metrics
    # =====================================
    print_section("MODEL PERFORMANCE METRICS")
    
    try:
        metrics_df = spark.read.parquet("hdfs://namenode:8020/results/metrics.parquet")
        metrics_data = metrics_df.collect()
        metrics_dict = [row.asDict() for row in metrics_data]
        print_metric_box(metrics_dict)
        
        # ASCII bar chart for comparison
        lr_acc = rf_acc = 0
        for m in metrics_dict:
            if m['model'] == 'LogisticRegression':
                lr_acc = m['accuracy'] * 100
            else:
                rf_acc = m['accuracy'] * 100
        
        print_ascii_bar_chart(
            [lr_acc, rf_acc],
            ['LogisticReg', 'RandomForest'],
            "Accuracy Comparison"
        )
        
    except Exception as e:
        print(f"❌ Cannot read metrics: {e}")
    
    # =====================================
    # 2. View Confusion Matrix
    # =====================================
    print_section("CONFUSION MATRIX ANALYSIS")
    
    try:
        cm_df = spark.read.parquet("hdfs://namenode:8020/results/confusion_matrix.parquet")
        cm_data = [row.asDict() for row in cm_df.collect()]
        
        print_confusion_matrix(cm_data, "LogisticRegression")
        print_confusion_matrix(cm_data, "RandomForest")
        
    except Exception as e:
        print(f"❌ Cannot read confusion matrix: {e}")
    
    # =====================================
    # 3. View Business Insight
    # =====================================
    print_section("BUSINESS INSIGHT SUMMARY")
    
    try:
        insight_df = spark.read.parquet("hdfs://namenode:8020/results/business_insight.parquet")
        
        print("\n    📋 Key Insights:")
        for row in insight_df.collect():
            print(f"       • {row['metric']}: {row['value']}")
            
    except Exception as e:
        print(f"❌ Cannot read business insight: {e}")
    
    # =====================================
    # 4. Sample Predictions Analysis
    # =====================================
    print_section("PREDICTION ANALYSIS")
    
    try:
        lr_pred = spark.read.parquet("hdfs://namenode:8020/results/lr_predictions.parquet")
        rf_pred = spark.read.parquet("hdfs://namenode:8020/results/rf_predictions.parquet")
        
        lr_total = lr_pred.count()
        rf_total = rf_pred.count()
        
        print(f"\n    📊 Prediction Statistics:")
        print(f"       Total test samples: {lr_total}")
        
        # LR breakdown
        lr_correct = lr_pred.filter(col("label") == col("prediction")).count()
        lr_wrong = lr_total - lr_correct
        
        # RF breakdown  
        rf_correct = rf_pred.filter(col("label") == col("prediction")).count()
        rf_wrong = rf_total - rf_correct
        
        print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │                    PREDICTION BREAKDOWN                      │
    ├─────────────────────┬──────────────────┬────────────────────┤
    │                     │ LogisticReg      │ RandomForest       │
    ├─────────────────────┼──────────────────┼────────────────────┤
    │ Correct Predictions │ {lr_correct:>12}     │ {rf_correct:>14}     │
    │ Wrong Predictions   │ {lr_wrong:>12}     │ {rf_wrong:>14}     │
    │ Accuracy            │ {lr_correct/lr_total*100:>11.2f}%     │ {rf_correct/rf_total*100:>13.2f}%     │
    └─────────────────────┴──────────────────┴────────────────────┘
        """)
        
        # Sample predictions
        print("\n    📋 Sample Predictions (First 10):")
        lr_pred.select("path", "label", "label_name", "prediction") \
            .withColumn("correct", when(col("label") == col("prediction"), "✓").otherwise("✗")) \
            .show(10, truncate=45)
            
    except Exception as e:
        print(f"❌ Cannot read predictions: {e}")
    
    # =====================================
    # 5. Feature Statistics
    # =====================================
    print_section("DATASET & FEATURE STATISTICS")
    
    try:
        train_features = spark.read.parquet("hdfs://namenode:8020/processed/train_features.parquet")
        test_features = spark.read.parquet("hdfs://namenode:8020/processed/test_features.parquet")
        
        train_count = train_features.count()
        test_count = test_features.count()
        total_count = train_count + test_count
        
        print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │                    DATASET STATISTICS                        │
    ├─────────────────────────────────────────────────────────────┤
    │  Dataset: CIFAKE (Real vs AI-Generated)                      │
    │  Feature Extractor: MobileNetV2 (ImageNet pretrained)        │
    │  Feature Dimension: 1280                                     │
    ├─────────────────────────────────────────────────────────────┤
    │  Training samples:  {train_count:>8}                                    │
    │  Test samples:      {test_count:>8}                                    │
    │  Total samples:     {total_count:>8}                                    │
    │  Train/Test ratio:  {train_count/total_count*100:.1f}% / {test_count/total_count*100:.1f}%                              │
    └─────────────────────────────────────────────────────────────┘
        """)
        
        # Label distribution
        print("\n    📊 Label Distribution:")
        print("\n    Training Set:")
        train_dist = train_features.groupBy("label_name").count().collect()
        for row in train_dist:
            pct = row['count'] / train_count * 100
            bar = "█" * int(pct / 2)
            print(f"       {row['label_name']:>6}: {row['count']:>6} ({pct:.1f}%) {bar}")
        
        print("\n    Test Set:")
        test_dist = test_features.groupBy("label_name").count().collect()
        for row in test_dist:
            pct = row['count'] / test_count * 100
            bar = "█" * int(pct / 2)
            print(f"       {row['label_name']:>6}: {row['count']:>6} ({pct:.1f}%) {bar}")
            
    except Exception as e:
        print(f"❌ Cannot read features: {e}")
    
    # =====================================
    # 6. HDFS Storage Summary
    # =====================================
    print_section("HDFS STORAGE SUMMARY")
    
    print("""
    📁 HDFS Directory Structure:
    
    /raw/cifake/                    ← Raw image data
    ├── train/
    │   ├── REAL/                   (~50,000 images)
    │   └── FAKE/                   (~50,000 images)
    └── test/
        ├── REAL/                   (~10,000 images)
        └── FAKE/                   (~10,000 images)
    
    /processed/                     ← Extracted features (Parquet)
    ├── train_features.parquet
    └── test_features.parquet
    
    /results/                       ← Model outputs
    ├── metrics.parquet
    ├── lr_predictions.parquet
    ├── rf_predictions.parquet
    ├── confusion_matrix.parquet
    ├── business_insight.parquet
    └── models/
        ├── logistic_regression/
        └── random_forest/
    
    /spark-logs/                    ← Spark History Server logs
    """)
    
    # =====================================
    # 7. Final Answer to Business Question
    # =====================================
    print_banner("ANSWER TO BUSINESS QUESTION", 70)
    
    try:
        metrics_df = spark.read.parquet("hdfs://namenode:8020/results/metrics.parquet")
        best_acc = metrics_df.agg({"accuracy": "max"}).collect()[0][0]
        
        answer = "✅ CÓ" if best_acc > 0.7 else "❌ CHƯA ĐỦ"
        
        print(f"""
    ❓ CÂU HỎI: 
       "Liệu model được chọn có trích xuất đủ thông tin để phát hiện 
        Deepfake không?"

    💡 TRẢ LỜI: {answer}

    📊 GIẢI THÍCH:
       • Accuracy đạt được: {best_acc*100:.2f}%
       • MobileNetV2 (pretrained ImageNet) trích xuất 1280 features
       • Features này chứa thông tin về textures, edges, patterns
       • AI-generated images có artifacts mà model có thể phát hiện:
         - Smooth textures không tự nhiên
         - Inconsistent lighting/shadows
         - Subtle pattern repetitions
       • Hybrid approach (DL features + Classical ML) hoạt động tốt
         và có thể scale trên Spark cluster

    🎯 KẾT LUẬN:
       MobileNetV2 features KẾT HỢP với Spark MLlib classifiers
       {"ĐỦ KHẢ NĂNG" if best_acc > 0.7 else "CHƯA ĐỦ KHẢ NĂNG"} phát hiện Deepfake trong dataset CIFAKE
       với độ chính xác {best_acc*100:.2f}%.
        """)
    except Exception as e:
        print(f"❌ Cannot generate answer: {e}")
    
    spark.stop()
    
    print_banner("RESULTS VIEWING COMPLETED ✅", 70)

if __name__ == "__main__":
    main()
