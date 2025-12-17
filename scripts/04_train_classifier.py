#!/usr/bin/env python3
"""
Script 04: Train Classifier using Spark MLlib
Huấn luyện bộ phân loại LogisticRegression và RandomForest
trên các features đã trích xuất từ HDFS
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import col
import time

def main():
    print("=" * 60)
    print("STEP 4: Train Distributed Classifier")
    print("=" * 60)
    
    # Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("DeepfakeHunter-Classification") \
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
    
    # Load features từ HDFS
    print("\n" + "=" * 40)
    print("Loading features from HDFS...")
    print("=" * 40)
    
    train_df = spark.read.parquet("hdfs://namenode:8020/processed/train_features.parquet")
    test_df = spark.read.parquet("hdfs://namenode:8020/processed/test_features.parquet")
    
    print(f"Training samples: {train_df.count()}")
    print(f"Test samples: {test_df.count()}")
    
    # Cache data for faster training
    train_df = train_df.cache()
    test_df = test_df.cache()
    
    # Show sample
    print("\nSample data:")
    train_df.select("path", "label", "label_name").show(5, truncate=50)
    
    # StandardScaler for better performance
    print("\n" + "=" * 40)
    print("Scaling features...")
    print("=" * 40)
    
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False
    )
    
    scaler_model = scaler.fit(train_df)
    train_scaled = scaler_model.transform(train_df)
    test_scaled = scaler_model.transform(test_df)
    
    # =====================================
    # MODEL 1: Logistic Regression
    # =====================================
    print("\n" + "=" * 60)
    print("Training Model 1: Logistic Regression")
    print("=" * 60)
    
    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="label",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.8
    )
    
    start_time = time.time()
    lr_model = lr.fit(train_scaled)
    lr_train_time = time.time() - start_time
    print(f"Training time: {lr_train_time:.2f}s")
    
    # Predictions
    lr_predictions = lr_model.transform(test_scaled)
    
    # =====================================
    # MODEL 2: Random Forest
    # =====================================
    print("\n" + "=" * 60)
    print("Training Model 2: Random Forest")
    print("=" * 60)
    
    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="label",
        numTrees=50,
        maxDepth=10,
        seed=42
    )
    
    start_time = time.time()
    rf_model = rf.fit(train_scaled)
    rf_train_time = time.time() - start_time
    print(f"Training time: {rf_train_time:.2f}s")
    
    # Predictions
    rf_predictions = rf_model.transform(test_scaled)
    
    # =====================================
    # EVALUATION
    # =====================================
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Evaluators
    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    auc_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    
    # Evaluate Logistic Regression
    print("\n--- Logistic Regression Results ---")
    lr_accuracy = accuracy_eval.evaluate(lr_predictions)
    lr_precision = precision_eval.evaluate(lr_predictions)
    lr_recall = recall_eval.evaluate(lr_predictions)
    lr_f1 = f1_eval.evaluate(lr_predictions)
    lr_auc = auc_eval.evaluate(lr_predictions)
    
    print(f"Accuracy:  {lr_accuracy:.4f}")
    print(f"Precision: {lr_precision:.4f}")
    print(f"Recall:    {lr_recall:.4f}")
    print(f"F1-Score:  {lr_f1:.4f}")
    print(f"AUC-ROC:   {lr_auc:.4f}")
    
    # Evaluate Random Forest
    print("\n--- Random Forest Results ---")
    rf_accuracy = accuracy_eval.evaluate(rf_predictions)
    rf_precision = precision_eval.evaluate(rf_predictions)
    rf_recall = recall_eval.evaluate(rf_predictions)
    rf_f1 = f1_eval.evaluate(rf_predictions)
    rf_auc = auc_eval.evaluate(rf_predictions)
    
    print(f"Accuracy:  {rf_accuracy:.4f}")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall:    {rf_recall:.4f}")
    print(f"F1-Score:  {rf_f1:.4f}")
    print(f"AUC-ROC:   {rf_auc:.4f}")
    
    # =====================================
    # SAVE RESULTS
    # =====================================
    print("\n" + "=" * 60)
    print("Saving Results to HDFS")
    print("=" * 60)
    
    # Save predictions to HDFS (Parquet format - yêu cầu đồ án)
    lr_predictions.select("path", "label", "label_name", "prediction") \
        .write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/results/lr_predictions.parquet")
    print("✓ LR predictions saved to HDFS")
    
    rf_predictions.select("path", "label", "label_name", "prediction") \
        .write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/results/rf_predictions.parquet")
    print("✓ RF predictions saved to HDFS")
    
    # Save metrics as DataFrame to HDFS
    metrics_data = [
        ("LogisticRegression", lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc, lr_train_time),
        ("RandomForest", rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc, rf_train_time)
    ]
    metrics_df = spark.createDataFrame(
        metrics_data,
        ["model", "accuracy", "precision", "recall", "f1_score", "auc_roc", "train_time_seconds"]
    )
    
    metrics_df.write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/results/metrics.parquet")
    print("✓ Metrics saved to HDFS")
    
    # Save models to HDFS
    lr_model.write().overwrite().save("hdfs://namenode:8020/results/models/logistic_regression")
    rf_model.write().overwrite().save("hdfs://namenode:8020/results/models/random_forest")
    print("✓ Models saved to HDFS")
    
    # =====================================
    # SUMMARY
    # =====================================
    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETED!")
    print("=" * 60)
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                    MODEL COMPARISON                      │")
    print("├─────────────────┬──────────────┬────────────────────────┤")
    print("│ Metric          │ LogisticReg  │ RandomForest           │")
    print("├─────────────────┼──────────────┼────────────────────────┤")
    print(f"│ Accuracy        │ {lr_accuracy:.4f}       │ {rf_accuracy:.4f}                  │")
    print(f"│ Precision       │ {lr_precision:.4f}       │ {rf_precision:.4f}                  │")
    print(f"│ Recall          │ {lr_recall:.4f}       │ {rf_recall:.4f}                  │")
    print(f"│ F1-Score        │ {lr_f1:.4f}       │ {rf_f1:.4f}                  │")
    print(f"│ AUC-ROC         │ {lr_auc:.4f}       │ {rf_auc:.4f}                  │")
    print(f"│ Train Time (s)  │ {lr_train_time:.2f}        │ {rf_train_time:.2f}                   │")
    print("└─────────────────┴──────────────┴────────────────────────┘")
    
    print("\nResults saved to HDFS: hdfs://namenode:8020/results/")
    print("📊 Check Spark History Server at http://localhost:18080")
    
    spark.stop()

if __name__ == "__main__":
    main()
