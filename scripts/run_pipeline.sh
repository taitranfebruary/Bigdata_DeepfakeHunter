#!/bin/bash
#==============================================================================
# DEEPFAKE HUNTER - FULL PIPELINE EXECUTION SCRIPT
#==============================================================================
# 
# Script này chạy toàn bộ pipeline End-to-End với HDFS:
# 0. Verify environment (Spark, HDFS, PyTorch)
# 1. Setup HDFS directories
# 2. Upload raw data to HDFS (batch upload, không dùng os.listdir)
# 3. Feature Extraction (MobileNetV2 trên Spark Workers)
# 4. Train Classifier (Spark MLlib)
# 5. Generate Business Insight Report
# 6. Generate HTML Report
#
# Usage: 
#   docker exec -it spark-master bash /scripts/run_pipeline.sh
#
# Yêu cầu đồ án:
#   - Raw data phải nằm trên HDFS
#   - Không sử dụng os.listdir để duyệt từng ảnh
#   - Spark xử lý phân tán với MobileNetV2
#   - Kết quả lưu trên HDFS dưới dạng Parquet
#   - Spark History Server để theo dõi jobs
#
#==============================================================================

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       DEEPFAKE HUNTER - BIG DATA PIPELINE (HDFS)                 ║"
echo "║       Distributed End-to-End ML Pipeline                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Spark configuration for event logging
SPARK_CONF="--conf spark.eventLog.enabled=true \
--conf spark.eventLog.dir=hdfs://namenode:8020/spark-logs \
--conf spark.history.fs.logDirectory=hdfs://namenode:8020/spark-logs"

# Function to print step
print_step() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}▶ STEP $1: $2${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Function to check success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 completed successfully!${NC}"
    else
        echo -e "${RED}✗ $1 failed!${NC}"
        exit 1
    fi
}

# Start time
START_TIME=$(date +%s)

#==============================================================================
# STEP 0: Verify Environment
#==============================================================================
print_step "0/6" "Verifying environment (Spark, HDFS, PyTorch)"

python3 /scripts/00_verify_env.py
check_success "Environment verification"

#==============================================================================
# STEP 1: Setup HDFS Directories
#==============================================================================
print_step "1/6" "Setting up HDFS directories"

python3 /scripts/01_setup_hdfs.py
check_success "HDFS directory setup"

# Verify HDFS directories
echo ""
echo -e "${BLUE}📁 HDFS Directory Structure:${NC}"
hdfs dfs -ls hdfs://namenode:8020/ 2>/dev/null || echo "HDFS not fully ready yet"

#==============================================================================
# STEP 2: Upload Raw Data to HDFS
#==============================================================================
print_step "2/6" "Uploading raw data to HDFS (batch upload)"

python3 /scripts/02_upload_to_hdfs.py
check_success "Data uploaded to HDFS"

# Verify data on HDFS
echo ""
echo -e "${BLUE}📊 Data on HDFS:${NC}"
hdfs dfs -ls hdfs://namenode:8020/raw/cifake/train/FAKE/ 2>/dev/null | head -5 || echo "Checking HDFS..."
echo ""
echo "Training FAKE images:"
hdfs dfs -count hdfs://namenode:8020/raw/cifake/train/FAKE/ 2>/dev/null || echo "N/A"
echo "Training REAL images:"
hdfs dfs -count hdfs://namenode:8020/raw/cifake/train/REAL/ 2>/dev/null || echo "N/A"
echo "Test FAKE images:"
hdfs dfs -count hdfs://namenode:8020/raw/cifake/test/FAKE/ 2>/dev/null || echo "N/A"
echo "Test REAL images:"
hdfs dfs -count hdfs://namenode:8020/raw/cifake/test/REAL/ 2>/dev/null || echo "N/A"

#==============================================================================
# STEP 3: Feature Extraction (MobileNetV2 on Spark)
#==============================================================================
print_step "3/6" "Feature Extraction (MobileNetV2 distributed on Spark)"

echo -e "${YELLOW}Using MobileNetV2 pretrained on ImageNet for feature extraction${NC}"
echo "Output: 1280-dimensional feature vectors per image"
echo ""

# Submit Spark job for feature extraction with event logging
spark-submit \
    --master spark://spark-master:7077 \
    --deploy-mode client \
    --executor-memory 4g \
    --driver-memory 4g \
    $SPARK_CONF \
    /scripts/03_feature_extraction.py

check_success "Feature extraction"

# Verify features on HDFS
echo ""
echo -e "${BLUE}📊 Extracted Features on HDFS:${NC}"
hdfs dfs -ls hdfs://namenode:8020/processed/features/ 2>/dev/null || echo "Checking HDFS..."

#==============================================================================
# STEP 4: Train Classifier (Spark MLlib)
#==============================================================================
print_step "4/6" "Training Distributed Classifier (LogisticRegression & RandomForest)"

echo -e "${YELLOW}Training with Spark MLlib:${NC}"
echo "  - LogisticRegression (baseline)"
echo "  - RandomForestClassifier (ensemble)"
echo ""

# Submit Spark job for training with event logging
spark-submit \
    --master spark://spark-master:7077 \
    --deploy-mode client \
    --executor-memory 4g \
    --driver-memory 4g \
    $SPARK_CONF \
    /scripts/04_train_classifier.py

check_success "Model training"

# Verify models on HDFS
echo ""
echo -e "${BLUE}📊 Models & Results on HDFS:${NC}"
hdfs dfs -ls hdfs://namenode:8020/results/ 2>/dev/null || echo "Checking HDFS..."

#==============================================================================
# STEP 5: Generate Business Insight Report
#==============================================================================
print_step "5/6" "Generating Business Insight Report"

# Submit Spark job for business insight with event logging
spark-submit \
    --master spark://spark-master:7077 \
    --deploy-mode client \
    --executor-memory 2g \
    --driver-memory 2g \
    $SPARK_CONF \
    /scripts/05_business_insight.py

check_success "Business insight generation"

# Verify insights on HDFS
echo ""
echo -e "${BLUE}📊 Business Insights on HDFS:${NC}"
hdfs dfs -ls hdfs://namenode:8020/results/insights/ 2>/dev/null || echo "Checking HDFS..."

#==============================================================================
# STEP 6: Generate HTML Report
#==============================================================================
print_step "6/6" "Generating HTML Report"

spark-submit \
    --master spark://spark-master:7077 \
    --deploy-mode client \
    $SPARK_CONF \
    /scripts/08_generate_html_report.py

check_success "HTML report generation"

#==============================================================================
# FINAL SUMMARY
#==============================================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    🎉 PIPELINE COMPLETED! 🎉                      ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
printf "║  Total execution time: %dm %ds                                ║\n" $TOTAL_MINUTES $TOTAL_SECONDS
echo "║                                                                  ║"
echo "║  📁 HDFS Results:                                                ║"
echo "║     hdfs://namenode:8020/raw/cifake/    - Raw images             ║"
echo "║     hdfs://namenode:8020/processed/     - Extracted features     ║"
echo "║     hdfs://namenode:8020/results/       - Models & metrics       ║"
echo "║     hdfs://namenode:8020/spark-logs/    - Spark event logs       ║"
echo "║                                                                  ║"
echo "║  🌐 Web UIs:                                                     ║"
echo "║     Spark Master:         http://localhost:8080                  ║"
echo "║     Spark History Server: http://localhost:18080                 ║"
echo "║     HDFS NameNode:        http://localhost:9870                  ║"
echo "║                                                                  ║"
echo "║  📊 View Results:                                                ║"
echo "║     HTML Report: docker cp spark-master:/scripts/report.html .   ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Show HDFS results summary
echo "📂 HDFS Results Summary:"
echo ""
echo "Raw Data:"
hdfs dfs -du -h hdfs://namenode:8020/raw/cifake/ 2>/dev/null || echo "N/A"
echo ""
echo "Processed Features:"
hdfs dfs -du -h hdfs://namenode:8020/processed/ 2>/dev/null || echo "N/A"
echo ""
echo "Results:"
hdfs dfs -du -h hdfs://namenode:8020/results/ 2>/dev/null || echo "N/A"
echo ""
echo "Spark Logs (for History Server):"
hdfs dfs -ls hdfs://namenode:8020/spark-logs/ 2>/dev/null | head -5 || echo "N/A"

echo ""
echo "📄 HTML Report generated at /scripts/report.html"
echo "   Run: docker cp spark-master:/scripts/report.html ./report.html"
echo "   Then open report.html in your browser"
echo ""
echo "📈 View Spark Job History at: http://localhost:18080"
