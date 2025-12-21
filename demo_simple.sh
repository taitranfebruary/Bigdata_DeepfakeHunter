#!/bin/bash
# Script Demo Deepfake Detection - Đơn giản

echo "=================================="
echo "🔮 DEEPFAKE DETECTION DEMO"
echo "=================================="

# Kiểm tra tham số
if [ -z "$1" ]; then
    echo "❌ Vui lòng chỉ định folder ảnh!"
    echo ""
    echo "Cách dùng:"
    echo "  ./demo_simple.sh <folder_ảnh>"
    echo ""
    echo "Ví dụ:"
    echo "  ./demo_simple.sh new_images"
    echo "  ./demo_simple.sh demo_images"
    exit 1
fi

IMAGE_FOLDER="$1"

# Kiểm tra folder tồn tại
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ Folder không tồn tại: $IMAGE_FOLDER"
    exit 1
fi

echo ""
echo "📂 Folder: $IMAGE_FOLDER"
echo "📊 Số ảnh: $(find "$IMAGE_FOLDER" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)"
echo ""

# Copy vào container
echo "📤 Đang upload ảnh vào container..."
docker cp "$IMAGE_FOLDER" spark-master:/scripts/demo_images

# Chạy prediction
echo "🧠 Đang phân tích với MobileNetV2..."
docker exec spark-master spark-submit /scripts/predict_new_images.py /scripts/demo_images 2>&1 | grep -E "(PREDICTION|Summary|Sample predictions|Total:|REAL:|FAKE:)" | tail -20

# Copy kết quả ra
echo ""
echo "💾 Đang lưu kết quả..."
docker cp spark-master:/scripts/output/prediction_report.html ./demo_result.html
docker cp spark-master:/scripts/output/new_predictions.csv ./demo_result.csv

echo ""
echo "=================================="
echo "✅ HOÀN THÀNH!"
echo "=================================="
echo ""
echo "📄 Xem kết quả:"
echo "   - HTML: demo_result.html"
echo "   - CSV:  demo_result.csv"
echo ""
echo "🌐 Mở HTML trong browser để xem chi tiết!"
echo ""

# Mở HTML (macOS)
open demo_result.html 2>/dev/null
