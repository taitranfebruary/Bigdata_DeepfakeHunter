#!/usr/bin/env python3
"""
Web Demo - Upload ảnh và phân loại REAL/FAKE
Streamlit web interface để upload ảnh trực tiếp
"""

import streamlit as st
import os
import time
import subprocess
import pandas as pd
from pathlib import Path
import shutil

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "demo_upload")
OUTPUT_DIR = os.path.join(BASE_DIR, "scripts/output")

def run_prediction(image_folder):
    """Chạy prediction script trong Docker container"""
    # XÓA dữ liệu cũ trước
    container_path = "/scripts/demo_upload"
    
    # Xóa folder cũ trong container
    subprocess.run(
        f"docker exec spark-master rm -rf {container_path}",
        shell=True, capture_output=True
    )
    
    # Xóa file CSV cũ
    csv_path = os.path.join(OUTPUT_DIR, "new_predictions.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    # Copy folder MỚI vào container
    copy_cmd = f"docker cp {image_folder} spark-master:{container_path}"
    subprocess.run(copy_cmd, shell=True, capture_output=True)
    
    # Chạy prediction trong container
    cmd = f"docker exec spark-master spark-submit /scripts/predict_new_images.py {container_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Copy kết quả MỚI ra ngoài
    if result.returncode == 0:
        subprocess.run(
            "docker cp spark-master:/scripts/output/new_predictions.csv " + csv_path,
            shell=True, capture_output=True
        )
    
    return result.returncode == 0, result.stdout, result.stderr

def main():
    st.set_page_config(
        page_title="Deepfake Detection",
        page_icon="🔮",
        layout="wide"
    )
    
    # Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 10px; text-align: center; color: white;'>
        <h1>🔮 Deepfake Detection System</h1>
        <p>Upload ảnh để phát hiện REAL hoặc FAKE sử dụng MobileNetV2 + Spark</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tạo upload directory
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📤 Upload ảnh (JPG/PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Chọn một hoặc nhiều ảnh để phân tích"
    )
    
    if uploaded_files:
        st.success(f"✅ Đã chọn {len(uploaded_files)} ảnh")
        
        # Preview images
        cols = st.columns(5)
        for idx, uploaded_file in enumerate(uploaded_files[:10]):
            with cols[idx % 5]:
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        
        if len(uploaded_files) > 10:
            st.info(f"... và {len(uploaded_files) - 10} ảnh khác")
        
        # Analyze button
        if st.button("🔍 Phân tích ngay", type="primary", use_container_width=True):
            # Clear old files
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            
            # Save uploaded files
            with st.spinner("📥 Đang lưu ảnh..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                time.sleep(0.5)
            
            st.success("✅ Đã lưu ảnh!")
            
            # Run prediction
            with st.spinner("🧠 Đang phân tích với MobileNetV2... (có thể mất 1-2 phút)"):
                progress_bar = st.progress(0)
                
                success, stdout, stderr = run_prediction(UPLOAD_DIR)
                
                progress_bar.progress(100)
            
            if success:
                st.success("✅ Phân tích hoàn tất!")
                
                # Load results
                csv_path = os.path.join(OUTPUT_DIR, "new_predictions.csv")
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    total = len(df)
                    real_count = len(df[df['result'] == 'REAL'])
                    fake_count = len(df[df['result'] == 'FAKE'])
                    
                    with col1:
                        st.metric("📊 Tổng số ảnh", total)
                    with col2:
                        st.metric("✅ REAL", real_count, delta=f"{real_count/total*100:.1f}%")
                    with col3:
                        st.metric("❌ FAKE", fake_count, delta=f"{fake_count/total*100:.1f}%", delta_color="inverse")
                    
                    st.markdown("---")
                    
                    # Results table
                    st.subheader("📋 Kết quả chi tiết")
                    
                    # Format dataframe
                    df['image'] = df['path'].apply(lambda x: x.split('/')[-1])
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2f}%")
                    
                    # Color coding
                    def highlight_result(row):
                        if row['result'] == 'REAL':
                            return ['background-color: #d1fae5'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    
                    st.dataframe(
                        df[['image', 'result', 'confidence']].style.apply(highlight_result, axis=1),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download results
                    st.download_button(
                        label="📥 Tải xuống kết quả (CSV)",
                        data=df.to_csv(index=False),
                        file_name="deepfake_results.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("❌ Không tìm thấy file kết quả")
                    st.text(stdout[-1000:])
            else:
                st.error("❌ Phân tích thất bại!")
                with st.expander("Xem log lỗi"):
                    st.text(stderr)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Big Data Project - Deepfake Detection System</p>
        <p>Technology: MobileNetV2 + Spark + HDFS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
