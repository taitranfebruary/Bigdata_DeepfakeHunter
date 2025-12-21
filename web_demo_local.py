#!/usr/bin/env python3
"""
Web Demo - Upload ảnh và phân loại REAL/FAKE (Local version)
Streamlit web interface chạy local, gọi Docker container để predict
"""

import streamlit as st
import os
import time
import subprocess
import pandas as pd
from pathlib import Path
import shutil
import tempfile

def run_prediction_in_docker(local_folder):
    """Upload folder vào docker và chạy prediction"""
    # Copy folder vào container
    cmd1 = f"docker exec spark-master rm -rf /scripts/demo_upload"
    subprocess.run(cmd1, shell=True, capture_output=True)
    
    cmd2 = f"docker cp {local_folder} spark-master:/scripts/demo_upload"
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        return False, "", f"Failed to copy: {result.stderr}"
    
    # Run prediction
    cmd3 = "docker exec spark-master spark-submit /scripts/predict_new_images.py /scripts/demo_upload"
    result = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
    
    # Copy results back
    cmd4 = "docker cp spark-master:/scripts/output/new_predictions.csv /tmp/predictions.csv"
    subprocess.run(cmd4, shell=True, capture_output=True)
    
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
            # Create temp folder
            with tempfile.TemporaryDirectory() as temp_dir:
                upload_folder = os.path.join(temp_dir, "images")
                os.makedirs(upload_folder, exist_ok=True)
                
                # Save uploaded files
                with st.spinner("📥 Đang lưu ảnh..."):
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(upload_folder, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    time.sleep(0.5)
                
                st.success("✅ Đã lưu ảnh!")
                
                # Run prediction
                with st.spinner("🧠 Đang phân tích với MobileNetV2... (có thể mất 1-2 phút)"):
                    progress_bar = st.progress(0)
                    progress_bar.progress(30)
                    
                    success, stdout, stderr = run_prediction_in_docker(upload_folder)
                    
                    progress_bar.progress(100)
                
                if success:
                    st.success("✅ Phân tích hoàn tất!")
                    
                    # Load results
                    csv_path = "/tmp/predictions.csv"
                    
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
                        
                        display_df = df[['image', 'result', 'confidence']].copy()
                        
                        st.dataframe(
                            display_df.style.apply(highlight_result, axis=1),
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
                        with st.expander("Xem output"):
                            st.text(stdout[-2000:])
                else:
                    st.error("❌ Phân tích thất bại!")
                    with st.expander("Xem log lỗi"):
                        st.text(stderr[-2000:])
                        st.text(stdout[-2000:])
    
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
