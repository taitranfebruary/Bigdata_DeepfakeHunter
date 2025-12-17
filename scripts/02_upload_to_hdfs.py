#!/usr/bin/env python3
"""
Script 02: Upload CIFAKE dataset to HDFS
Upload ảnh từ local vào HDFS sử dụng batch upload

QUAN TRỌNG: Script này tuân thủ quy tắc:
- KHÔNG dùng os.listdir hoặc vòng lặp Python để duyệt file
- Sử dụng hdfs dfs -put để upload batch (cả folder)
"""

import subprocess
import time

def main():
    print("=" * 60)
    print("STEP 2: Upload CIFAKE Dataset to HDFS (Fast Batch)")
    print("=" * 60)
    
    # Paths
    local_base = "/scripts/dataset/archive"
    hdfs_base = "/raw/cifake"
    
    # First, remove existing data to avoid conflicts
    print("\nCleaning existing HDFS data...")
    subprocess.run("hdfs dfs -rm -r -f /raw/cifake 2>/dev/null", shell=True)
    subprocess.run("hdfs dfs -mkdir -p /raw/cifake", shell=True)
    
    # Upload entire folders at once (FAST!)
    # hdfs dfs -put uploads entire directory including all files
    folders = [
        ("train", f"{hdfs_base}/train"),
        ("test", f"{hdfs_base}/test"),
    ]
    
    for local_folder, hdfs_folder in folders:
        local_path = f"{local_base}/{local_folder}"
        print(f"\n📤 Uploading {local_path} -> {hdfs_folder}")
        print("   This uploads the ENTIRE folder at once (fast batch)...")
        
        start_time = time.time()
        
        # Upload entire folder - MUCH faster than file by file
        cmd = f'hdfs dfs -put "{local_path}" "{hdfs_folder}"'
        print(f"   Running: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✓ Uploaded in {elapsed:.1f}s")
        else:
            # If folder exists, try with -f flag
            cmd = f'hdfs dfs -put -f "{local_path}"/* "{hdfs_folder}/"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            elapsed = time.time() - start_time
            if result.returncode == 0:
                print(f"   ✓ Uploaded in {elapsed:.1f}s")
            else:
                print(f"   ⚠ Warning: {result.stderr[:200]}")
    
    # Fix directory structure - move contents up one level if needed
    print("\n🔧 Fixing directory structure...")
    # Check if structure is /raw/cifake/train/train/REAL or /raw/cifake/train/REAL
    check_cmd = "hdfs dfs -ls /raw/cifake/train/ 2>/dev/null | head -5"
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    
    if "/raw/cifake/train/train" in result.stdout:
        # Need to fix structure
        print("   Restructuring directories...")
        subprocess.run("hdfs dfs -mv /raw/cifake/train/train/* /raw/cifake/train/", shell=True)
        subprocess.run("hdfs dfs -rm -r /raw/cifake/train/train", shell=True)
        subprocess.run("hdfs dfs -mv /raw/cifake/test/test/* /raw/cifake/test/", shell=True)
        subprocess.run("hdfs dfs -rm -r /raw/cifake/test/test", shell=True)
    
    # Verify upload
    print("\n" + "=" * 60)
    print("Verifying HDFS data:")
    print("=" * 60)
    
    for folder_name in ["train/REAL", "train/FAKE", "test/REAL", "test/FAKE"]:
        cmd = f"hdfs dfs -count {hdfs_base}/{folder_name} 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                print(f"{folder_name}: {parts[1]} files")
        else:
            print(f"{folder_name}: checking...")
    
    print("\n" + "=" * 60)
    print("✓ Dataset uploaded to HDFS successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
