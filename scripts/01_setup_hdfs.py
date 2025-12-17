#!/usr/bin/env python3
"""
Script 01: Setup HDFS directories
Tạo các thư mục cần thiết trên HDFS
"""

import subprocess
import sys

def run_hdfs_cmd(cmd):
    """Run HDFS command"""
    full_cmd = f"hdfs dfs {cmd}"
    print(f"Running: {full_cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")
    return result.returncode

def main():
    print("=" * 60)
    print("STEP 1: Creating HDFS directories")
    print("=" * 60)
    
    # Tạo các thư mục HDFS
    directories = [
        "/raw",
        "/raw/cifake",
        "/raw/cifake/train",
        "/raw/cifake/train/REAL",
        "/raw/cifake/train/FAKE", 
        "/raw/cifake/test",
        "/raw/cifake/test/REAL",
        "/raw/cifake/test/FAKE",
        "/processed",
        "/results",
        "/spark-logs"
    ]
    
    for dir_path in directories:
        run_hdfs_cmd(f"-mkdir -p {dir_path}")
    
    # Set permissions for spark-logs
    run_hdfs_cmd("-chmod 777 /spark-logs")
    
    print("\n" + "=" * 60)
    print("HDFS directories created successfully!")
    print("=" * 60)
    
    # List directories
    print("\nHDFS structure:")
    run_hdfs_cmd("-ls -R /")

if __name__ == "__main__":
    main()
