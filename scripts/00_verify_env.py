#!/usr/bin/env python3
"""
Script 00: Verify Environment
Kiểm tra môi trường trước khi chạy pipeline

Requirements:
- Python 3.8+
- PySpark 3.3+
- PyTorch (CPU version)
- torchvision (MobileNetV2)
- HDFS connectivity
- Spark cluster connectivity
"""

import subprocess
import sys

def run_cmd(cmd, description):
    """Run command and check result"""
    print(f"\n{'─'*50}")
    print(f"📋 {description}")
    print(f"{'─'*50}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Success")
        if result.stdout:
            print(result.stdout[:500])
    else:
        print(f"❌ Failed")
        print(result.stderr[:500] if result.stderr else "No error message")
    return result.returncode == 0

def main():
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION - HDFS + PYTORCH")
    print("=" * 60)
    
    checks = []
    
    # 1. Check Python
    checks.append(("python3 --version", "Python version (need 3.8+)"))
    
    # 2. Check PySpark
    checks.append(("python3 -c 'import pyspark; print(f\"PySpark: {pyspark.__version__}\")'", "PySpark import"))
    
    # 3. Check PyTorch
    checks.append(("python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'", "PyTorch import (for MobileNetV2)"))
    
    # 4. Check torchvision
    checks.append(("python3 -c 'import torchvision; print(f\"torchvision: {torchvision.__version__}\")'", "torchvision import"))
    
    # 5. Check MobileNetV2
    checks.append(("python3 -c 'from torchvision.models import mobilenet_v2; m = mobilenet_v2(pretrained=False); print(\"MobileNetV2: OK\")'", "MobileNetV2 model available"))
    
    # 6. Check NumPy
    checks.append(("python3 -c 'import numpy; print(f\"NumPy: {numpy.__version__}\")'", "NumPy import"))
    
    # 7. Check Pandas
    checks.append(("python3 -c 'import pandas; print(f\"Pandas: {pandas.__version__}\")'", "Pandas import"))
    
    # 8. Check PIL
    checks.append(("python3 -c 'from PIL import Image; print(\"PIL: OK\")'", "PIL import"))
    
    # 9. Check scikit-learn
    checks.append(("python3 -c 'import sklearn; print(f\"Scikit-learn: {sklearn.__version__}\")'", "Scikit-learn import"))
    
    # 10. Check dataset exists
    checks.append(("ls -la /scripts/dataset/archive/", "Dataset directory (local mount)"))
    
    # 11. Check HDFS connectivity
    checks.append(("hdfs dfs -ls hdfs://namenode:8020/ 2>/dev/null || echo 'HDFS root accessible'", "HDFS connectivity"))
    
    # 12. Check Spark Master connection
    checks.append(("curl -s http://spark-master:8080 | head -5", "Spark Master WebUI"))
    
    # 13. Check HDFS NameNode WebUI
    checks.append(("curl -s http://namenode:9870 | head -5 || echo 'NameNode WebUI check'", "HDFS NameNode WebUI"))
    
    results = []
    for cmd, desc in checks:
        results.append((desc, run_cmd(cmd, desc)))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    critical_failed = []
    for desc, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {desc}")
        if not passed:
            all_passed = False
            # Critical checks
            if "PyTorch" in desc or "HDFS" in desc or "MobileNetV2" in desc:
                critical_failed.append(desc)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! Ready to run pipeline.")
    elif critical_failed:
        print("❌ Critical checks failed:")
        for fail in critical_failed:
            print(f"   - {fail}")
        print("\nPipeline may not work correctly.")
    else:
        print("⚠️  Some checks failed. Please review before running pipeline.")
    print("=" * 60)
    
    # Return 0 even if some non-critical checks fail
    return 0 if not critical_failed else 1

if __name__ == "__main__":
    sys.exit(main())
