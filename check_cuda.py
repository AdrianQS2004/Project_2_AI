#!/usr/bin/env python
"""Quick CUDA availability check"""

print("Checking for CUDA availability...")
print("=" * 60)

# Method 1: Check via PyTorch
try:
    import torch
    if torch.cuda.is_available():
        print("✓ PyTorch CUDA: YES")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("✗ PyTorch CUDA: NO (PyTorch installed but CUDA not available)")
except ImportError:
    print("✗ PyTorch: NOT INSTALLED")

# Method 2: Check via nvidia-smi
print("\nChecking nvidia-smi...")
try:
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ nvidia-smi: AVAILABLE")
        # Extract GPU name from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line or 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                print(f"  {line.strip()}")
                break
        print("  → CUDA-capable GPU detected")
    else:
        print("✗ nvidia-smi: Command failed")
except FileNotFoundError:
    print("✗ nvidia-smi: NOT FOUND (NVIDIA drivers may not be installed)")
except Exception as e:
    print(f"✗ nvidia-smi: Error - {e}")

# Method 3: Check XGBoost GPU support
print("\nChecking XGBoost GPU support...")
try:
    import xgboost as xgb
    print("✓ XGBoost: INSTALLED")
    # Try to create a simple test to see if GPU is available
    try:
        # This will fail if GPU is not available
        import numpy as np
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        test_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1, verbosity=0)
        test_model.fit(X, y)
        print("  → XGBoost GPU support: WORKING")
    except Exception as e:
        if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
            print("  → XGBoost GPU support: NOT AVAILABLE")
        else:
            print(f"  → XGBoost GPU test: {e}")
except ImportError:
    print("✗ XGBoost: NOT INSTALLED")

# Method 4: Check LightGBM GPU support
print("\nChecking LightGBM GPU support...")
try:
    import lightgbm as lgb
    print("✓ LightGBM: INSTALLED")
    # LightGBM GPU requires OpenCL, harder to test without actual training
    print("  → LightGBM GPU support: Check at runtime (requires OpenCL)")
except ImportError:
    print("✗ LightGBM: NOT INSTALLED")

print("\n" + "=" * 60)
print("Summary: If nvidia-smi works, you have CUDA-capable hardware.")
print("        The script will automatically use GPU if available.")

