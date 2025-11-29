# Weighted Ensemble - Give More Weight to Better Performing Models
# Instead of simple average, weight Advanced ensemble more heavily

import pandas as pd
import numpy as np

print("="*80)
print("WEIGHTED ENSEMBLE - Optimizing Weights")
print("="*80)

# Load the two ensemble predictions
print("\nLoading predictions...")
advanced = pd.read_csv('my_submission_advanced.csv')
catboost = pd.read_csv('my_submission_catboost.csv')

# Verify they have the same IDs
assert (advanced['id'] == catboost['id']).all(), "IDs don't match!"

ids = advanced['id'].values
adv_preds = advanced['loan_paid_back'].values
cat_preds = catboost['loan_paid_back'].values

print(f"✓ Loaded {len(ids)} predictions")
print(f"  Advanced mean: {adv_preds.mean():.6f}")
print(f"  CatBoost mean: {cat_preds.mean():.6f}")

# Try different weight combinations
# Advanced performed better (0.92088 vs 0.92016), so weight it more
weight_combinations = [
    (0.5, 0.5, "Equal weights"),
    (0.6, 0.4, "Advanced 60%, CatBoost 40%"),
    (0.7, 0.3, "Advanced 70%, CatBoost 30%"),
    (0.8, 0.2, "Advanced 80%, CatBoost 20%"),
    (0.9, 0.1, "Advanced 90%, CatBoost 10%"),
    (0.95, 0.05, "Advanced 95%, CatBoost 5%"),
    (1.0, 0.0, "Advanced only (baseline)"),
]

print("\n" + "="*80)
print("Testing Weight Combinations")
print("="*80)

results = []
for w_adv, w_cat, desc in weight_combinations:
    combined = w_adv * adv_preds + w_cat * cat_preds
    
    # Save this combination
    submission = pd.DataFrame({
        'id': ids,
        'loan_paid_back': combined
    })
    
    filename = f'submissions/my_submission_weighted_{int(w_adv*100)}.csv'
    submission.to_csv(filename, index=False)
    
    results.append({
        'weights': (w_adv, w_cat),
        'description': desc,
        'mean': combined.mean(),
        'std': combined.std(),
        'file': filename
    })
    
    print(f"\n{desc}:")
    print(f"  Weights: Advanced={w_adv:.2f}, CatBoost={w_cat:.2f}")
    print(f"  Mean: {combined.mean():.6f}")
    print(f"  Std: {combined.std():.6f}")
    print(f"  Saved: {filename}")

# Recommend based on performance difference
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nSince Advanced (0.92088) > CatBoost (0.92016):")
print("  → Try higher weights for Advanced (70-90%)")
print("\nGenerated files (in submissions/ folder):")
for r in results:
    if r['weights'][0] >= 0.7:
        print(f"  - {r['file']} ({r['description']})")

print("\n" + "="*80)
print("SUBMIT THESE FILES TO TEST:")
print("="*80)
print("1. submissions/my_submission_weighted_80.csv (Advanced 80%, CatBoost 20%)")
print("2. submissions/my_submission_weighted_90.csv (Advanced 90%, CatBoost 10%)")
print("3. submissions/my_submission_weighted_95.csv (Advanced 95%, CatBoost 5%)")
print("\nExpected: Slightly better than 0.92088 (maybe +0.0001 to +0.0003)")
print("="*80 + "\n")

