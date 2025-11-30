# Combine Optimized Ensemble (0.9217) with CatBoost Ensemble
# Quick win: Two-stage ensemble (ensemble of ensembles)
# Expected: 0.9218-0.9220

import pandas as pd
import numpy as np
import sklearn.metrics

print("="*80)
print("COMBINING OPTIMIZED ENSEMBLE WITH CATBOOST")
print("="*80)

# Load optimized ensemble predictions
print("\n[1/3] Loading optimized ensemble predictions...")
try:
    optimized = pd.read_csv('submissions/my_submission_9217.csv')
    print(f"  ✓ Loaded: my_submission_9217.csv")
    print(f"  Score: 0.9217 (from Mac, 20 trials for all 3 models)")
except FileNotFoundError:
    try:
        optimized = pd.read_csv('submissions/my_submission_optimized50trials.csv')
        print(f"  ✓ Loaded: my_submission_optimized50trials.csv")
        print(f"  Score: 0.92175 (from PC, 50 trials)")
    except FileNotFoundError:
        try:
            optimized = pd.read_csv('submissions/my_submission_optimized20trials.csv')
            print(f"  ✓ Loaded: my_submission_optimized20trials.csv")
            print(f"  Score: ~0.92057 (from Mac run)")
        except FileNotFoundError:
            print("  ✗ Error: Could not find optimized submission file")
            print("  Expected: submissions/my_submission_9217.csv")
            print("  Or: submissions/my_submission_optimized50trials.csv")
            print("  Or: submissions/my_submission_optimized20trials.csv")
            exit(1)

# Load CatBoost ensemble predictions
print("\n[2/3] Loading CatBoost ensemble predictions...")
try:
    catboost = pd.read_csv('submissions/my_submission_catboost.csv')
    print(f"  ✓ Loaded: my_submission_catboost.csv")
    print(f"  Score: 0.92016 (CatBoost ensemble)")
except FileNotFoundError:
    print("  ✗ Error: Could not find CatBoost submission file")
    print("  Expected: submissions/my_submission_catboost.csv")
    print("  Run ensemble_catboost.py first if needed")
    exit(1)

# Verify IDs match
assert (optimized['id'] == catboost['id']).all(), "IDs don't match!"

opt_preds = optimized['loan_paid_back'].values
cat_preds = catboost['loan_paid_back'].values

print(f"\n  Optimized mean: {opt_preds.mean():.6f}")
print(f"  CatBoost mean: {cat_preds.mean():.6f}")

# Try different weight combinations
print("\n[3/3] Testing different weight combinations...")
print("-"*80)

weight_combinations = [
    (0.85, 0.15, "Optimized 85%, CatBoost 15%"),
    (0.90, 0.10, "Optimized 90%, CatBoost 10%"),
    (0.95, 0.05, "Optimized 95%, CatBoost 5%"),
    (0.80, 0.20, "Optimized 80%, CatBoost 20%"),
    (0.75, 0.25, "Optimized 75%, CatBoost 25%"),
]

results = []

for w_opt, w_cat, description in weight_combinations:
    combined = w_opt * opt_preds + w_cat * cat_preds
    
    submission = pd.DataFrame({
        'id': optimized['id'],
        'loan_paid_back': combined
    })
    
    filename = f'submissions/my_submission_optimized{int(w_opt*100)}_catboost{int(w_cat*100)}.csv'
    submission.to_csv(filename, index=False)
    
    results.append({
        'weights': f"{int(w_opt*100)}/{int(w_cat*100)}",
        'description': description,
        'filename': filename,
        'mean': combined.mean()
    })
    
    print(f"  {description}:")
    print(f"    Saved: {filename}")
    print(f"    Mean prediction: {combined.mean():.6f}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("Since Optimized (0.9217) > CatBoost (0.92016):")
print("  - Give more weight to Optimized ensemble")
print("  - Try 85/15, 90/10, or 95/5 combinations")
print("\nGenerated files:")
for r in results:
    print(f"  - {r['filename']}")
print("\nNext: Submit these files and test which weight combination works best!")
print("Expected: 0.9218-0.9220 (improvement over 0.9217)")

