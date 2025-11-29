# Results Analysis - All Experiments

## Final Scores Summary

| Method                | ROC AUC     | vs Baseline | vs Previous Best | Rank |
| --------------------- | ----------- | ----------- | ---------------- | ---- |
| **Advanced Ensemble** | **0.92088** | +0.00790    | Best             | 🥇   |
| Combined Ensemble     | 0.92077     | +0.00779    | -0.00011         | 🥈   |
| Initial Ensemble      | 0.92045     | +0.00747    | -0.00043         | 🥉   |
| CatBoost Ensemble     | 0.92016     | +0.00718    | -0.00072         | 4th  |
| Baseline              | 0.91298     | -           | -                | -    |

## Key Findings

### ✅ What Worked Best

1. **Advanced Ensemble (0.92088)** - WINNER

   - Feature engineering was key
   - 5 models (RF, GB, XGB, LGB, Extra Trees)
   - Stacking with meta-learner
   - Lower learning rates (0.03)
   - More regularization

2. **Feature Engineering Impact**
   - Income-to-loan ratio
   - Debt-score interactions
   - Interest rate interactions
   - These features helped Advanced outperform others

### ❌ What Didn't Work

1. **Simple Averaging**

   - Combined (0.92077) < Advanced (0.92088)
   - Averaging diluted strong predictions
   - Should have used weighted average favoring Advanced

2. **CatBoost**
   - Performed worse than expected (0.92016)
   - Native categorical handling didn't help as much
   - Feature engineering was more effective

## Lessons Learned

### 1. Not All Ensembles Benefit from Averaging

- When one method is clearly superior, averaging can hurt
- Simple average assumes equal quality - not always true
- Should use weighted average based on validation performance

### 2. Feature Engineering > Model Choice

- Advanced ensemble's feature engineering was key
- Interaction features captured important patterns
- Better features > better models (in this case)

### 3. Stacking Can Help

- Advanced ensemble used stacking meta-learner
- This likely contributed to its success
- Meta-learner learned how to combine base models

### 4. More Models ≠ Always Better

- Advanced had 5 models, CatBoost had 4
- But Advanced's feature engineering mattered more
- Quality of models > quantity

## Recommendations

### For Final Submission

**Use: `my_submission_advanced.csv` (0.92088)**

### If Wanting to Improve Further

1. **Weighted Combination** (not simple average)

   - Weight Advanced more heavily (e.g., 0.7-0.8)
   - Weight CatBoost less (e.g., 0.2-0.3)
   - Optimize weights on validation set

2. **Better Feature Engineering**

   - More interaction features
   - Polynomial features
   - Domain-specific features

3. **Hyperparameter Tuning**

   - Fine-tune Advanced ensemble further
   - Try different learning rates
   - Adjust regularization

4. **Cross-Validation**
   - Use K-fold CV for more robust evaluation
   - Out-of-fold predictions
   - Reduce overfitting risk

## Score Progression

```
Baseline:     0.91298
    ↓ +0.00747
Ensemble:     0.92045
    ↓ +0.00043
Advanced:     0.92088 ⭐ BEST
    ↓ -0.00011
Combined:     0.92077
    ↓ -0.00072
CatBoost:     0.92016
```

## Conclusion

**Best Method: Advanced Ensemble (0.92088)**

- Feature engineering was the key differentiator
- 5-model ensemble with stacking
- Well-tuned hyperparameters
- **Total improvement: +0.00790 from baseline (+0.87% relative)**

The Advanced ensemble's feature engineering approach proved more effective than CatBoost's native categorical handling for this dataset.
